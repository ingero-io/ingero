package cudagraph

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"reflect"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
	"github.com/ingero-io/ingero/pkg/events"
)

// Tracer attaches eBPF uprobes to CUDA Graph lifecycle functions and streams
// events to a Go channel.
type Tracer struct {
	libPath     string
	ringBufSize uint32 // 0 = use compiled default
	objs        cudaGraphTraceObjects
	// configMap is resolved from the loaded collection — nil if the
	// current BPF object predates graph_config_map (pre `make generate`).
	// Populated in Attach() so SetSamplingRate works both with the current
	// bindings and with future bpf2go-regenerated bindings.
	configMap   *ebpf.Map
	links       []link.Link
	reader      *ringbuf.Reader
	eventCh     chan events.Event
	dropped     atomic.Uint64
	readErrors  atomic.Uint64
	parseErrors atomic.Uint64
	closed      atomic.Bool
}

// Option configures a Tracer.
type Option func(*Tracer)

// WithRingBufSize overrides the compiled-in ring buffer size (default 4MB).
// The value must be a power of 2 and at least 4096 (one page).
func WithRingBufSize(bytes uint32) Option {
	return func(t *Tracer) {
		t.ringBufSize = bytes
	}
}

// probeSpec defines a uprobe/uretprobe pair to attach.
type probeSpec struct {
	symbol    string
	uprobe    *ebpf.Program
	uretprobe *ebpf.Program
}

// New creates a new CUDA Graph tracer for the given libcudart.so path.
func New(libcudartPath string, opts ...Option) *Tracer {
	t := &Tracer{
		libPath: libcudartPath,
		eventCh: make(chan events.Event, 4096),
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// Attach loads the eBPF program and attaches uprobes to CUDA Graph functions.
// All symbols are best-effort: if a symbol is not found in libcudart.so, it is
// skipped with a warning (graceful degradation per NFR28).
func (t *Tracer) Attach() error {
	spec, err := loadCudaGraphTrace()
	if err != nil {
		return fmt.Errorf("loading graph eBPF spec: %w", err)
	}

	if t.ringBufSize > 0 {
		if eventsMap, ok := spec.Maps["graph_events"]; ok {
			eventsMap.MaxEntries = t.ringBufSize
		}
	}

	if err := spec.LoadAndAssign(&t.objs, nil); err != nil {
		return fmt.Errorf("loading graph eBPF objects: %w", err)
	}

	// Wire configMap for sampling when the regenerated bindings expose it.
	// Pre-regeneration this stays nil and SetSamplingRate is a graceful no-op.
	t.configMap = t.resolveConfigMap()

	var closeFn func()
	closeFn = func() {
		for _, l := range t.links {
			l.Close()
		}
		t.links = nil
		t.objs.Close()
	}
	defer func() {
		if closeFn != nil {
			closeFn()
		}
	}()

	exe, err := link.OpenExecutable(t.libPath)
	if err != nil {
		return fmt.Errorf("opening %s: %w", t.libPath, err)
	}

	// All graph probes are best-effort — skip gracefully if symbol not found (NFR28).
	specs := []probeSpec{
		{"cudaStreamBeginCapture", t.objs.UprobeGraphBeginCapture, t.objs.UretprobeGraphBeginCapture},
		{"cudaStreamEndCapture", t.objs.UprobeGraphEndCapture, t.objs.UretprobeGraphEndCapture},
		{"cudaGraphInstantiate", t.objs.UprobeGraphInstantiate, t.objs.UretprobeGraphInstantiate},
		{"cudaGraphLaunch", t.objs.UprobeGraphLaunch, t.objs.UretprobeGraphLaunch},
	}

	for _, spec := range specs {
		up, err := exe.Uprobe(spec.symbol, spec.uprobe, nil)
		if err != nil {
			log.Printf("INFO: graph probes: %s not found, skipping", spec.symbol)
			continue
		}
		uret, err := exe.Uretprobe(spec.symbol, spec.uretprobe, nil)
		if err != nil {
			up.Close()
			log.Printf("INFO: graph probes: %s uretprobe failed, skipping", spec.symbol)
			continue
		}
		t.links = append(t.links, up, uret)
	}

	// Create ring buffer reader.
	t.reader, err = ringbuf.NewReader(t.objs.GraphEvents)
	if err != nil {
		return fmt.Errorf("creating graph ring buffer reader: %w", err)
	}

	closeFn = nil
	return nil
}

// Events returns a read-only channel on which parsed graph events are delivered.
func (t *Tracer) Events() <-chan events.Event {
	return t.eventCh
}

// Run starts reading events from the eBPF ring buffer and sending them
// to the Events() channel. Blocks until ctx is cancelled.
func (t *Tracer) Run(ctx context.Context) {
	defer close(t.eventCh)

	go func() {
		<-ctx.Done()
		t.reader.Close()
	}()

	for {
		record, err := t.reader.Read()
		if err != nil {
			if errors.Is(err, ringbuf.ErrClosed) {
				return
			}
			t.readErrors.Add(1)
			continue
		}

		evt, err := parseGraphEvent(record.RawSample)
		if err != nil {
			t.parseErrors.Add(1)
			continue
		}

		select {
		case t.eventCh <- evt:
		default:
			t.dropped.Add(1)
		}
	}
}

// Close releases all eBPF resources. Safe to call multiple times.
func (t *Tracer) Close() error {
	if t.closed.Swap(true) {
		return nil
	}

	var errs []error
	if t.reader != nil {
		if err := t.reader.Close(); err != nil && !errors.Is(err, ringbuf.ErrClosed) {
			errs = append(errs, err)
		}
	}
	for _, l := range t.links {
		if err := l.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	t.links = nil
	if err := t.objs.Close(); err != nil {
		errs = append(errs, fmt.Errorf("closing graph eBPF objects: %w", err))
	}
	return errors.Join(errs...)
}

// ProbeCount returns the number of attached probes.
func (t *Tracer) ProbeCount() int {
	return len(t.links)
}

// Dropped returns the number of events dropped due to a full channel.
func (t *Tracer) Dropped() uint64 {
	return t.dropped.Load()
}

// resolveConfigMap returns the graph_config_map from t.objs by reflection,
// traversing the embedded cudaGraphTraceMaps struct. Returns nil when the
// field is absent from the current bpf2go-generated bindings (pre `make
// generate` regeneration after the B3-4 BPF changes). Using reflection
// lets the Go-side wiring ship ahead of the regenerated bindings without
// breaking the build — once bpf2go emits `GraphConfigMap *ebpf.Map`, this
// picks it up automatically.
func (t *Tracer) resolveConfigMap() *ebpf.Map {
	v := reflect.ValueOf(&t.objs).Elem()
	f := v.FieldByName("GraphConfigMap")
	if !f.IsValid() || f.IsZero() {
		return nil
	}
	m, ok := f.Interface().(*ebpf.Map)
	if !ok {
		return nil
	}
	return m
}

// SetSamplingRate updates the BPF sampling rate for this tracer.
// Rate 0 or 1 = emit all events. Rate N > 1 = emit 1 in N events
// (per-CPU, so actual ratio varies slightly).
//
// Must be called after Attach(). Thread-safe.
//
// The graph tracer's config map is resolved dynamically from the loaded
// spec — the bpf2go-generated typed bindings are regenerated on Linux via
// `make generate` after a BPF C change. Until regeneration, this returns
// without error (sampling simply stays at the BPF default of "emit all").
func (t *Tracer) SetSamplingRate(rate uint32) error {
	if t.configMap == nil {
		// Config map not present in current BPF object — graceful no-op.
		// After `make generate` runs on Linux, loadCudaGraphTrace will
		// return a spec containing graph_config_map and this will wire up.
		return nil
	}
	// Build the full 12-byte ingero_config struct (u32 alignment, no trailing pad).
	// Graph tracer has no stack-capture option; capture_stack stays 0.
	var cfg [12]byte
	// sampling_rate at offset 4 (little-endian uint32)
	binary.LittleEndian.PutUint32(cfg[4:8], rate)
	if err := t.configMap.Put(uint32(0), cfg[:]); err != nil {
		return fmt.Errorf("updating graph sampling rate: %w", err)
	}
	return nil
}

// parseGraphEvent converts raw bytes from the graph ring buffer into a typed Event.
// Size is derived from the bpf2go-generated struct so it auto-updates when the
// BPF event header grows (e.g. v0.10 added comm[16], shifting from 72 to 88 bytes).
func parseGraphEvent(raw []byte) (events.Event, error) {
	eventSize := int(unsafe.Sizeof(cudaGraphTraceCudaGraphEvent{}))
	if len(raw) < eventSize {
		return events.Event{}, fmt.Errorf("graph event too short: %d bytes, need %d", len(raw), eventSize)
	}

	ge := (*cudaGraphTraceCudaGraphEvent)(unsafe.Pointer(&raw[0]))

	return events.Event{
		Timestamp:    events.KtimeToWallClock(ge.Hdr.TimestampNs),
		PID:          ge.Hdr.Pid,
		TID:          ge.Hdr.Tid,
		Comm:         events.CommToString(ge.Hdr.Comm),
		Source:       events.Source(ge.Hdr.Source),
		Op:           ge.Hdr.Op,
		Duration:     time.Duration(ge.DurationNs),
		RetCode:      ge.ReturnCode,
		CGroupID:     ge.Hdr.CgroupId,
		StreamHandle: ge.StreamHandle,
		GraphHandle:  ge.GraphHandle,
		ExecHandle:   ge.ExecHandle,
		CaptureMode:  ge.CaptureMode,
	}, nil
}
