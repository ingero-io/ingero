package driver

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
	"github.com/ingero-io/ingero/pkg/events"
)

// Tracer attaches eBPF uprobes to CUDA Driver API functions (libcuda.so) and
// streams events to a Go channel. Mirrors the cuda.Tracer lifecycle.
type Tracer struct {
	libPath      string
	stackEnabled bool
	objs         driverTraceObjects
	links        []link.Link
	reader       *ringbuf.Reader
	eventCh      chan events.Event
	dropped      atomic.Uint64
	readErrors   atomic.Uint64  // ring buffer read errors
	parseErrors  atomic.Uint64  // event parse failures
	closed       atomic.Bool    // prevents double-close
}

// probeSpec defines a uprobe/uretprobe pair for a driver API function.
// symbols is a list of symbol names to try (for _v2/_v3 variants).
type probeSpec struct {
	symbols   []string
	uprobe    *ebpf.Program
	uretprobe *ebpf.Program
}

// Option configures a Tracer.
type Option func(*Tracer)

// WithStackCapture enables userspace stack trace capture in eBPF.
func WithStackCapture(enabled bool) Option {
	return func(t *Tracer) {
		t.stackEnabled = enabled
	}
}

// New creates a new driver API tracer for the given libcuda.so path.
func New(libcudaPath string, opts ...Option) *Tracer {
	t := &Tracer{
		libPath: libcudaPath,
		eventCh: make(chan events.Event, 4096),
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// Attach loads the eBPF program and attaches uprobes to driver API functions.
func (t *Tracer) Attach() error {
	if err := loadDriverTraceObjects(&t.objs, nil); err != nil {
		return fmt.Errorf("loading driver eBPF objects: %w", err)
	}

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

	// Define probes. Symbol names must match those exported by libcuda.so.
	// Some functions have _v2/_v3/_ptsz variants — try each until one works.
	// Symbol fallback order: base name first, then _v2/_v3/_ptsz variants.
	// NVIDIA driver API uses versioned symbols — _v2 for CUDA 11+, _ptsz for
	// per-thread default stream (Hopper+). We try each until one attaches.
	specs := []probeSpec{
		{[]string{"cuLaunchKernel", "cuLaunchKernel_ptsz"}, t.objs.UprobeCuLaunchKernel, t.objs.UretprobeCuLaunchKernel},
		{[]string{"cuMemcpy", "cuMemcpy_v2", "cuMemcpy_ptsz"}, t.objs.UprobeCuMemcpy, t.objs.UretprobeCuMemcpy},
		{[]string{"cuMemcpyAsync", "cuMemcpyAsync_v2", "cuMemcpyAsync_ptsz"}, t.objs.UprobeCuMemcpyAsync, t.objs.UretprobeCuMemcpyAsync},
		{[]string{"cuCtxSynchronize", "cuCtxSynchronize_v2", "cuCtxSynchronize_ptsz"}, t.objs.UprobeCuCtxSync, t.objs.UretprobeCuCtxSync},
		{[]string{"cuMemAlloc_v2", "cuMemAlloc_v3", "cuMemAlloc"}, t.objs.UprobeCuMemAlloc, t.objs.UretprobeCuMemAlloc},
	}

	for _, spec := range specs {
		attached := false
		for _, sym := range spec.symbols {
			up, err := exe.Uprobe(sym, spec.uprobe, nil)
			if err != nil {
				continue
			}
			t.links = append(t.links, up)

			uret, err := exe.Uretprobe(sym, spec.uretprobe, nil)
			if err != nil {
				up.Close()
				t.links = t.links[:len(t.links)-1]
				continue
			}
			t.links = append(t.links, uret)
			attached = true
			break
		}
		if !attached {
			// Non-fatal: some symbols may not exist on all driver versions.
			continue
		}
	}

	if len(t.links) == 0 {
		return fmt.Errorf("no driver API symbols found in %s", t.libPath)
	}

	// Enable stack capture in the eBPF config map if requested.
	if t.stackEnabled && t.objs.DriverConfigMap != nil {
		cfg := [8]byte{1}
		if err := t.objs.DriverConfigMap.Put(uint32(0), cfg[:]); err != nil {
			return fmt.Errorf("enabling driver stack capture: %w", err)
		}
	}

	// Create ring buffer reader.
	t.reader, err = ringbuf.NewReader(t.objs.DriverEvents)
	if err != nil {
		return fmt.Errorf("creating driver ring buffer reader: %w", err)
	}

	closeFn = nil
	return nil
}

// Events returns the channel on which parsed driver events are delivered.
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

		evt, err := parseEvent(record.RawSample)
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

// stackEventSize is sizeof(cuda_event_stack) = 576 bytes.
const stackEventSize = 576

// parseEvent converts raw bytes from the ring buffer into a typed Event.
// Handles both base events (56 bytes) and stack events (576 bytes).
func parseEvent(raw []byte) (events.Event, error) {
	baseSize := int(unsafe.Sizeof(driverTraceCudaEvent{}))
	if len(raw) < baseSize {
		return events.Event{}, fmt.Errorf("driver event too short: %d bytes, need %d", len(raw), baseSize)
	}

	ce := (*driverTraceCudaEvent)(unsafe.Pointer(&raw[0]))

	evt := events.Event{
		Timestamp: events.KtimeToWallClock(ce.Hdr.TimestampNs),
		PID:       ce.Hdr.Pid,
		TID:       ce.Hdr.Tid,
		Source:    events.Source(ce.Hdr.Source),
		Op:        ce.Hdr.Op,
		Duration:  time.Duration(ce.DurationNs),
		GPUID:     ce.GpuId,
		Args:      [2]uint64{ce.Arg0, ce.Arg1},
		RetCode:   ce.ReturnCode,
	}

	// Check for stack event (576 bytes).
	if len(raw) >= stackEventSize {
		evt.Stack = events.ParseStackIPs(raw, baseSize)
	}

	return evt, nil
}

// Close releases all eBPF resources. Safe to call multiple times (idempotent).
func (t *Tracer) Close() error {
	if t.closed.Swap(true) {
		return nil
	}

	var errs []error
	if t.reader != nil {
		if err := t.reader.Close(); err != nil {
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
		errs = append(errs, fmt.Errorf("closing driver eBPF objects: %w", err))
	}
	return errors.Join(errs...)
}

// LibPath returns the path to the libcuda.so being traced.
func (t *Tracer) LibPath() string {
	return t.libPath
}

// ProbeCount returns the number of attached probes.
func (t *Tracer) ProbeCount() int {
	return len(t.links)
}

// Dropped returns the number of events dropped due to a full channel.
func (t *Tracer) Dropped() uint64 {
	return t.dropped.Load()
}

// ReadErrors returns the number of ring buffer read errors.
func (t *Tracer) ReadErrors() uint64 {
	return t.readErrors.Load()
}

// ParseErrors returns the number of malformed events that failed to parse.
func (t *Tracer) ParseErrors() uint64 {
	return t.parseErrors.Load()
}
