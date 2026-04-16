package blockio

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

// Tracer attaches to block I/O tracepoints and emits events.
type Tracer struct {
	ringBufSize uint32 // 0 = use compiled default
	objs        ioTraceObjects
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

// WithRingBufSize overrides the compiled-in ring buffer size (default 1MB).
// The value must be a power of 2 and at least 4096 (one page).
func WithRingBufSize(bytes uint32) Option {
	return func(t *Tracer) {
		t.ringBufSize = bytes
	}
}

// New creates a block I/O tracer. Call Attach() to start.
func New(opts ...Option) *Tracer {
	t := &Tracer{
		eventCh: make(chan events.Event, 4096),
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// Attach loads the eBPF programs, attaches tracepoints, and creates the ring
// buffer reader. bpf2go's LoadAndAssign loads programs into the kernel but does
// NOT attach them — explicit link.Tracepoint calls are required.
func (t *Tracer) Attach() error {
	closeFn := func() {
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

	spec, err := loadIoTrace()
	if err != nil {
		return fmt.Errorf("loading I/O trace spec: %w", err)
	}

	if t.ringBufSize > 0 {
		if eventsMap, ok := spec.Maps["io_events"]; ok {
			eventsMap.MaxEntries = t.ringBufSize
		}
	}

	if err := spec.LoadAndAssign(&t.objs, nil); err != nil {
		return fmt.Errorf("loading I/O trace objects: %w", err)
	}

	// Attach tracepoints — programs are loaded but dormant until linked.
	type tpSpec struct {
		group string
		name  string
		prog  *ebpf.Program
	}
	specs := []tpSpec{
		{"block", "block_rq_issue", t.objs.HandleBlockRqIssue},
		{"block", "block_rq_complete", t.objs.HandleBlockRqComplete},
	}
	for _, s := range specs {
		tp, err := link.Tracepoint(s.group, s.name, s.prog, nil)
		if err != nil {
			return fmt.Errorf("attaching tracepoint %s/%s: %w", s.group, s.name, err)
		}
		t.links = append(t.links, tp)
	}

	t.reader, err = ringbuf.NewReader(t.objs.IoEvents)
	if err != nil {
		return fmt.Errorf("creating I/O ring buffer reader: %w", err)
	}

	closeFn = nil
	return nil
}

// Events returns the read-only event channel.
func (t *Tracer) Events() <-chan events.Event {
	return t.eventCh
}

// Run reads events from the ring buffer until ctx is cancelled.
// Call in a goroutine.
func (t *Tracer) Run(ctx context.Context) {
	defer close(t.eventCh)

	// Close reader when context signals done.
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

		evt, ok := t.parseEvent(record.RawSample)
		if !ok {
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

// parseEvent converts raw ring buffer bytes into an events.Event.
func (t *Tracer) parseEvent(raw []byte) (events.Event, bool) {
	expected := int(unsafe.Sizeof(ioTraceIngeroIoEvent{}))
	if len(raw) < expected {
		return events.Event{}, false
	}

	e := (*ioTraceIngeroIoEvent)(unsafe.Pointer(&raw[0]))

	return events.Event{
		Timestamp: events.KtimeToWallClock(e.Hdr.TimestampNs),
		PID:       e.Hdr.Pid,
		TID:       e.Hdr.Tid,
		Comm:      events.CommToString(e.Hdr.Comm),
		Source:    events.SourceIO,
		Op:        e.Hdr.Op,
		Duration:  time.Duration(e.DurationNs),
		Args:      [2]uint64{uint64(e.NrSector), e.Sector},
		GPUID:     0, // dev number not stored; gpu_id is reserved for GPU device index
		CGroupID:  e.Hdr.CgroupId,
	}, true
}

// Close releases all resources.
func (t *Tracer) Close() error {
	if t.closed.Swap(true) {
		return nil
	}
	var errs []error
	if t.reader != nil {
		errs = append(errs, t.reader.Close())
	}
	for _, l := range t.links {
		errs = append(errs, l.Close())
	}
	errs = append(errs, t.objs.Close())
	return errors.Join(errs...)
}

// Dropped returns the number of events dropped due to a full channel.
func (t *Tracer) Dropped() uint64 { return t.dropped.Load() }

// ReadErrors returns the number of ring buffer read errors.
func (t *Tracer) ReadErrors() uint64 { return t.readErrors.Load() }

// ParseErrors returns the number of event parse failures.
func (t *Tracer) ParseErrors() uint64 { return t.parseErrors.Load() }
