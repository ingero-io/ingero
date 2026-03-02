package blockio

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/cilium/ebpf/ringbuf"

	"github.com/ingero-io/ingero/pkg/events"
)

// Tracer attaches to block I/O tracepoints and emits events.
type Tracer struct {
	objs        ioTraceObjects
	reader      *ringbuf.Reader
	eventCh     chan events.Event
	dropped     atomic.Uint64
	readErrors  atomic.Uint64
	parseErrors atomic.Uint64
	closed      atomic.Bool
}

// New creates a block I/O tracer. Call Attach() to start.
func New() *Tracer {
	return &Tracer{
		eventCh: make(chan events.Event, 4096),
	}
}

// Attach loads the eBPF program and attaches tracepoints.
// tp_btf tracepoints auto-attach via the bpf2go loader.
func (t *Tracer) Attach() error {
	// closeFn cleans up on partial failure. Set to nil on success.
	closeFn := func() {
		t.objs.Close()
	}
	defer func() {
		if closeFn != nil {
			closeFn()
		}
	}()

	// Load and auto-attach all programs.
	// tp_btf programs are attached during loading by the kernel.
	if err := loadIoTraceObjects(&t.objs, nil); err != nil {
		return fmt.Errorf("loading I/O trace objects: %w", err)
	}

	// Create ring buffer reader.
	var err error
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
		Source:    events.SourceIO,
		Op:        e.Hdr.Op,
		Duration:  time.Duration(e.DurationNs),
		Args:      [2]uint64{uint64(e.NrSector), e.Sector},
		GPUID:     e.Dev,
		CGroupID:  e.Hdr.CgroupId,
	}, true
}

// Close releases all resources.
func (t *Tracer) Close() error {
	if !t.closed.Swap(true) {
		return nil
	}
	var errs []error
	if t.reader != nil {
		errs = append(errs, t.reader.Close())
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
