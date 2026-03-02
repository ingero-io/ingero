package net

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"unsafe"

	"github.com/cilium/ebpf/ringbuf"

	"github.com/ingero-io/ingero/pkg/events"
)

// Tracer attaches to network socket syscall tracepoints and emits events.
type Tracer struct {
	objs        netTraceObjects
	reader      *ringbuf.Reader
	eventCh     chan events.Event
	dropped     atomic.Uint64
	readErrors  atomic.Uint64
	parseErrors atomic.Uint64
	closed      atomic.Bool
}

// New creates a network socket tracer. Call Attach() to start.
func New() *Tracer {
	return &Tracer{
		eventCh: make(chan events.Event, 4096),
	}
}

// Attach loads the eBPF program and attaches tracepoints.
func (t *Tracer) Attach() error {
	closeFn := func() {
		t.objs.Close()
	}
	defer func() {
		if closeFn != nil {
			closeFn()
		}
	}()

	if err := loadNetTraceObjects(&t.objs, nil); err != nil {
		return fmt.Errorf("loading net trace objects: %w", err)
	}

	var err error
	t.reader, err = ringbuf.NewReader(t.objs.NetEvents)
	if err != nil {
		return fmt.Errorf("creating net ring buffer reader: %w", err)
	}

	closeFn = nil
	return nil
}

// SetTargetPID adds a PID to the filter map.
func (t *Tracer) SetTargetPID(pid uint32) error {
	val := uint8(1)
	return t.objs.NetTargetPids.Put(pid, val)
}

// Events returns the read-only event channel.
func (t *Tracer) Events() <-chan events.Event {
	return t.eventCh
}

// Run reads events until ctx signals done.
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

// parseEvent converts raw bytes into an events.Event.
// Args[0] = fd, Args[1] = bytes transferred
func (t *Tracer) parseEvent(raw []byte) (events.Event, bool) {
	expected := int(unsafe.Sizeof(netTraceIngeroNetEvent{}))
	if len(raw) < expected {
		return events.Event{}, false
	}

	e := (*netTraceIngeroNetEvent)(unsafe.Pointer(&raw[0]))

	return events.Event{
		Timestamp: events.KtimeToWallClock(e.Hdr.TimestampNs),
		PID:       e.Hdr.Pid,
		TID:       e.Hdr.Tid,
		Source:    events.SourceNet,
		Op:        e.Hdr.Op,
		Args:      [2]uint64{uint64(e.Fd), uint64(e.Bytes)},
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
