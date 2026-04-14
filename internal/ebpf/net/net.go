package net

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

// Tracer attaches to network socket syscall tracepoints and emits events.
type Tracer struct {
	ringBufSize uint32 // 0 = use compiled default
	objs        netTraceObjects
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

// WithRingBufSize overrides the compiled-in ring buffer size (default 512KB).
// The value must be a power of 2 and at least 4096 (one page).
func WithRingBufSize(bytes uint32) Option {
	return func(t *Tracer) {
		t.ringBufSize = bytes
	}
}

// New creates a network socket tracer. Call Attach() to start.
func New(opts ...Option) *Tracer {
	t := &Tracer{
		eventCh: make(chan events.Event, 4096),
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// Attach loads the eBPF programs, attaches syscall tracepoints, and creates the
// ring buffer reader. bpf2go's LoadAndAssign loads programs into the kernel but
// does NOT attach them — explicit link.Tracepoint calls are required.
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

	spec, err := loadNetTrace()
	if err != nil {
		return fmt.Errorf("loading net trace spec: %w", err)
	}

	if t.ringBufSize > 0 {
		if eventsMap, ok := spec.Maps["net_events"]; ok {
			eventsMap.MaxEntries = t.ringBufSize
		}
	}

	if err := spec.LoadAndAssign(&t.objs, nil); err != nil {
		return fmt.Errorf("loading net trace objects: %w", err)
	}

	// Attach syscall tracepoints — programs are loaded but dormant until linked.
	type tpSpec struct {
		group string
		name  string
		prog  *ebpf.Program
	}
	specs := []tpSpec{
		{"syscalls", "sys_enter_sendto", t.objs.HandleSysEnterSendto},
		{"syscalls", "sys_exit_sendto", t.objs.HandleSysExitSendto},
		{"syscalls", "sys_enter_recvfrom", t.objs.HandleSysEnterRecvfrom},
		{"syscalls", "sys_exit_recvfrom", t.objs.HandleSysExitRecvfrom},
	}
	for _, s := range specs {
		tp, err := link.Tracepoint(s.group, s.name, s.prog, nil)
		if err != nil {
			return fmt.Errorf("attaching tracepoint %s/%s: %w", s.group, s.name, err)
		}
		t.links = append(t.links, tp)
	}

	t.reader, err = ringbuf.NewReader(t.objs.NetEvents)
	if err != nil {
		return fmt.Errorf("creating net ring buffer reader: %w", err)
	}

	closeFn = nil
	return nil
}

// SetTargetPID adds a PID to the filter map.
// Also inserts a sentinel entry at key=0 so the eBPF-side net_pid_map_empty()
// check (which probes key=0) detects a non-empty map.
func (t *Tracer) SetTargetPID(pid uint32) error {
	val := uint8(1)
	// Sentinel: key=0 signals "map is populated" to net_pid_map_empty().
	if err := t.objs.NetTargetPids.Put(uint32(0), val); err != nil {
		return err
	}
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
		Comm:      events.CommToString(e.Hdr.Comm),
		Source:    events.SourceNet,
		Op:        e.Hdr.Op,
		Duration:  time.Duration(e.DurationNs),
		Args:      [2]uint64{uint64(e.Fd), uint64(e.Bytes)},
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
