package tcp

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"unsafe"

	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"

	"github.com/ingero-io/ingero/pkg/events"
)

// Tracer attaches to TCP tracepoints and emits retransmit events.
type Tracer struct {
	ringBufSize uint32 // 0 = use compiled default
	objs        tcpTraceObjects
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

// WithRingBufSize overrides the compiled-in ring buffer size (default 256KB).
// The value must be a power of 2 and at least 4096 (one page).
func WithRingBufSize(bytes uint32) Option {
	return func(t *Tracer) {
		t.ringBufSize = bytes
	}
}

// New creates a TCP tracer. Call Attach() to start.
func New(opts ...Option) *Tracer {
	t := &Tracer{
		eventCh: make(chan events.Event, 4096),
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// Attach loads the eBPF program, attaches the BTF tracepoint, and creates the
// ring buffer reader. The tcp_retransmit_skb program uses tp_btf (BTF-enabled
// tracing) which attaches via link.AttachTracing — the kernel resolves the
// tracepoint from BTF info embedded in the program.
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

	spec, err := loadTcpTrace()
	if err != nil {
		return fmt.Errorf("loading TCP trace spec: %w", err)
	}

	if t.ringBufSize > 0 {
		if eventsMap, ok := spec.Maps["tcp_events"]; ok {
			eventsMap.MaxEntries = t.ringBufSize
		}
	}

	if err := spec.LoadAndAssign(&t.objs, nil); err != nil {
		return fmt.Errorf("loading TCP trace objects: %w", err)
	}

	// tp_btf programs attach via AttachTracing (kernel resolves tracepoint from BTF).
	tp, err := link.AttachTracing(link.TracingOptions{
		Program: t.objs.HandleTcpRetransmit,
	})
	if err != nil {
		return fmt.Errorf("attaching tp_btf/tcp_retransmit_skb: %w", err)
	}
	t.links = append(t.links, tp)

	t.reader, err = ringbuf.NewReader(t.objs.TcpEvents)
	if err != nil {
		return fmt.Errorf("creating TCP ring buffer reader: %w", err)
	}

	closeFn = nil
	return nil
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

// parseEvent converts raw ring buffer bytes into an events.Event.
// TCP retransmit events store addresses in Args:
//   Args[0] = (saddr << 32) | daddr — IPv4 packed pair
//   Args[1] = (sport << 16) | dport — port pair
func (t *Tracer) parseEvent(raw []byte) (events.Event, bool) {
	expected := int(unsafe.Sizeof(tcpTraceIngeroTcpEvent{}))
	if len(raw) < expected {
		return events.Event{}, false
	}

	e := (*tcpTraceIngeroTcpEvent)(unsafe.Pointer(&raw[0]))

	return events.Event{
		Timestamp: events.KtimeToWallClock(e.Hdr.TimestampNs),
		PID:       e.Hdr.Pid,
		TID:       e.Hdr.Tid,
		Comm:      events.CommToString(e.Hdr.Comm),
		Source:    events.SourceTCP,
		Op:        e.Hdr.Op,
		Args: [2]uint64{
			uint64(e.Saddr)<<32 | uint64(e.Daddr),
			uint64(e.Sport)<<16 | uint64(e.Dport),
		},
		GPUID:    uint32(e.State), // reuse GPUID field for TCP state
		CGroupID: e.Hdr.CgroupId,
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
