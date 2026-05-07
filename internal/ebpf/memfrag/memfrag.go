package memfrag

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
)

// kprobeTarget is the closed-driver symbol the BPF program attaches
// to. Stable across NVIDIA driver releases 470 series and later in
// our testing; if a future driver renames it, the kprobe.Open call
// will fail with ENOENT, which is logged at startup and the tracer
// is disabled for that run (the agent does not crash).
const kprobeTarget = "nvidia_unlocked_ioctl"

// EventSize is the on-the-wire size of one ringbuf record. Must
// match `struct memfrag_ioctl_event` in bpf/memfrag_kprobe.bpf.c.
const EventSize = 32

// Event is the userspace mirror of struct memfrag_ioctl_event.
type Event struct {
	TimestampNs uint64
	CgroupID    uint64
	PID         uint32
	TGID        uint32
	Cmd         uint32 // raw IOCTL cmd from nvidia_unlocked_ioctl
	Pad0        uint32
}

// Tracer holds the loaded BPF program + the ringbuf reader.
type Tracer struct {
	objs   memfragKprobeObjects
	link   link.Link
	reader *ringbuf.Reader

	eventCh  chan Event
	dropped  atomic.Uint64
	parseErr atomic.Uint64
	closed   atomic.Bool

	mu sync.Mutex
}

// New constructs a not-yet-attached tracer. Call Attach to load the
// BPF program and wire the kprobe.
func New() *Tracer {
	return &Tracer{
		eventCh: make(chan Event, 4096),
	}
}

// Events returns the read-side channel.
func (t *Tracer) Events() <-chan Event { return t.eventCh }

// Dropped returns the count of events dropped due to a full
// userspace channel.
func (t *Tracer) Dropped() uint64 { return t.dropped.Load() }

// ParseErrors returns the count of malformed ringbuf records.
func (t *Tracer) ParseErrors() uint64 { return t.parseErr.Load() }

// Attach loads the BPF program and wires the kprobe on
// nvidia_unlocked_ioctl. Returns an error if the kernel doesn't
// expose the symbol (e.g., NVIDIA driver not loaded).
func (t *Tracer) Attach() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if err := loadMemfragKprobeObjects(&t.objs, nil); err != nil {
		return fmt.Errorf("memfrag: load BPF: %w", err)
	}
	kp, err := link.Kprobe(kprobeTarget, t.objs.NvidiaUnlockedIoctlEnter, nil)
	if err != nil {
		t.objs.Close()
		return fmt.Errorf("memfrag: attach kprobe %s: %w", kprobeTarget, err)
	}
	rd, err := ringbuf.NewReader(t.objs.MemfragEvents)
	if err != nil {
		_ = kp.Close()
		t.objs.Close()
		return fmt.Errorf("memfrag: open ringbuf reader: %w", err)
	}
	t.link = kp
	t.reader = rd
	return nil
}

// Run drains the ringbuf into the eventCh until ctx is cancelled
// or the reader is closed. Mirrors ncclprobe.Tracer.Run.
func (t *Tracer) Run(ctx context.Context) error {
	defer close(t.eventCh)
	for {
		if err := ctx.Err(); err != nil {
			return nil
		}
		rec, err := t.reader.Read()
		if err != nil {
			if errors.Is(err, ringbuf.ErrClosed) {
				return nil
			}
			return fmt.Errorf("memfrag: ringbuf read: %w", err)
		}
		t.handleRawSample(rec.RawSample)
	}
}

// handleRawSample parses one ringbuf record and forwards a
// well-formed Event onto the channel. Extracted from Run so unit
// tests can exercise the deserialization path without a real BPF
// program.
func (t *Tracer) handleRawSample(raw []byte) {
	if len(raw) < EventSize {
		t.parseErr.Add(1)
		return
	}
	var e Event
	if err := binary.Read(bytes.NewReader(raw[:EventSize]), binary.LittleEndian, &e); err != nil {
		t.parseErr.Add(1)
		return
	}
	select {
	case t.eventCh <- e:
	default:
		t.dropped.Add(1)
	}
}

// Close detaches the kprobe and releases all BPF resources.
// Idempotent: a second Close after the first is a no-op.
func (t *Tracer) Close() error {
	if !t.closed.CompareAndSwap(false, true) {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.reader != nil {
		_ = t.reader.Close()
		t.reader = nil
	}
	if t.link != nil {
		_ = t.link.Close()
		t.link = nil
	}
	t.objs.Close()
	return nil
}
