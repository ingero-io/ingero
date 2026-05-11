package kernellaunch

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

// EventSize is the on-the-wire size of one ringbuf record. Must
// match struct kernel_launch_event in bpf/kernel_launch.bpf.c.
const EventSize = 56

// Event is the userspace mirror of struct kernel_launch_event.
//
// PID is the userspace process ID (kernel tgid); TID is the kernel
// thread ID. Field order matches the BPF struct after the v0.15
// label swap; the wire bytes did not change.
type Event struct {
	TimestampNs uint64
	CgroupID    uint64
	FuncHandle  uint64
	TID         uint32 // kernel TID
	PID         uint32 // userspace PID
	GridX       uint32
	GridY       uint32
	GridZ       uint32
	BlockX      uint32
	BlockY      uint32 // BlockZ requires arch-specific stack arg read; deferred to v0.15.x
	Pad0        uint32
}

// ThreadsPerBlock returns BlockX * BlockY (BlockZ assumed 1 in
// v0.15; the bulk of CUDA workloads use 1D / 2D blocks). Unknown
// (BlockX==0) returns 0.
func (e Event) ThreadsPerBlock() uint64 {
	if e.BlockX == 0 {
		return 0
	}
	by := uint64(e.BlockY)
	if by == 0 {
		by = 1
	}
	return uint64(e.BlockX) * by
}

// TotalGridBlocks returns GridX * GridY * GridZ.
func (e Event) TotalGridBlocks() uint64 {
	gx, gy, gz := uint64(e.GridX), uint64(e.GridY), uint64(e.GridZ)
	if gy == 0 {
		gy = 1
	}
	if gz == 0 {
		gz = 1
	}
	return gx * gy * gz
}

// Tracer holds the loaded BPF program + the ringbuf reader and
// the uprobe link.
type Tracer struct {
	libcudaPath string

	objs   kernelLaunchObjects
	link   link.Link
	reader *ringbuf.Reader

	eventCh  chan Event
	dropped  atomic.Uint64
	parseErr atomic.Uint64
	closed   atomic.Bool

	mu sync.Mutex
}

// New constructs a not-yet-attached tracer. libcudaPath should be
// the absolute path to libcuda.so.1; pass "" to fall back to the
// loader's default search.
func New(libcudaPath string) *Tracer {
	return &Tracer{
		libcudaPath: libcudaPath,
		eventCh:     make(chan Event, 4096),
	}
}

// Events returns the read-side channel.
func (t *Tracer) Events() <-chan Event { return t.eventCh }

// Dropped returns the count of events dropped due to a full
// userspace channel.
func (t *Tracer) Dropped() uint64 { return t.dropped.Load() }

// ParseErrors returns the count of malformed ringbuf records.
func (t *Tracer) ParseErrors() uint64 { return t.parseErr.Load() }

// Attach loads the BPF program and wires the cuLaunchKernel uprobe.
func (t *Tracer) Attach() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.libcudaPath == "" {
		return fmt.Errorf("kernellaunch: libcuda path is empty")
	}
	if err := loadKernelLaunchObjects(&t.objs, nil); err != nil {
		return fmt.Errorf("kernellaunch: load BPF: %w", err)
	}
	ex, err := link.OpenExecutable(t.libcudaPath)
	if err != nil {
		t.objs.Close()
		return fmt.Errorf("kernellaunch: open libcuda %s: %w", t.libcudaPath, err)
	}
	up, err := ex.Uprobe("cuLaunchKernel", t.objs.UprobeCuLaunchKernel, nil)
	if err != nil {
		t.objs.Close()
		return fmt.Errorf("kernellaunch: attach uprobe cuLaunchKernel: %w", err)
	}
	rd, err := ringbuf.NewReader(t.objs.KernelLaunchEvents)
	if err != nil {
		_ = up.Close()
		t.objs.Close()
		return fmt.Errorf("kernellaunch: open ringbuf reader: %w", err)
	}
	t.link = up
	t.reader = rd
	return nil
}

// Run drains the ringbuf into the eventCh until ctx is cancelled
// or the reader is closed.
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
			return fmt.Errorf("kernellaunch: ringbuf read: %w", err)
		}
		t.handleRawSample(rec.RawSample)
	}
}

// handleRawSample parses one ringbuf record. Extracted from Run so
// unit tests can exercise the deserialization path without a real
// BPF program.
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

// Close detaches the uprobe and releases all BPF resources.
// Idempotent.
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

