// Package ncclhang detects NCCL collective hangs by tracking
// per-PID inactivity on the existing libnccl uprobe event stream
// from internal/ebpf/ncclprobe.
//
// The ncclprobe tracer emits one Event per uretprobe return — meaning
// successful completion of a collective call (ncclAllReduce,
// ncclAllGather, etc.). A PID that has previously called NCCL but
// stops emitting return events for tens of seconds is almost certainly
// stuck inside a collective: all ranks waiting on each other, no
// progress.
//
//   - Observe(pid, commIDHash, ts) called per ncclprobe.Event
//   - Sweep(now) -> []Hang   drained on a fixed tick (default 1s)
//
// Sweep emits at most one Hang per PID per episode. An episode
// starts when the per-PID idle age first crosses the configured
// threshold AND the tracker has been observing the PID long enough
// to be sure the silence isn't just "we just started watching."
// Suppression-once-per-episode rearms when fresh NCCL activity
// resumes for the PID — same shape as the cuidle and tcpretransmit
// packages.
//
// Anchor: catalog row T1 (NCCL collective hang). The 60s default
// threshold is conservative: healthy training loops issue collectives
// many times per second; even a long-checkpoint pause rarely exceeds
// 30-45 seconds between collective calls.
package ncclhang

import (
	"sync"
	"syscall"
	"time"
)

// DefaultIdleThreshold is the per-PID idle age (time since the last
// observed ncclAllReduce/ncclAllGather return) above which the PID
// is considered hung. 60 seconds is well above typical training-loop
// inter-collective gaps and below the time at which an operator
// would manually investigate.
const DefaultIdleThreshold = 60 * time.Second

// MinObservationWindow is the minimum elapsed time since first
// observing a PID before that PID is eligible to emit a hang. Without
// this, a PID observed for the first time in a steady-state agent
// could emit immediately on the first Sweep if we just happened to
// catch it in a long inter-collective pause.
const MinObservationWindow = DefaultIdleThreshold

// DefaultSuppressionWindow is how long a PID must show fresh NCCL
// activity before a previously-emitted hang can re-fire.
const DefaultSuppressionWindow = 120 * time.Second

// Hang is the per-episode summary emitted by Sweep. Maps 1:1 onto
// the orchestrator's NcclHangState wire message.
type Hang struct {
	PID        uint32
	CommIDHash uint64
	IdleMs     uint64
}

// Tracker holds per-PID last-NCCL-event state.
type Tracker struct {
	mu                sync.Mutex
	pids              map[uint32]*pidState
	idleThreshold     time.Duration
	minObservation    time.Duration
	suppressionWindow time.Duration
	isPidAlive        func(uint32) bool
}

type pidState struct {
	firstSeen  time.Time
	lastEvent  time.Time
	commIDHash uint64
	emitted    bool
	emittedAt  time.Time
}

// New returns a Tracker with default thresholds.
func New() *Tracker {
	return NewWithThresholds(DefaultIdleThreshold, MinObservationWindow, DefaultSuppressionWindow)
}

// NewWithThresholds returns a Tracker with custom thresholds. Tests
// use short windows for deterministic assertions.
func NewWithThresholds(idle, minObservation, suppression time.Duration) *Tracker {
	return &Tracker{
		pids:              make(map[uint32]*pidState),
		idleThreshold:     idle,
		minObservation:    minObservation,
		suppressionWindow: suppression,
		isPidAlive:        defaultIsPidAlive,
	}
}

// Observe records a NCCL return event for pid on commIDHash at ts.
// pid==0 is dropped. A fresh event always resets any in-flight
// emitted flag — the workload is clearly making progress again.
func (t *Tracker) Observe(pid uint32, commIDHash uint64, ts time.Time) {
	if pid == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st, ok := t.pids[pid]
	if !ok {
		st = &pidState{firstSeen: ts}
		t.pids[pid] = st
	}
	st.lastEvent = ts
	if commIDHash != 0 {
		st.commIDHash = commIDHash
	}
	if st.emitted {
		st.emitted = false
		st.emittedAt = time.Time{}
	}
}

// Sweep emits one Hang per PID whose idle age crosses the threshold
// this tick, is past the minimum-observation window, is still alive,
// and has not yet emitted in the current episode. Returns at most
// one Hang per PID per episode.
//
// Behavior:
//
//   - Below idle threshold OR before min-observation -> no-op.
//   - Idle and alive and not-yet-emitted -> emit Hang.
//   - Idle and PID gone -> drop the tracker entry (workload exited
//     on its own; not a hang).
//   - Already emitted, suppression window elapsed -> clear emitted
//     flag so a re-detection on a still-stuck PID can re-fire.
func (t *Tracker) Sweep(now time.Time) []Hang {
	t.mu.Lock()
	defer t.mu.Unlock()
	var out []Hang
	for pid, st := range t.pids {
		idle := now.Sub(st.lastEvent)
		observed := now.Sub(st.firstSeen)
		if idle <= t.idleThreshold || observed < t.minObservation {
			continue
		}
		if !t.isPidAlive(pid) {
			delete(t.pids, pid)
			continue
		}
		if !st.emitted {
			out = append(out, Hang{
				PID:        pid,
				CommIDHash: st.commIDHash,
				IdleMs:     uint64(idle.Milliseconds()),
			})
			st.emitted = true
			st.emittedAt = now
			continue
		}
		if now.Sub(st.emittedAt) >= t.suppressionWindow {
			st.emitted = false
			st.emittedAt = time.Time{}
		}
	}
	return out
}

// Forget drops state for a PID. Call this on observed process exit.
func (t *Tracker) Forget(pid uint32) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.pids, pid)
}

// TrackedPIDs returns the number of PIDs currently tracked.
func (t *Tracker) TrackedPIDs() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.pids)
}

// SetPidLivenessProbe overrides the default kill(pid, 0) check.
// Test-only seam.
func (t *Tracker) SetPidLivenessProbe(probe func(uint32) bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.isPidAlive = probe
}

func defaultIsPidAlive(pid uint32) bool {
	if pid == 0 {
		return false
	}
	err := syscall.Kill(int(pid), 0)
	if err == nil {
		return true
	}
	if err == syscall.ESRCH {
		return false
	}
	return true
}
