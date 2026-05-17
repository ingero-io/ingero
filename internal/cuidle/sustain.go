// Package cuidle aggregates per-PID `cuLaunchKernel` events from the
// existing internal/ebpf/driver tracer into an idle-timer signal
// suitable for orchestrator dispatch.
//
// A workload that has GPU memory but has not issued a kernel launch
// for tens of seconds is almost certainly stuck inside a kernel call
// (cuLaunchKernel issued, the matching stream-sync never returns
// because the kernel itself hung). The orchestrator's
// InferenceProcessHang chain (gpu_context_reset -> process_recycle)
// dispatches against this signal.
//
//   - Observe(pid, gpuID, ts) called per cuLaunchKernel event
//   - Sweep(now) -> []Hang   drained on a fixed tick (default 1s)
//
// Sweep emits at most one Hang per PID per episode. An episode
// starts when the per-PID idle age first crosses the configured
// threshold; ends when a new launch resets the timer; rearms after
// the same suppression-by-quiet pattern the tcpretransmit package
// uses so a flapping workload does not flood the orchestrator.
//
// Anchor: catalog row I5 (inference process hang). The 30s default
// threshold is conservative: a healthy serving workload launches
// kernels many times per second; even a long-running training step
// rarely exceeds 10s between launches. Thirty seconds catches the
// hang shape without firing on legitimate idle gaps.
package cuidle

import (
	"sync"
	"syscall"
	"time"
)

// DefaultIdleThreshold is the per-PID idle age (time since the last
// observed cuLaunchKernel) above which the PID is considered hung.
// Thirty seconds is well above the upper end of legitimate idle
// gaps in production workloads and well below the time at which an
// operator would manually investigate.
const DefaultIdleThreshold = 30 * time.Second

// DefaultSuppressionWindow is how long a PID must show fresh kernel
// activity (or be gone from the tracker entirely) before a fresh
// Hang can be emitted. Mirrors tcpretransmit's rearm design.
const DefaultSuppressionWindow = 60 * time.Second

// Hang is the per-episode summary emitted by Sweep. Maps 1:1 onto
// the orchestrator's InferenceProcessHangState wire message.
type Hang struct {
	PID    uint32
	GPUID  uint32
	IdleMs uint64
}

// Tracker holds per-PID last-launch state and emits one Hang per
// sustained episode. Safe for concurrent Observe + Sweep callers.
type Tracker struct {
	mu                sync.Mutex
	pids              map[uint32]*pidState
	idleThreshold     time.Duration
	suppressionWindow time.Duration
	// isPidAlive is overridable for tests; defaults to a real
	// kill(pid, 0) check so a process that exited between the last
	// observed launch and a Sweep tick is not falsely reported as
	// hung (it's not hung, it's gone).
	isPidAlive func(uint32) bool
}

type pidState struct {
	lastLaunch time.Time
	gpuID      uint32
	// emitted is set when a Hang has already been emitted for the
	// current episode; cleared when either a new launch lands or
	// the suppression window elapses with the PID gone.
	emitted bool
	// emittedAt records when the episode signal landed so the
	// suppression rearm can measure elapsed time.
	emittedAt time.Time
}

// New returns a Tracker with default thresholds.
func New() *Tracker {
	return NewWithThresholds(DefaultIdleThreshold, DefaultSuppressionWindow)
}

// NewWithThresholds returns a Tracker with custom thresholds. Useful
// for tests that want short idle / suppression windows.
func NewWithThresholds(idle time.Duration, suppression time.Duration) *Tracker {
	return &Tracker{
		pids:              make(map[uint32]*pidState),
		idleThreshold:     idle,
		suppressionWindow: suppression,
		isPidAlive:        defaultIsPidAlive,
	}
}

// Observe records a kernel launch for pid on gpuID at wall-clock ts.
// pid==0 is dropped; gpuID is stored verbatim and reported in the
// emitted Hang for operator audit. A new observation always resets
// any in-flight episode by updating lastLaunch — the workload is
// clearly still issuing kernels.
func (t *Tracker) Observe(pid uint32, gpuID uint32, ts time.Time) {
	if pid == 0 {
		return
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	st, ok := t.pids[pid]
	if !ok {
		st = &pidState{}
		t.pids[pid] = st
	}
	st.lastLaunch = ts
	st.gpuID = gpuID
	// A fresh launch closes the current episode. Without this, a
	// transient hang that recovered would never let a NEW hang on
	// the same PID re-emit (the suppression window would still be
	// counting from the first emission).
	if st.emitted {
		st.emitted = false
		st.emittedAt = time.Time{}
	}
}

// Sweep returns one Hang per PID whose idle age crosses the
// threshold this tick and is still alive on the host. Caller is
// expected to invoke this on a fixed tick (the watcher uses 1s).
//
// Behavior:
//
//   - Idle > threshold AND PID alive AND not-yet-emitted -> emit Hang.
//   - Idle > threshold AND PID gone -> drop the tracker entry (the
//     process exited; the hang is not actionable, and the
//     ZombieGpuAllocation reconciler will pick up any orphan VRAM
//     separately if the driver retained the allocation).
//   - Idle <= threshold -> no-op (PID still issuing launches OR
//     fresh enough that the watcher should give it more time).
//   - Already-emitted PID stays alive past the suppression window
//     -> drop the emitted flag so a subsequent fresh hang can
//     re-emit (mirrors tcpretransmit's rearm pattern).
func (t *Tracker) Sweep(now time.Time) []Hang {
	t.mu.Lock()
	defer t.mu.Unlock()

	var out []Hang
	for pid, st := range t.pids {
		idle := now.Sub(st.lastLaunch)

		if idle <= t.idleThreshold {
			continue
		}

		if !t.isPidAlive(pid) {
			delete(t.pids, pid)
			continue
		}

		if !st.emitted {
			out = append(out, Hang{
				PID:    pid,
				GPUID:  st.gpuID,
				IdleMs: uint64(idle.Milliseconds()),
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

// Forget drops state for a PID. Call this on observed process exit
// so the per-PID map doesn't grow unboundedly across short-lived
// workloads.
func (t *Tracker) Forget(pid uint32) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.pids, pid)
}

// TrackedPIDs returns the number of PIDs currently tracked. For
// metrics / debug output; not used in the dispatch path.
func (t *Tracker) TrackedPIDs() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.pids)
}

// SetPidLivenessProbe overrides the default kill(pid, 0) check.
// Test-only seam; the watcher relies on the default. Exposed
// so callers can plug in a mock for deterministic Sweep tests
// without forking real processes.
func (t *Tracker) SetPidLivenessProbe(probe func(uint32) bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.isPidAlive = probe
}

// defaultIsPidAlive does `kill(pid, 0)`. ESRCH means the kernel has
// no such PID; EPERM means the PID exists but we lack permission to
// signal it (still alive); any other errno is conservatively treated
// as alive so a transient kernel error never wipes a legitimate
// in-flight detection.
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
