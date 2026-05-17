// Package zombiegpu detects orphan GPU VRAM allocations: PIDs that
// the NVIDIA driver still reports as owning compute memory but that
// have already exited from the kernel's PID table.
//
// The mechanism: `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory`
// returns rows from the driver's per-context tracking. On some
// driver versions a PID's allocation persists past process exit
// (cgroup teardown or full GPU reset releases it). The reconciler
// walks each row and cross-references with `kill(pid, 0)` — when the
// driver row exists but the kernel reports ESRCH for the PID, that's
// a zombie allocation worth reclaiming.
//
// The orchestrator's ZombieGpuAllocation chain
// (gpu_context_reset -> pod_drain) dispatches against this signal.
// gpu_context_reset is a no-op step (the PID is gone, nothing to
// SIGKILL) that records the audit event; pod_drain then evicts the
// parent pod so cgroup teardown reclaims the VRAM.
//
// Anchor: NEW catalog row I21 (zombie GPU resource leak). Cycle
// time matches the memfrag poller (default 5s) — the leak is not
// time-sensitive (it persists until reclaimed) but reporting it
// within a few seconds keeps the operator dashboard responsive.
package zombiegpu

import (
	"context"
	"sync"
	"syscall"

	"github.com/ingero-io/ingero/internal/nvml"
)

// Allocation is the per-zombie record the reconciler emits. UUIDs
// are GPU identifiers from nvidia-smi; the orchestrator maps them
// to gpu_id via positional enumeration (same convention the
// throttle poller uses).
type Allocation struct {
	PID            uint32
	GPUUUID        string
	AllocatedBytes uint64
}

// Reconciler walks one nvidia-smi --query-compute-apps snapshot per
// Tick call and returns the rows where the driver-reported PID is
// gone. Suppression-once-per-PID-per-episode mirrors the
// tcpretransmit and cuidle patterns: once emitted, a PID won't
// re-emit until it disappears from the driver readings entirely.
// That cycle happens when the operator action (pod_drain ->
// cgroup teardown) reclaims the allocation, the driver stops
// reporting it, and a fresh zombie on a future PID rearms cleanly.
type Reconciler struct {
	mu       sync.Mutex
	emitted  map[uint32]struct{}
	isAlive  func(uint32) bool
	getApps  func(context.Context, nvml.Runner) ([]nvml.ComputeAppReading, error)
}

// New returns a Reconciler that uses kill(pid, 0) for liveness and
// nvml.GetComputeApps for the driver enumeration. Caller-supplied
// Runner controls the actual nvidia-smi invocation.
func New() *Reconciler {
	return &Reconciler{
		emitted: make(map[uint32]struct{}),
		isAlive: defaultIsPidAlive,
		getApps: nvml.GetComputeApps,
	}
}

// Tick runs one reconcile pass. Returns zero or more Allocations
// for orphan PIDs that have not yet emitted in the current episode.
// A new Allocation is emitted only once per PID until that PID
// disappears entirely from the driver enumeration; after disappearance
// any future re-appearance is treated as a fresh zombie episode.
//
// Returns an error only on Runner failure (e.g. nvidia-smi exec
// failed). An empty Allocation slice with nil error means "no
// zombies this tick" — the expected steady state on a healthy host.
func (r *Reconciler) Tick(ctx context.Context, run nvml.Runner) ([]Allocation, error) {
	if run == nil {
		// No runner configured — degrade silently. Same shape as
		// the memfrag poller's nil-runner path.
		return nil, nil
	}
	readings, err := r.getApps(ctx, run)
	if err != nil {
		return nil, err
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	// Build the set of PIDs the driver currently reports so we can
	// drop emitted-flag entries for PIDs that have fully cleared.
	currentlyReported := make(map[uint32]struct{}, len(readings))
	var out []Allocation
	for _, rd := range readings {
		if rd.PID == 0 {
			continue
		}
		currentlyReported[rd.PID] = struct{}{}
		if r.isAlive(rd.PID) {
			// Live PID with a driver allocation is normal — not
			// our concern.
			continue
		}
		// Zombie: driver still reports the row, PID is gone.
		if _, alreadyEmitted := r.emitted[rd.PID]; alreadyEmitted {
			continue
		}
		bytes := uint64(0)
		if rd.UsedBytes > 0 {
			bytes = uint64(rd.UsedBytes)
		}
		out = append(out, Allocation{
			PID:            rd.PID,
			GPUUUID:        rd.UUID,
			AllocatedBytes: bytes,
		})
		r.emitted[rd.PID] = struct{}{}
	}

	// Rearm: any PID we previously emitted that is no longer in the
	// driver reading at all has been reclaimed; drop the emitted
	// flag so a future zombie on the same PID re-emits.
	for pid := range r.emitted {
		if _, stillThere := currentlyReported[pid]; !stillThere {
			delete(r.emitted, pid)
		}
	}

	return out, nil
}

// EmittedCount returns the current number of PIDs that have been
// emitted-not-yet-cleared. For metrics / debug only.
func (r *Reconciler) EmittedCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.emitted)
}

// SetPidLivenessProbe overrides the default kill(pid, 0) check.
// Test-only seam; production code relies on the default.
func (r *Reconciler) SetPidLivenessProbe(probe func(uint32) bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.isAlive = probe
}

// SetGetAppsForTest overrides the nvml.GetComputeApps wrapper so
// tests can feed canned readings without a runnable nvidia-smi.
func (r *Reconciler) SetGetAppsForTest(fn func(context.Context, nvml.Runner) ([]nvml.ComputeAppReading, error)) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.getApps = fn
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
	// EPERM (no permission to signal): PID exists. Other errors:
	// conservatively treat as alive so a transient kernel error
	// doesn't yield a false zombie emission.
	return true
}
