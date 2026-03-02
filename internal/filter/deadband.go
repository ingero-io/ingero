// Package filter provides suppression filters for system metric snapshots.
//
// Deadband filtering suppresses writes when metrics change by less than a
// configurable percentage threshold. Heartbeat ensures at least one report
// every N seconds even when values are stable. This reduces SQLite bloat
// and OTLP network traffic on idle systems without losing data fidelity.
package filter

import (
	"math"
	"sync"
	"time"
)

// Config holds deadband and heartbeat parameters.
type Config struct {
	DeadbandPct       float64       // suppress if change < this % (0 = disabled)
	HeartbeatInterval time.Duration // force emit at least this often (0 = no heartbeat)
}

// Disabled returns true when the deadband filter would be a no-op.
func (c Config) Disabled() bool {
	return c.DeadbandPct <= 0
}

// NewSnapshotFilter creates a SnapshotFilter from a Config.
// Returns nil if the config is disabled, making the filter a no-op.
func (c Config) NewSnapshotFilter() *SnapshotFilter {
	if c.Disabled() {
		return nil
	}
	return &SnapshotFilter{
		deadbandPct: c.DeadbandPct,
		heartbeat:   c.HeartbeatInterval,
		nowFn:       time.Now,
	}
}

// SnapshotFilter tracks the last-emitted system metric values and decides
// whether a new snapshot should be emitted based on deadband and heartbeat.
//
// Thread-safe: all methods are guarded by a mutex.
// Nil-safe: calling ShouldEmit on a nil *SnapshotFilter always returns true.
type SnapshotFilter struct {
	mu          sync.Mutex
	deadbandPct float64
	heartbeat   time.Duration

	// Last emitted values.
	lastCPU     float64
	lastMem     float64
	lastMemMB   int64
	lastSwapMB  int64
	lastLoad    float64
	lastEmit    time.Time
	initialized bool

	// For testing: injectable clock.
	nowFn func() time.Time
}

// ShouldEmit returns true if the snapshot should be written/pushed.
//
// Rules (evaluated in order):
//  1. First call: always emit (establish baseline).
//  2. Heartbeat: if time since last emit >= heartbeat interval, emit unconditionally.
//  3. Deadband: if ANY metric changed by more than deadbandPct% of max(|prev|, 1.0), emit.
//
// A nil *SnapshotFilter always returns true (disabled / no-op).
func (f *SnapshotFilter) ShouldEmit(cpuPct, memPct float64, memAvailMB, swapMB int64, loadAvg1 float64) bool {
	if f == nil {
		return true
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	now := f.nowFn()

	// First call: always emit to establish baseline.
	if !f.initialized {
		f.update(cpuPct, memPct, memAvailMB, swapMB, loadAvg1, now)
		return true
	}

	// Heartbeat: force emit after interval elapses.
	if f.heartbeat > 0 && now.Sub(f.lastEmit) >= f.heartbeat {
		f.update(cpuPct, memPct, memAvailMB, swapMB, loadAvg1, now)
		return true
	}

	// Deadband: emit if ANY metric exceeds the threshold.
	if exceedsDeadband(f.lastCPU, cpuPct, f.deadbandPct) ||
		exceedsDeadband(f.lastMem, memPct, f.deadbandPct) ||
		exceedsDeadband(float64(f.lastMemMB), float64(memAvailMB), f.deadbandPct) ||
		exceedsDeadband(float64(f.lastSwapMB), float64(swapMB), f.deadbandPct) ||
		exceedsDeadband(f.lastLoad, loadAvg1, f.deadbandPct) {
		f.update(cpuPct, memPct, memAvailMB, swapMB, loadAvg1, now)
		return true
	}

	return false
}

// update stores the latest values and timestamp.
// Caller must hold f.mu.
func (f *SnapshotFilter) update(cpuPct, memPct float64, memAvailMB, swapMB int64, loadAvg1 float64, now time.Time) {
	f.lastCPU = cpuPct
	f.lastMem = memPct
	f.lastMemMB = memAvailMB
	f.lastSwapMB = swapMB
	f.lastLoad = loadAvg1
	f.lastEmit = now
	f.initialized = true
}

// exceedsDeadband checks if |new - old| > pct% of max(|old|, 1.0).
// The max(|old|, 1.0) base prevents division-by-zero and ensures
// zero-to-nonzero transitions (e.g., swap 0 → 1 MB) are detected.
func exceedsDeadband(old, new, pct float64) bool {
	base := math.Max(math.Abs(old), 1.0)
	threshold := base * pct / 100.0
	return math.Abs(new-old) > threshold
}
