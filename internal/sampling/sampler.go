// Package sampling provides a mode-controlled, edge-triggered event sampler.
//
// The sampler sits at the store-insert chokepoint in inference deployments:
// inference workloads emit kernel events at rates that flood the agent's
// SQLite event store under uniform admission. A small healthy-state admit
// fraction (e.g. 1%) keeps storage bounded; on degradation the rate jumps
// to 100% so the events-of-interest are captured at full fidelity, and a
// cooldown window after recovery preserves the trailing context.
//
// Training and unknown modes bypass the sampler entirely (always emit) so
// existing training-only deployments observe no behavior change.
package sampling

import (
	"log/slog"
	"math/rand/v2"
	"sync"
	"time"
)

// DefaultHealthyRate is the fraction of events admitted in the healthy
// inference state. 0.01 = 1%.
const DefaultHealthyRate = 0.01

// DefaultCooldownDuration is how long the sampler retains the 100% admit
// rate after a degradation falling edge.
const DefaultCooldownDuration = 30 * time.Second

type state int

const (
	stateHealthy state = iota
	stateDegraded
	stateCooldown
)

func (s state) String() string {
	switch s {
	case stateHealthy:
		return "healthy"
	case stateDegraded:
		return "degraded"
	case stateCooldown:
		return "cooldown"
	default:
		return "unknown"
	}
}

// Sampler admits a fraction of events to the store based on workload mode
// and a degradation-edge state machine. Safe for concurrent use.
type Sampler struct {
	bypass       bool
	healthyRate  float64
	cooldownDur  time.Duration
	nowFn        func() time.Time

	mu          sync.Mutex
	state       state
	cooldownEnd time.Time
}

// New creates a Sampler. mode is one of "training", "inference", "unknown",
// or "" — only "inference" engages the rate-limiting state machine; all
// other modes bypass and ShouldEmit always returns true.
//
// healthyRate is the admit fraction in the healthy state (0.0 = drop all,
// 1.0 = admit all). cooldownDur is how long to retain the 100% rate after
// a degradation falling edge.
func New(mode string, healthyRate float64, cooldownDur time.Duration) *Sampler {
	return newWithClock(mode, healthyRate, cooldownDur, time.Now)
}

// newWithClock is the test-only constructor that accepts an injected clock.
func newWithClock(mode string, healthyRate float64, cooldownDur time.Duration, nowFn func() time.Time) *Sampler {
	return &Sampler{
		bypass:      mode != "inference",
		healthyRate: healthyRate,
		cooldownDur: cooldownDur,
		nowFn:       nowFn,
		state:       stateHealthy,
	}
}

// SetDegraded informs the sampler of degradation-state edge transitions.
// Idempotent within a state - only the rising and falling edges matter,
// so callers can invoke it on every tick without filtering. No-op in
// non-inference modes.
func (s *Sampler) SetDegraded(degraded bool) {
	if s.bypass {
		return
	}
	// Capture the post-transition state under the lock and log AFTER unlock
	// using the captured value. Logging while still holding the lock would
	// keep the critical section longer than necessary; logging without
	// capturing would let a concurrent SetDegraded race in between unlock
	// and log, mis-attributing this transition's state name.
	s.mu.Lock()
	var transitioned bool
	var loggedState string
	if degraded {
		// Rising edge from healthy or cooldown. Already-degraded is a no-op
		// so we don't re-log a transition that hasn't actually happened.
		if s.state != stateDegraded {
			s.state = stateDegraded
			s.cooldownEnd = time.Time{}
			transitioned = true
			loggedState = s.state.String()
		}
	} else if s.state == stateDegraded {
		// Falling edge: only meaningful from degraded. Healthy->healthy and
		// cooldown->cooldown are no-ops (don't restart cooldown on every tick).
		s.state = stateCooldown
		s.cooldownEnd = s.nowFn().Add(s.cooldownDur)
		transitioned = true
		loggedState = s.state.String()
	}
	s.mu.Unlock()

	if transitioned {
		slog.Info("sampling: state transition", "state", loggedState)
	}
}

// ShouldEmit returns true when an event should be admitted to the store.
// Always true in non-inference modes. In inference mode the result depends
// on the current state: healthy uses the configured rate, degraded and
// cooldown are 100%. Cooldown expiry transitions back to healthy on the
// next call.
func (s *Sampler) ShouldEmit() bool {
	// Bypass check is the very first thing — non-inference deployments
	// pay only one branch on the hot path.
	if s.bypass {
		return true
	}

	s.mu.Lock()
	var transitioned bool
	var loggedState string
	if s.state == stateCooldown && !s.nowFn().Before(s.cooldownEnd) {
		s.state = stateHealthy
		s.cooldownEnd = time.Time{}
		transitioned = true
		loggedState = s.state.String()
	}
	rate := s.healthyRate
	if s.state != stateHealthy {
		rate = 1.0
	}
	s.mu.Unlock()
	// Log the cooldown->healthy transition AFTER releasing the lock with the
	// captured state name. Otherwise a concurrent SetDegraded(true) racing
	// between unlock and log would observe state==degraded and emit a log
	// line that misreports this transition.
	if transitioned {
		slog.Info("sampling: state transition", "state", loggedState)
	}

	if rate >= 1.0 {
		return true
	}
	if rate <= 0.0 {
		return false
	}
	return rand.Float64() < rate
}
