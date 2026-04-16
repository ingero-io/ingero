package health

import (
	"fmt"
	"log/slog"
	"math"
	"sync"
	"time"

	"github.com/ingero-io/ingero/pkg/contract"
)

// State is the agent's current health lifecycle phase. Three emit-states —
// ACTIVE, CALIBRATING, IDLE — round-trip through the OTLP
// ingero.node.state attribute (see pkg/contract). STALE is an internal-only
// state: the agent cannot read CUDA events so it does not emit at all.
type State string

const (
	StateCalibrating State = contract.StateCalibrating
	StateActive      State = contract.StateActive
	StateIdle        State = contract.StateIdle
	StateStale       State = "stale"
)

// Observation is one tick of inputs for the state machine.
type Observation struct {
	// KernelLaunchCount is the number of CUDA kernel launches seen in the
	// push interval ending at Timestamp. Negative values are coerced to 0
	// at the boundary.
	KernelLaunchCount int
	// EventReadOK is false when the agent could not read CUDA trace events
	// for this interval (uprobe unattached, ringbuf closed, etc.).
	EventReadOK bool
	// Timestamp is when the interval ended. A timestamp more than
	// FutureTolerance ahead of the state machine's clock is treated as
	// EventReadOK=false — malicious or skewed future timestamps should
	// not influence idle detection.
	Timestamp time.Time
}

// FutureTolerance bounds how far ahead of "now" an observation's timestamp
// may be before the state machine treats it as a read failure.
const FutureTolerance = time.Minute

// StateConfig tunes the state machine thresholds.
type StateConfig struct {
	// IdleIntervals: consecutive zero-launch observations before transitioning
	// ACTIVE -> IDLE. Default 3.
	IdleIntervals int `yaml:"idle_intervals"`
	// WarmupSamples: observations required in CALIBRATING before moving to
	// ACTIVE. Should match BaselineConfig.WarmupSamples so the baseline is
	// ready when classification starts. Default 30.
	WarmupSamples int `yaml:"warmup_samples"`
	// StaleReadFailures: observations with EventReadOK=false required to
	// transition to STALE (from any state). STALE triggers when the
	// counter >= this value. Default 3.
	StaleReadFailures int `yaml:"stale_read_failures"`
	// RecentWindow: how many past observations to retain for
	// KernelLaunchesSince. Capped to IdleIntervals * 4 if zero. Default 0
	// (meaning derive). Clamped to a safe maximum to avoid pathological
	// configurations.
	RecentWindow int `yaml:"recent_window"`
}

// MaxIdleIntervals is the largest IdleIntervals value accepted by
// Validate. Bounded so that derived RecentWindow cannot overflow.
const MaxIdleIntervals = 1 << 20

// DefaultStateConfig returns the canonical values.
func DefaultStateConfig() StateConfig {
	return StateConfig{
		IdleIntervals:     3,
		WarmupSamples:     30,
		StaleReadFailures: 3,
		RecentWindow:      0, // derived
	}
}

// Validate rejects nonsensical configs.
func (c StateConfig) Validate() error {
	if c.IdleIntervals <= 0 {
		return fmt.Errorf("state.idle_intervals must be > 0: got %d", c.IdleIntervals)
	}
	if c.IdleIntervals > MaxIdleIntervals {
		return fmt.Errorf("state.idle_intervals must be <= %d: got %d", MaxIdleIntervals, c.IdleIntervals)
	}
	if c.WarmupSamples < 0 {
		return fmt.Errorf("state.warmup_samples must be >= 0: got %d", c.WarmupSamples)
	}
	if c.StaleReadFailures <= 0 {
		return fmt.Errorf("state.stale_read_failures must be > 0: got %d", c.StaleReadFailures)
	}
	if c.RecentWindow < 0 {
		return fmt.Errorf("state.recent_window must be >= 0: got %d", c.RecentWindow)
	}
	return nil
}

// StateMachine drives the CALIBRATING/ACTIVE/IDLE/STALE lifecycle.
//
// All methods are safe for concurrent use.
type StateMachine interface {
	// Current returns the current state.
	Current() State
	// TransitionIfNeeded folds one observation into the state machine. It
	// may or may not change state; the returned (prev, next, reason,
	// changed) quadruple describes the outcome. The reason is empty when
	// no change occurred.
	TransitionIfNeeded(obs Observation) (prev, next State, reason string, changed bool)
	// KernelLaunchesSince sums kernel launches across retained observations
	// whose Timestamp >= t. Retention is bounded by StateConfig.RecentWindow.
	KernelLaunchesSince(t time.Time) int
}

type stateMachine struct {
	cfg StateConfig
	log *slog.Logger

	mu               sync.RWMutex
	current          State
	consecutiveZeros int // ACTIVE -> IDLE counter
	consecutiveFails int // any -> STALE counter (saturates at cfg.StaleReadFailures * 2)
	calibratingSeen  int // observations seen while in CALIBRATING

	recent []Observation // ring-esque slice, bounded by cfg.RecentWindow
}

// NewStateMachine returns a StateMachine starting in CALIBRATING. Pass a
// nil logger to use slog.Default().
func NewStateMachine(cfg StateConfig, log *slog.Logger) (StateMachine, error) {
	return newStateMachineAt(cfg, log, StateCalibrating)
}

// NewStateMachineFromRestore returns a StateMachine starting in ACTIVE so
// that a freshly-restored baseline (Story 2.4) does not have to re-warm
// through CALIBRATING. Callers should only use this when the baseline was
// successfully restored from a recent persistence file.
func NewStateMachineFromRestore(cfg StateConfig, log *slog.Logger) (StateMachine, error) {
	return newStateMachineAt(cfg, log, StateActive)
}

func newStateMachineAt(cfg StateConfig, log *slog.Logger, initial State) (StateMachine, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.RecentWindow == 0 {
		cfg.RecentWindow = cfg.IdleIntervals * 4
	}
	if log == nil {
		log = slog.Default()
	}
	return &stateMachine{
		cfg:     cfg,
		log:     log,
		current: initial,
		recent:  make([]Observation, 0, cfg.RecentWindow),
	}, nil
}

func (s *stateMachine) Current() State {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.current
}

func (s *stateMachine) TransitionIfNeeded(obs Observation) (prev, next State, reason string, changed bool) {
	// Normalize at the boundary so downstream logic only sees well-formed
	// observations. KernelLaunchCount is coerced non-negative; a
	// far-future Timestamp is treated as a read failure.
	obs = normalizeObservation(obs)

	s.mu.Lock()
	defer s.mu.Unlock()

	prev = s.current
	s.rememberLocked(obs)

	// STALE is the highest-priority transition.
	if !obs.EventReadOK {
		if s.consecutiveFails < math.MaxInt/2 {
			s.consecutiveFails++
		}
		if s.consecutiveFails >= s.cfg.StaleReadFailures && s.current != StateStale {
			return s.transitionLocked(StateStale, fmt.Sprintf("event read failed %d consecutive times", s.consecutiveFails))
		}
	} else {
		s.consecutiveFails = 0
	}

	switch s.current {
	case StateStale:
		// Recover via CALIBRATING on first good observation. The
		// triggering sample counts toward warmup so operators get N total
		// samples including this one.
		if obs.EventReadOK {
			p, n, r, c := s.transitionLocked(StateCalibrating, "event reads recovered")
			s.calibratingSeen = 1
			return p, n, r, c
		}
		return prev, prev, "", false

	case StateCalibrating:
		s.calibratingSeen++
		s.updateIdleCounterLocked(obs)
		if s.calibratingSeen >= s.cfg.WarmupSamples {
			return s.transitionLocked(StateActive, fmt.Sprintf("warmup complete after %d samples", s.calibratingSeen))
		}
		return prev, prev, "", false

	case StateActive:
		s.updateIdleCounterLocked(obs)
		if s.consecutiveZeros >= s.cfg.IdleIntervals {
			return s.transitionLocked(StateIdle, fmt.Sprintf("no kernel launches for %d intervals", s.consecutiveZeros))
		}
		return prev, prev, "", false

	case StateIdle:
		// Any kernel activity wakes the node but sends it through
		// CALIBRATING, not ACTIVE — the baseline is stale.
		if obs.KernelLaunchCount > 0 {
			p, n, r, c := s.transitionLocked(StateCalibrating, "kernel launches resumed from idle")
			s.calibratingSeen = 1
			return p, n, r, c
		}
		return prev, prev, "", false
	}

	// Unreachable — defensive no-op.
	return prev, prev, "", false
}

func (s *stateMachine) KernelLaunchesSince(t time.Time) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	total := 0
	for _, o := range s.recent {
		if !o.Timestamp.Before(t) {
			total += o.KernelLaunchCount
		}
	}
	return total
}

// transitionLocked switches to next, logs, resets per-state counters, and
// returns the tuple expected by TransitionIfNeeded. Caller must hold s.mu.
func (s *stateMachine) transitionLocked(next State, reason string) (State, State, string, bool) {
	prev := s.current
	s.current = next

	switch next {
	case StateCalibrating:
		s.calibratingSeen = 0
		s.consecutiveZeros = 0
	case StateActive:
		s.calibratingSeen = 0
		s.consecutiveZeros = 0
	case StateIdle:
		s.consecutiveZeros = 0
	case StateStale:
		s.consecutiveZeros = 0
		// leave consecutiveFails as-is; it counts failures, not entries.
	}

	s.log.Info("health state transition",
		"prev", string(prev),
		"next", string(next),
		"reason", reason,
	)
	return prev, next, reason, true
}

// updateIdleCounterLocked maintains the consecutive-zero-launch counter
// used for ACTIVE -> IDLE detection. Only relevant when event reads are
// OK — failed reads are handled by the STALE path. Caller must hold s.mu.
func (s *stateMachine) updateIdleCounterLocked(obs Observation) {
	if !obs.EventReadOK {
		return
	}
	if obs.KernelLaunchCount == 0 {
		s.consecutiveZeros++
	} else {
		s.consecutiveZeros = 0
	}
}

// rememberLocked inserts obs into the retention window. Caller must hold
// s.mu.
func (s *stateMachine) rememberLocked(obs Observation) {
	if cap(s.recent) == 0 {
		return
	}
	if len(s.recent) < cap(s.recent) {
		s.recent = append(s.recent, obs)
		return
	}
	copy(s.recent, s.recent[1:])
	s.recent[len(s.recent)-1] = obs
}

// normalizeObservation clamps caller-provided values at the state-machine
// boundary: negative launch counts become 0, far-future timestamps are
// treated as read failures (with a clock-skew tolerance). The state
// machine is the trust boundary between raw collection and classification
// logic, so sanitization lives here rather than at every consumer.
func normalizeObservation(obs Observation) Observation {
	if obs.KernelLaunchCount < 0 {
		obs.KernelLaunchCount = 0
	}
	if !obs.Timestamp.IsZero() && time.Since(obs.Timestamp) < -FutureTolerance {
		// obs is in the future past our tolerance — treat as read failure.
		obs.EventReadOK = false
	}
	return obs
}
