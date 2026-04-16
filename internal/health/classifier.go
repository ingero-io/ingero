package health

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// StragglerEvent carries everything a downstream consumer (OTLP collector
// or UDS remediation subscriber) needs to react to a self-classification
// transition. The agent produces one per tick while a straggler, plus one
// edge event on recovery.
type StragglerEvent struct {
	NodeID         string
	ClusterID      string
	Score          float64
	Threshold      float64
	DetectionMode  DetectionMode
	DominantSignal string // "throughput" | "compute" | "memory" | "cpu" | "unknown"
	Timestamp      time.Time
}

// StragglerSink is the optional UDS / remediation consumer. The existing
// `internal/remediate.Server` satisfies this interface. A nil sink means
// UDS streaming is disabled (no --remediate flag).
type StragglerSink interface {
	SendStragglerState(ev StragglerEvent) error
	SendStragglerResolved(nodeID, clusterID string, ts time.Time) error
}

// ClassifierConfig tunes classification behavior.
type ClassifierConfig struct {
	// Hysteresis adds a dead-band above the threshold so that a score
	// oscillating right at the threshold doesn't flap between straggler
	// and healthy. Once straggler, the agent stays straggler until
	// score >= threshold + Hysteresis. Default 0.02.
	Hysteresis float64 `yaml:"hysteresis"`
}

// DefaultClassifierConfig returns the canonical values.
func DefaultClassifierConfig() ClassifierConfig {
	return ClassifierConfig{
		Hysteresis: 0.02,
	}
}

// Validate returns an error on invalid configuration.
func (c ClassifierConfig) Validate() error {
	if c.Hysteresis < 0 || c.Hysteresis >= 0.5 {
		return fmt.Errorf("classifier.hysteresis must be in [0, 0.5): got %v", c.Hysteresis)
	}
	return nil
}

// Classification is the last computed outcome.
type Classification struct {
	IsStraggler bool
	ChangedAt   time.Time
}

// Classifier compares a health score against a threshold and decides
// whether the agent is currently a straggler. State is sticky with
// configurable hysteresis.
//
// All methods are safe for concurrent use.
type Classifier interface {
	// Classify folds one (score, threshold, timestamp) sample into the
	// classifier. Returns the current straggler state and whether it
	// changed from the previous call. The classifier does NOT consult
	// the Baseliner's state machine — the caller (Loop) is responsible
	// for skipping Classify when the state machine is not ACTIVE or the
	// detection mode is ModeNone.
	Classify(score, threshold float64, now time.Time) (isStraggler bool, changed bool)
	// LastClassification returns the most recent outcome and the
	// timestamp of the last state change.
	LastClassification() Classification
}

type classifier struct {
	cfg ClassifierConfig

	mu          sync.Mutex
	isStraggler bool
	changedAt   time.Time
}

// NewClassifier constructs a Classifier with the given config.
func NewClassifier(cfg ClassifierConfig) (Classifier, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &classifier{cfg: cfg}, nil
}

// floatEpsilon absorbs float64 round-off at the hysteresis boundary. For
// a threshold like 0.80 and hysteresis 0.02, `threshold + hysteresis` in
// IEEE-754 can be 0.82000000000000006, so the literal score 0.82 would
// fail a strict `>=` comparison. 1e-9 is well below any plausible
// operational threshold resolution.
const floatEpsilon = 1e-9

func (c *classifier) Classify(score, threshold float64, now time.Time) (bool, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	next := c.isStraggler
	if c.isStraggler {
		// Currently straggler: recover only when score climbs above
		// threshold + hysteresis (with epsilon for float round-off).
		if score+floatEpsilon >= threshold+c.cfg.Hysteresis {
			next = false
		}
	} else {
		// Currently healthy: become straggler when score drops below
		// threshold (epsilon keeps exact-boundary scores in healthy).
		if score+floatEpsilon < threshold {
			next = true
		}
	}

	changed := next != c.isStraggler
	if changed {
		c.isStraggler = next
		c.changedAt = now
	}
	return next, changed
}

func (c *classifier) LastClassification() Classification {
	c.mu.Lock()
	defer c.mu.Unlock()
	return Classification{
		IsStraggler: c.isStraggler,
		ChangedAt:   c.changedAt,
	}
}

// DominantSignal returns the name of the most-degraded signal relative
// to the agent's fast EMA baseline. The returned string is one of
// {"throughput", "compute", "memory", "cpu"} or "unknown" when no
// signal is meaningfully below its baseline.
func DominantSignal(current, baseline Baselines) string {
	deltas := map[string]float64{
		"throughput": normalizedDrop(baseline.Throughput, current.Throughput),
		"compute":    normalizedDrop(baseline.Compute, current.Compute),
		"memory":     normalizedDrop(baseline.Memory, current.Memory),
		"cpu":        normalizedDrop(baseline.CPU, current.CPU),
	}
	var winner string
	var maxDrop float64
	for name, d := range deltas {
		if d > maxDrop {
			maxDrop = d
			winner = name
		}
	}
	if winner == "" || maxDrop <= 0 {
		return "unknown"
	}
	return winner
}

// normalizedDrop returns (baseline - current) / baseline, or 0 when
// baseline is zero or current is not below it.
func normalizedDrop(baseline, current float64) float64 {
	if baseline <= 0 || current >= baseline {
		return 0
	}
	return (baseline - current) / baseline
}

// ---------- UDS helpers ----------

// ErrSinkClosed is returned from a StragglerSink when the UDS is not
// accepting messages (e.g., no consumer connected). The caller should
// treat it as advisory; classification itself is not affected.
var ErrSinkClosed = errors.New("straggler sink closed")
