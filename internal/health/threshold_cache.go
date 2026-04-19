package health

import (
	"log/slog"
	"math"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ingero-io/ingero/pkg/contract"
)

// warnLogMinInterval rate-limits per-kind WARN logs so a broken Fleet
// doesn't flood at the push cadence.
const warnLogMinInterval = 5 * time.Minute

// Sanity bounds for threshold values delivered by Fleet. Values outside
// this range are rejected — 0.0 would mean "nobody is a straggler" and
// 1.0 would mean "everybody is a straggler", both of which indicate a
// bug on the Fleet side, not a legitimate threshold.
const (
	thresholdMin = 0.1
	thresholdMax = 0.99
)

// ThresholdCache holds the last threshold value received from Fleet,
// along with when it arrived and whether Fleet's statistical quorum was
// met at that point.
//
// Writers: the Emitter (after each push response) and the Poller (after
// each GET response). Readers: Story 3.3's ModeEvaluator.
//
// All methods are safe for concurrent use.
type ThresholdCache struct {
	mu                 sync.RWMutex
	value              float64
	quorumMet          bool
	ever               bool // true once any valid value has been observed
	receivedAt         time.Time
	piggybackAvailable bool // headers were present on most recent push

	// Observability counters.
	//
	// hits      - valid header or GET payload consumed into the cache
	// misses    - no headers at all (absent); poller should take over
	// rejected  - headers present but unusable (out-of-bounds value,
	//             malformed float, malformed bool). Piggyback is still
	//             marked as "available" because the server did attempt
	//             to deliver a threshold -- the poller stays quiet.
	hits     atomic.Int64
	misses   atomic.Int64
	rejected atomic.Int64

	// Rate-limited logging for malformed-value paths. Last-log time is
	// protected by its own lock so a WARN decision doesn't contend with
	// the primary cache read/write path.
	logMu         sync.Mutex
	lastWarnAt    map[string]time.Time
	log           *slog.Logger
}

// NewThresholdCache returns an empty cache. Safe to use immediately;
// Get returns ok=false until Set is called. Pass a nil logger to use
// slog.Default() for malformed-value WARN logs.
func NewThresholdCache() *ThresholdCache {
	return &ThresholdCache{
		lastWarnAt: make(map[string]time.Time),
		log:        slog.Default(),
	}
}

// SetLogger overrides the default slog used for malformed-value WARN
// logs. Intended for wiring up the agent's configured logger; safe to
// call once at construction time (no concurrent callers expected).
func (c *ThresholdCache) SetLogger(log *slog.Logger) {
	if log == nil {
		return
	}
	c.log = log
}

// ThresholdSnapshot is the immutable view returned by Get.
type ThresholdSnapshot struct {
	Value      float64
	QuorumMet  bool
	ReceivedAt time.Time
}

// Get returns the last valid threshold the cache has seen. `ok` is false
// if no valid threshold has ever been received. `snap.QuorumMet` reflects
// the quorum flag Fleet advertised with that value.
func (c *ThresholdCache) Get() (snap ThresholdSnapshot, ok bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if !c.ever {
		return ThresholdSnapshot{}, false
	}
	return ThresholdSnapshot{
		Value:      c.value,
		QuorumMet:  c.quorumMet,
		ReceivedAt: c.receivedAt,
	}, true
}

// Set records a new threshold value. Callers have already parsed and
// bounds-checked via ParseAndSet; this is the low-level setter for tests
// and the poller.
func (c *ThresholdCache) Set(value float64, quorumMet bool, now time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.value = value
	c.quorumMet = quorumMet
	c.ever = true
	c.receivedAt = now
}

// PiggybackAvailable returns true when the most recent push response
// carried threshold headers (usable or not). Story 3.2's poller consults
// this to decide whether to issue GET fallback requests.
func (c *ThresholdCache) PiggybackAvailable() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.piggybackAvailable
}

// MarkPiggybackUnavailable is called by the Emitter after a push whose
// response did not include threshold headers. The poller will resume on
// the next tick. Kept as a public helper for tests; internal code paths
// use the lower-level markAbsent/markRejected functions directly.
func (c *ThresholdCache) MarkPiggybackUnavailable() {
	c.markAbsent()
}

// Stats returns cumulative observation counters.
//
//   - hits     valid headers or GET payload consumed.
//   - misses   no headers present (emitter response empty).
//   - rejected headers present but unusable (out-of-bounds, malformed
//     float, malformed bool). Piggyback remains marked available.
func (c *ThresholdCache) Stats() (hits, misses, rejected int64) {
	return c.hits.Load(), c.misses.Load(), c.rejected.Load()
}

// ParseAndSetHTTPHeaders reads the two contract headers from h. The h
// argument may be nil (treated as empty headers). Unified outcomes:
//
//   - Both headers absent: `piggybackAvailable = false`, miss counter.
//   - Partial pair (one present, one empty): treated as absent.
//   - Headers present but unusable (malformed float/bool, out-of-bounds
//     value): `piggybackAvailable = true`, rejected counter, cache value
//     unchanged. Malformed values also log a rate-limited WARN.
//   - Valid headers: cache updated, `piggybackAvailable = true`, hit
//     counter.
//
// quorum_met = "false" is a legitimate server response; the cache stores
// the (value, false) pair so Story 3.3's ModeEvaluator sees quorum=false
// and selects `fleet-cached`. This is NOT the "rejected" path.
//
// The caller should pass the `now` clock so tests stay deterministic.
func (c *ThresholdCache) ParseAndSetHTTPHeaders(h httpHeaderGetter, now time.Time) (ok bool) {
	if h == nil {
		c.markAbsent()
		return false
	}
	tRaw := h.Get(contract.HeaderThreshold)
	qRaw := h.Get(contract.HeaderQuorumMet)
	return c.parseAndSet(tRaw, qRaw, now)
}

// parseAndSet is the shared parse/apply path for HTTP headers and GET
// responses. Returns true if the threshold was applied to the cache.
func (c *ThresholdCache) parseAndSet(tRaw, qRaw string, now time.Time) bool {
	tRaw = strings.TrimSpace(tRaw)
	qRaw = strings.TrimSpace(qRaw)
	if tRaw == "" && qRaw == "" {
		c.markAbsent()
		return false
	}
	if tRaw == "" || qRaw == "" {
		// Partial pair — treat as absent.
		c.markAbsent()
		return false
	}

	value, err := strconv.ParseFloat(tRaw, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
		c.markRejected("malformed threshold", "value", tRaw)
		return false
	}

	quorumMet, qOK := parseBoolStrict(qRaw)
	if !qOK {
		c.markRejected("malformed quorum_met", "value", qRaw)
		return false
	}

	// Piggyback was *available* (headers were present), but the value may
	// still fail sanity bounds. Mark rejected + piggyback=true; the
	// cache's previous value is preserved so Story 3.3 can still serve
	// fleet-cached with an older (now-stale-timestamped) threshold.
	if value < thresholdMin || value > thresholdMax {
		c.markRejected("threshold out of sanity bounds", "value", value,
			"min", thresholdMin, "max", thresholdMax)
		return false
	}

	c.hits.Add(1)
	c.mu.Lock()
	c.value = value
	c.quorumMet = quorumMet
	c.ever = true
	c.receivedAt = now
	c.piggybackAvailable = true
	c.mu.Unlock()
	return true
}

// markAbsent handles the "no headers, no threshold" case: piggyback is
// no longer available, miss counter increments so the poller can take
// over.
func (c *ThresholdCache) markAbsent() {
	c.mu.Lock()
	c.piggybackAvailable = false
	c.mu.Unlock()
	c.misses.Add(1)
}

// markRejected handles the "headers present but unusable" case:
// piggyback stays true (server attempted), rejected counter increments,
// and a rate-limited WARN is logged.
func (c *ThresholdCache) markRejected(reason string, kv ...any) {
	c.mu.Lock()
	c.piggybackAvailable = true
	c.mu.Unlock()
	c.rejected.Add(1)
	c.warnOnce(reason, kv...)
}

// warnOnce logs a WARN at most once per warnLogMinInterval per reason.
// Designed for malformed/out-of-bounds payloads from a buggy Fleet so
// the log isn't flooded at the push cadence.
func (c *ThresholdCache) warnOnce(reason string, kv ...any) {
	if c.log == nil {
		return
	}
	c.logMu.Lock()
	last := c.lastWarnAt[reason]
	now := time.Now()
	if !last.IsZero() && now.Sub(last) < warnLogMinInterval {
		c.logMu.Unlock()
		return
	}
	c.lastWarnAt[reason] = now
	c.logMu.Unlock()
	args := append([]any{"reason", reason}, kv...)
	c.log.Warn("threshold cache: rejecting unusable value from Fleet", args...)
}

// httpHeaderGetter is the narrow interface Emitter calls with a real
// http.Header. Tests pass a map-backed stub.
type httpHeaderGetter interface {
	Get(key string) string
}

// parseBoolStrict accepts only "true" / "false" (case-insensitive,
// already trimmed) — never "yes", "1", "y". Rejecting lenient forms
// prevents the cache from accepting an ambiguous header value.
func parseBoolStrict(s string) (bool, bool) {
	switch strings.ToLower(s) {
	case "true":
		return true, true
	case "false":
		return false, true
	default:
		return false, false
	}
}
