// Package tcpretransmit aggregates per-PID retransmit events from the
// existing internal/ebpf/tcp BTF tracepoint tracer into a sustained-rate
// signal suitable for orchestrator dispatch.
//
// The eBPF layer fires once per `tcp_retransmit_skb` call (typically a
// handful per second on a healthy host, into the hundreds during a
// real retransmit storm). The orchestrator's TcpRetransmitStorm
// dispatch chain (drain_lb_endpoint -> pod_drain) only wants one
// signal per episode per PID, so this package sits between the raw
// event stream and the remediate UDS:
//
//   - Observe(pid, ts)         called per ring-buffer record
//   - Sweep(now) -> []Storm    drained on a fixed tick (default 1s)
//
// Sweep returns at most one Storm per PID per episode. An episode
// starts when a PID's per-second rate first crosses the configured
// threshold, ends when the PID stays below threshold for a full
// suppression window, and rearms after that quiet window so the next
// elevated period gets a fresh signal.
//
// Anchor: catalog row I17 (TCP retransmit storm). The threshold +
// sustain defaults are sized to filter steady-state baseline traffic
// while catching the storm shape documented in the catalog: hundreds
// of retransmits/sec sustained for tens of seconds while one host's
// network path degrades.
package tcpretransmit

import (
	"sync"
	"time"
)

// DefaultRateThreshold is the per-PID retransmit rate above which the
// PID is considered "in storm". 20 events/sec is well above the ~0
// baseline a healthy steady-state TCP connection holds and below the
// hundreds-per-sec rate observed during a sustained path-degradation
// incident; the threshold catches the storm shape without firing on
// transient bursts.
const DefaultRateThreshold = 20.0

// DefaultSustainedThreshold is the minimum continuous time above the
// rate threshold required to emit a Storm. Three seconds rejects
// micro-bursts (a single ring-buffer flush spike) while catching a
// real storm well before user-visible latency damage.
const DefaultSustainedThreshold = 3 * time.Second

// DefaultSuppressionWindow is how long a PID must stay quiet (below
// threshold) after a storm ends before a fresh episode can be
// declared. Without this, a flapping PID at the threshold edge would
// re-emit every Sweep tick.
const DefaultSuppressionWindow = 30 * time.Second

// WindowSeconds is the sliding-window size (in seconds) over which
// the per-PID rate is computed. Keeping this short (5s) means the
// rate tracks the current condition rather than averaging in a long
// quiet tail before the storm started, which would delay detection.
const WindowSeconds = 5

// Storm is the per-episode summary emitted by Sweep. Maps 1:1 onto
// the orchestrator's TcpRetransmitStormState wire message.
type Storm struct {
	PID         uint32
	RatePerSec  float64
	SustainedMs uint64
}

// Tracker aggregates retransmit events per-PID and emits one Storm
// per sustained episode. Safe for concurrent Observe + Sweep callers.
type Tracker struct {
	mu                sync.Mutex
	pids              map[uint32]*pidState
	rateThreshold     float64
	sustainedThresh   time.Duration
	suppressionWindow time.Duration
}

type pidState struct {
	// Bucket counts indexed by (unixSecond % WindowSeconds). lastSecond
	// tracks the most recent bucket index seen; older buckets are
	// zeroed on advance.
	buckets    [WindowSeconds]uint32
	lastSecond int64
	// elevatedSince is non-zero when the PID is currently in an
	// elevated run (rate > threshold for at least one observation).
	// Sweep emits a Storm once elevatedSince has been continuous for
	// sustainedThresh; the emitted flag prevents re-emission until
	// the PID has gone quiet for suppressionWindow.
	elevatedSince time.Time
	emitted       bool
	// quietSince is when the PID first dropped below threshold after
	// an emitted storm. Used to gate the suppressionWindow rearm.
	quietSince time.Time
}

// New returns a Tracker with default thresholds. Tests can construct
// with NewWithThresholds for deterministic windows.
func New() *Tracker {
	return NewWithThresholds(DefaultRateThreshold, DefaultSustainedThreshold, DefaultSuppressionWindow)
}

// NewWithThresholds returns a Tracker with custom thresholds. Useful
// for tests that want short sustain / suppression windows.
func NewWithThresholds(rateThreshold float64, sustained time.Duration, suppression time.Duration) *Tracker {
	return &Tracker{
		pids:              make(map[uint32]*pidState),
		rateThreshold:     rateThreshold,
		sustainedThresh:   sustained,
		suppressionWindow: suppression,
	}
}

// Observe records a single retransmit event for pid at wall-clock ts.
// pid==0 is dropped (kernel idle / unknown attribution); negative time
// is silently treated as "now" semantics by the bucket index.
func (t *Tracker) Observe(pid uint32, ts time.Time) {
	if pid == 0 {
		return
	}
	sec := ts.Unix()
	idx := int(sec % WindowSeconds)
	if idx < 0 {
		idx += WindowSeconds
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	st, ok := t.pids[pid]
	if !ok {
		st = &pidState{lastSecond: sec}
		t.pids[pid] = st
	}

	if sec != st.lastSecond {
		t.advanceLocked(st, sec)
	}
	st.buckets[idx]++
}

// advanceLocked zeroes any bucket positions that should have rolled
// off between st.lastSecond and now. When the gap exceeds the window
// width, the whole array is zeroed in one shot — equivalent and
// cheaper than looping WindowSeconds-1 times.
func (t *Tracker) advanceLocked(st *pidState, now int64) {
	gap := now - st.lastSecond
	if gap <= 0 {
		st.lastSecond = now
		return
	}
	if gap >= WindowSeconds {
		for i := range st.buckets {
			st.buckets[i] = 0
		}
	} else {
		for i := int64(1); i <= gap; i++ {
			idx := int((st.lastSecond + i) % WindowSeconds)
			st.buckets[idx] = 0
		}
	}
	st.lastSecond = now
}

// Sweep drains the tracker at wall-clock now and returns one Storm
// per PID whose elevated run has crossed the sustain threshold this
// tick and hasn't been emitted yet. Quiet PIDs are also state-machined
// here: if a previously-emitted PID has stayed below threshold for
// suppressionWindow, its emitted flag clears so the next storm gets
// a fresh signal. PIDs that drift below threshold without ever
// having emitted are quietly reset (their elevatedSince clears),
// preserving "must be continuously elevated for sustainedThresh"
// without requiring epoch tracking.
//
// Returned Storms are stable across re-Sweep calls within one second
// (the PID's emitted flag stays set until the suppression rearm).
func (t *Tracker) Sweep(now time.Time) []Storm {
	t.mu.Lock()
	defer t.mu.Unlock()

	var out []Storm
	for pid, st := range t.pids {
		t.advanceLocked(st, now.Unix())
		rate := t.rateLocked(st)

		if rate > t.rateThreshold {
			// Above threshold this tick.
			if st.elevatedSince.IsZero() {
				st.elevatedSince = now
			}
			st.quietSince = time.Time{}
			if !st.emitted && now.Sub(st.elevatedSince) >= t.sustainedThresh {
				out = append(out, Storm{
					PID:         pid,
					RatePerSec:  rate,
					SustainedMs: uint64(now.Sub(st.elevatedSince).Milliseconds()),
				})
				st.emitted = true
			}
		} else {
			// Below threshold. Drop the elevated-since stamp so the
			// next elevated period must be continuously sustained
			// again before emission. Rearm the suppression window if
			// we previously emitted.
			st.elevatedSince = time.Time{}
			if st.emitted {
				if st.quietSince.IsZero() {
					st.quietSince = now
				} else if now.Sub(st.quietSince) >= t.suppressionWindow {
					st.emitted = false
					st.quietSince = time.Time{}
				}
			}
		}
	}
	return out
}

// rateLocked returns the per-second rate (events / WindowSeconds)
// averaged across the full window. Caller must hold t.mu.
func (t *Tracker) rateLocked(st *pidState) float64 {
	var sum uint32
	for _, c := range st.buckets {
		sum += c
	}
	return float64(sum) / float64(WindowSeconds)
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
