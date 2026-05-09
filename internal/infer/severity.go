package infer

import (
	"strings"
	"sync"
	"time"
)

// severity ranks the correlate package's CausalChain.Severity strings
// as integers so we can compare with >=. The ordering matches the
// causal-chain producer: HIGH > MEDIUM > LOW. Unknown strings rank as
// 0 so a typo or missing severity never accidentally trips a gate.
type severityRank int

const (
	sevNone   severityRank = 0
	sevLow    severityRank = 1
	sevMedium severityRank = 2
	sevHigh   severityRank = 3
)

func parseSeverity(s string) severityRank {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "HIGH":
		return sevHigh
	case "MEDIUM":
		return sevMedium
	case "LOW":
		return sevLow
	default:
		return sevNone
	}
}

// severityEntry records the highest severity observed for a PID and the
// time it was set. Entries TTL out so a stale HIGH from minutes ago does
// not gate fresh baseline updates after the chain has resolved.
type severityEntry struct {
	rank severityRank
	at   time.Time
}

// severityGate maps PIDs to their most recent causal-chain severity.
// Updated from the snapshot loop via Set; queried from the sync-event
// hot path via IsAtLeast. PruneExpired runs from Set so the map never
// grows unboundedly across long-running daemons even if a PID never
// returns to chain-clear state.
//
// The threshold against which IsAtLeast compares is a parameter on the
// caller side; the gate is workload-agnostic and stores raw rank.
type severityGate struct {
	mu  sync.Mutex
	per map[uint32]severityEntry
	ttl time.Duration
}

func newSeverityGate(ttl time.Duration) *severityGate {
	if ttl <= 0 {
		ttl = 30 * time.Second
	}
	return &severityGate{
		per: make(map[uint32]severityEntry),
		ttl: ttl,
	}
}

// Set records severity for pid at time at. Overwrites any prior entry,
// so a chain transitioning from HIGH to MEDIUM is reflected immediately
// (the producer is expected to send the current chain state, not a
// stream of edges). Calling Set with sevNone effectively clears.
func (g *severityGate) Set(pid uint32, sev severityRank, at time.Time) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.pruneExpiredLocked(at)
	if sev == sevNone {
		delete(g.per, pid)
		return
	}
	g.per[pid] = severityEntry{rank: sev, at: at}
}

// IsAtLeast returns true when pid's most recent (non-expired) severity
// is >= threshold. A threshold of sevNone (e.g. caller passed an empty
// "PauseOnSeverity" config) disables gating for everyone — IsAtLeast
// always returns false.
func (g *severityGate) IsAtLeast(pid uint32, threshold severityRank, now time.Time) bool {
	if threshold == sevNone {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	e, ok := g.per[pid]
	if !ok {
		return false
	}
	if now.Sub(e.at) > g.ttl {
		delete(g.per, pid)
		return false
	}
	return e.rank >= threshold
}

// PruneExpired drops entries older than ttl. Test seam; production
// pruning happens lazily inside Set.
func (g *severityGate) PruneExpired(now time.Time) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.pruneExpiredLocked(now)
}

func (g *severityGate) pruneExpiredLocked(now time.Time) {
	for pid, e := range g.per {
		if now.Sub(e.at) > g.ttl {
			delete(g.per, pid)
		}
	}
}

// Len returns the number of live entries. Test-only helper.
func (g *severityGate) Len() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.per)
}
