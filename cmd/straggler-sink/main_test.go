package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestHandleLine_CountsByType(t *testing.T) {
	m := newMetrics(0)
	handleLine([]byte(`{"type":"memory","pid":123}`), m)
	handleLine([]byte(`{"type":"memory","pid":456}`), m)
	handleLine([]byte(`{"type":"straggle","pid":789}`), m)

	if got := m.eventsByType["memory"]; got != 2 {
		t.Errorf("memory count=%d, want 2", got)
	}
	if got := m.eventsByType["straggle"]; got != 1 {
		t.Errorf("straggle count=%d, want 1", got)
	}
}

func TestHandleLine_StragglerStateTransitions(t *testing.T) {
	m := newMetrics(0)
	handleLine([]byte(`{"type":"straggler_state","node_id":"n1","cluster_id":"c1","score":0.2,"threshold":0.5}`), m)
	if got := m.stragglerState[stragglerKey{"c1", "n1"}]; got != 1 {
		t.Errorf("after straggler_state: gauge=%d, want 1", got)
	}

	handleLine([]byte(`{"type":"straggler_resolved","node_id":"n1","cluster_id":"c1"}`), m)
	// Entry must be deleted on resolve, not zeroed, so the gauge stops
	// emitting a series for it.
	if _, present := m.stragglerState[stragglerKey{"c1", "n1"}]; present {
		t.Errorf("after straggler_resolved: key still present, want deleted")
	}
	if _, present := m.index[stragglerKey{"c1", "n1"}]; present {
		t.Errorf("after straggler_resolved: key still in FIFO index, want deleted")
	}
}

func TestHandleLine_MultipleClustersIsolated(t *testing.T) {
	m := newMetrics(0)
	handleLine([]byte(`{"type":"straggler_state","node_id":"n1","cluster_id":"prod"}`), m)
	handleLine([]byte(`{"type":"straggler_state","node_id":"n1","cluster_id":"staging"}`), m)
	handleLine([]byte(`{"type":"straggler_resolved","node_id":"n1","cluster_id":"prod"}`), m)

	if got := m.stragglerState[stragglerKey{"prod", "n1"}]; got != 0 {
		t.Errorf("prod gauge=%d, want 0", got)
	}
	if got := m.stragglerState[stragglerKey{"staging", "n1"}]; got != 1 {
		t.Errorf("staging gauge=%d, want 1 (not affected by prod resolution)", got)
	}
}

func TestHandleLine_ParseError(t *testing.T) {
	m := newMetrics(0)
	handleLine([]byte(`not-json`), m)
	handleLine([]byte(`{"type":""}`), m) // empty type also counts as parse error
	handleLine([]byte{}, m)              // empty line is a no-op

	if got := atomic.LoadUint64(&m.parseErrors); got != 2 {
		t.Errorf("parseErrors=%d, want 2", got)
	}
	if len(m.eventsByType) != 0 {
		t.Errorf("eventsByType=%v, want empty", m.eventsByType)
	}
}

func TestMetricsOutput_Exposition(t *testing.T) {
	m := newMetrics(0)
	handleLine([]byte(`{"type":"memory","pid":1}`), m)
	handleLine([]byte(`{"type":"straggler_state","node_id":"n1","cluster_id":"c1"}`), m)
	m.setConnected(true)

	rec := httptest.NewRecorder()
	m.handleMetrics(rec, nil)

	body := rec.Body.String()
	wantSubstrings := []string{
		`ingero_sink_events_total{type="memory"} 1`,
		`ingero_sink_events_total{type="straggler_state"} 1`,
		`ingero_sink_connected 1`,
		`ingero_sink_active_stragglers{cluster_id="c1",node_id="n1"} 1`,
		`# TYPE ingero_sink_events_total counter`,
		`# TYPE ingero_sink_active_stragglers gauge`,
	}
	for _, want := range wantSubstrings {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing %q\nfull output:\n%s", want, body)
		}
	}

	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("Content-Type=%q, want text/plain prefix", ct)
	}
}

func TestMetricsOutput_DeterministicOrder(t *testing.T) {
	m := newMetrics(0)
	// Insert in reverse-sort order to check the handler sorts.
	handleLine([]byte(`{"type":"zebra"}`), m)
	handleLine([]byte(`{"type":"apple"}`), m)
	handleLine([]byte(`{"type":"middle"}`), m)

	rec := httptest.NewRecorder()
	m.handleMetrics(rec, nil)
	body := rec.Body.String()

	iApple := strings.Index(body, `type="apple"`)
	iMiddle := strings.Index(body, `type="middle"`)
	iZebra := strings.Index(body, `type="zebra"`)
	if iApple < 0 || iMiddle < 0 || iZebra < 0 {
		t.Fatalf("expected all three types in output:\n%s", body)
	}
	if !(iApple < iMiddle && iMiddle < iZebra) {
		t.Errorf("event types not sorted alphabetically: apple=%d middle=%d zebra=%d", iApple, iMiddle, iZebra)
	}
}

// The gauge series must be removed entirely on resolve, not just zeroed
// — otherwise cardinality grows unboundedly in a high-churn fleet.
func TestStragglerState_DeletesOnResolve(t *testing.T) {
	m := newMetrics(0)
	handleLine([]byte(`{"type":"straggler_state","node_id":"n1","cluster_id":"c1"}`), m)
	handleLine([]byte(`{"type":"straggler_resolved","node_id":"n1","cluster_id":"c1"}`), m)

	if len(m.stragglerState) != 0 {
		t.Errorf("stragglerState not empty after resolve: %v", m.stragglerState)
	}
	if len(m.index) != 0 {
		t.Errorf("FIFO index not empty after resolve: %v", m.index)
	}
	if m.order.Len() != 0 {
		t.Errorf("FIFO order not empty after resolve: len=%d", m.order.Len())
	}

	rec := httptest.NewRecorder()
	m.handleMetrics(rec, nil)
	if strings.Contains(rec.Body.String(), `ingero_sink_active_stragglers{cluster_id="c1"`) {
		t.Errorf("gauge series still emitted after resolve:\n%s", rec.Body.String())
	}
}

// At cap, inserting a new active straggler evicts the oldest (FIFO) and
// bumps the dropped counter with reason cap_reached.
func TestStragglerState_EvictsAtCap(t *testing.T) {
	cap := 3
	m := newMetrics(cap)
	for i := 0; i < cap; i++ {
		handleLine([]byte(fmt.Sprintf(`{"type":"straggler_state","node_id":"n%d","cluster_id":"c"}`, i)), m)
	}
	if got := len(m.stragglerState); got != cap {
		t.Fatalf("pre-eviction size=%d, want %d", got, cap)
	}

	// Insert one more; oldest (n0) should be evicted.
	handleLine([]byte(`{"type":"straggler_state","node_id":"n99","cluster_id":"c"}`), m)

	if got := len(m.stragglerState); got != cap {
		t.Errorf("post-eviction size=%d, want %d", got, cap)
	}
	if _, present := m.stragglerState[stragglerKey{"c", "n0"}]; present {
		t.Errorf("oldest key n0 should have been evicted, still present")
	}
	if _, present := m.stragglerState[stragglerKey{"c", "n99"}]; !present {
		t.Errorf("newest key n99 should have been inserted, not present")
	}
	if got := m.dropped["cap_reached"]; got != 1 {
		t.Errorf("dropped[cap_reached]=%d, want 1", got)
	}
}

// Re-sending straggler_state for an already-tracked key is a no-op
// (no eviction, no dropped count).
func TestStragglerState_DedupeDoesNotEvict(t *testing.T) {
	cap := 2
	m := newMetrics(cap)
	handleLine([]byte(`{"type":"straggler_state","node_id":"n1","cluster_id":"c"}`), m)
	handleLine([]byte(`{"type":"straggler_state","node_id":"n2","cluster_id":"c"}`), m)
	// Re-send n1 — at cap, but this is a dedupe, not an insert.
	handleLine([]byte(`{"type":"straggler_state","node_id":"n1","cluster_id":"c"}`), m)

	if got := len(m.stragglerState); got != 2 {
		t.Errorf("size=%d after dedupe, want 2", got)
	}
	if got := m.dropped["cap_reached"]; got != 0 {
		t.Errorf("dropped[cap_reached]=%d after dedupe, want 0", got)
	}
}

// The dropped counter is exposed on /metrics with the expected label.
func TestMetricsOutput_DroppedCounter(t *testing.T) {
	m := newMetrics(1)
	handleLine([]byte(`{"type":"straggler_state","node_id":"n1","cluster_id":"c"}`), m)
	handleLine([]byte(`{"type":"straggler_state","node_id":"n2","cluster_id":"c"}`), m) // evicts n1

	rec := httptest.NewRecorder()
	m.handleMetrics(rec, nil)
	body := rec.Body.String()
	if !strings.Contains(body, `ingero_sink_dropped_total{reason="cap_reached"} 1`) {
		t.Errorf("missing dropped counter line:\n%s", body)
	}
	if !strings.Contains(body, `# TYPE ingero_sink_dropped_total counter`) {
		t.Errorf("missing TYPE line for dropped counter:\n%s", body)
	}
}

// TestReadyzStatus_ConnectedReturns200 covers the happy path:
// while UDS is connected, /readyz must return 200 regardless of
// last-event freshness. QA audit ★4 #5 / project_qa_test_audit_2026-05-02.md.
func TestReadyzStatus_ConnectedReturns200(t *testing.T) {
	m := newMetrics(1024)
	m.setConnected(true)
	code, body := readyzStatus(m, 60*time.Second, time.Now())
	if code != http.StatusOK {
		t.Errorf("connected: code=%d, want 200", code)
	}
	if !strings.Contains(body, "ready") {
		t.Errorf("connected: body=%q, want 'ready'", body)
	}
}

// TestReadyzStatus_DisconnectedReturns503AfterWindow asserts the K8s
// readiness gate flips to 503 once the sink has been disconnected
// AND the last event is older than the readiness window. K8s uses
// this to stop routing traffic to a stale sidecar.
func TestReadyzStatus_DisconnectedReturns503AfterWindow(t *testing.T) {
	m := newMetrics(1024)
	m.setConnected(false)
	// Last event 90s ago, window is 60s -> stale -> 503.
	now := time.Now()
	atomic.StoreInt64(&m.lastEventUnixNano, now.Add(-90*time.Second).UnixNano())
	code, body := readyzStatus(m, 60*time.Second, now)
	if code != http.StatusServiceUnavailable {
		t.Errorf("disconnected+stale: code=%d, want 503", code)
	}
	if !strings.Contains(body, "uds_disconnected") {
		t.Errorf("disconnected+stale: body=%q, want 'uds_disconnected'", body)
	}
}

// TestReadyzStatus_DisconnectedRecentEventReturns200 asserts the
// grace period: even when UDS is disconnected, a recent event keeps
// /readyz at 200 so a brief reconnect blip doesn't bounce traffic.
func TestReadyzStatus_DisconnectedRecentEventReturns200(t *testing.T) {
	m := newMetrics(1024)
	m.setConnected(false)
	now := time.Now()
	atomic.StoreInt64(&m.lastEventUnixNano, now.Add(-10*time.Second).UnixNano())
	code, body := readyzStatus(m, 60*time.Second, now)
	if code != http.StatusOK {
		t.Errorf("disconnected+recent: code=%d, want 200", code)
	}
	if !strings.Contains(body, "recent event") {
		t.Errorf("disconnected+recent: body=%q, want 'recent event'", body)
	}
}

// TestReadyzStatus_NeverConnectedReturns503 asserts the cold-start
// path: /readyz on a freshly-started sink that has not yet seen
// any UDS connection AND no event must return 503.
func TestReadyzStatus_NeverConnectedReturns503(t *testing.T) {
	m := newMetrics(1024)
	code, _ := readyzStatus(m, 60*time.Second, time.Now())
	if code != http.StatusServiceUnavailable {
		t.Errorf("cold-start: code=%d, want 503", code)
	}
}

// TestSetConnectedToggle is a small companion test asserting the
// atomic flag round-trips correctly. Defends against a bit-pattern
// regression where setConnected(false) leaves connected != 0.
func TestSetConnectedToggle(t *testing.T) {
	m := newMetrics(1024)
	m.setConnected(true)
	if got := atomic.LoadUint32(&m.connected); got != 1 {
		t.Errorf("after setConnected(true), connected=%d, want 1", got)
	}
	m.setConnected(false)
	if got := atomic.LoadUint32(&m.connected); got != 0 {
		t.Errorf("after setConnected(false), connected=%d, want 0", got)
	}
}
