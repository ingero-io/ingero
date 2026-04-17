package main

import (
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestHandleLine_CountsByType(t *testing.T) {
	m := newMetrics()
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
	m := newMetrics()
	handleLine([]byte(`{"type":"straggler_state","node_id":"n1","cluster_id":"c1","score":0.2,"threshold":0.5}`), m)
	if got := m.stragglerState[stragglerKey{"c1", "n1"}]; got != 1 {
		t.Errorf("after straggler_state: gauge=%d, want 1", got)
	}

	handleLine([]byte(`{"type":"straggler_resolved","node_id":"n1","cluster_id":"c1"}`), m)
	if got := m.stragglerState[stragglerKey{"c1", "n1"}]; got != 0 {
		t.Errorf("after straggler_resolved: gauge=%d, want 0", got)
	}
}

func TestHandleLine_MultipleClustersIsolated(t *testing.T) {
	m := newMetrics()
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
	m := newMetrics()
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
	m := newMetrics()
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
	m := newMetrics()
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
