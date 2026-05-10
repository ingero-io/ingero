package export

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/contract"
)

// otlpTracesPayloadOut is the receiver-side decode shape - mirrors
// otlpTracesPayload but lives in the test so the production type can
// stay tightly internal.
type otlpTracesPayloadOut struct {
	ResourceSpans []struct {
		Resource struct {
			Attributes []struct {
				Key   string                 `json:"key"`
				Value map[string]interface{} `json:"value"`
			} `json:"attributes"`
		} `json:"resource"`
		ScopeSpans []struct {
			Scope struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			} `json:"scope"`
			Spans []struct {
				TraceID           string `json:"traceId"`
				SpanID            string `json:"spanId"`
				Name              string `json:"name"`
				Kind              int    `json:"kind"`
				StartTimeUnixNano string `json:"startTimeUnixNano"`
				EndTimeUnixNano   string `json:"endTimeUnixNano"`
				Status            struct {
					Code    int    `json:"code"`
					Message string `json:"message"`
				} `json:"status"`
				Attributes []struct {
					Key   string                 `json:"key"`
					Value map[string]interface{} `json:"value"`
				} `json:"attributes"`
			} `json:"spans"`
		} `json:"scopeSpans"`
	} `json:"resourceSpans"`
}

func TestPushSpans_RoundtripDecodeMatchesInput(t *testing.T) {
	// End-to-end: build a synthetic OutlierSpan, push to a fake OTLP
	// /v1/traces receiver, decode the JSON body, and confirm the
	// span carries the expected workload-key + bucket + ratios +
	// status. Drives both the encoder and the HTTP shim so a
	// regression in either surface fails the test.
	var (
		mu      sync.Mutex
		bodies  [][]byte
		gotPath string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, body)
		gotPath = r.URL.Path
		mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	exp := NewOTLP(OTLPConfig{
		Endpoint:  srv.URL,
		Insecure:  true,
		NodeID:    "test-node",
		ClusterID: "test-cluster",
	})
	if exp == nil {
		t.Fatal("NewOTLP returned nil")
	}

	end := time.Now()
	in := stats.OutlierSpan{
		EventID:        "11111111-2222-3333-4444-555555555555",
		Bucket:         "3x",
		StepStart:      end.Add(-50 * time.Millisecond),
		StepEnd:        end,
		StepDurationNs: 50 * 1000 * 1000,
		BaselineP95Ns:  10 * 1000 * 1000,
		BaselineMeanNs: 8 * 1000 * 1000,
		CGroupHash:     "abc123",
		PID:            4242,
		StreamHandle:   0xdeadbeef,
		Phase:          "decode",
		MemfragEvents:  3,
		ThrottleReasons: 0x0000000000000040, // HW_SLOWDOWN reason bit
		ModelName:    "meta-llama/Llama-3-7b",
		EngineSystem: "vllm",
	}
	if err := exp.PushSpans([]stats.OutlierSpan{in}); err != nil {
		t.Fatalf("PushSpans: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if gotPath != "/v1/traces" {
		t.Errorf("POST path = %q, want /v1/traces", gotPath)
	}
	if len(bodies) != 1 {
		t.Fatalf("receiver got %d bodies, want 1", len(bodies))
	}
	var p otlpTracesPayloadOut
	if err := json.Unmarshal(bodies[0], &p); err != nil {
		t.Fatalf("decode body: %v\nbody=%s", err, string(bodies[0]))
	}
	if len(p.ResourceSpans) != 1 || len(p.ResourceSpans[0].ScopeSpans) != 1 {
		t.Fatalf("unexpected envelope shape: %+v", p)
	}
	scope := p.ResourceSpans[0].ScopeSpans[0]
	if len(scope.Spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(scope.Spans))
	}
	got := scope.Spans[0]

	// Trace + span IDs are random hex; check shape/length only.
	if len(got.TraceID) != 32 {
		t.Errorf("traceId len = %d, want 32 (hex)", len(got.TraceID))
	}
	if len(got.SpanID) != 16 {
		t.Errorf("spanId len = %d, want 16 (hex)", len(got.SpanID))
	}
	if got.Name != "infer.outlier.3x" {
		t.Errorf("Name = %q, want infer.outlier.3x", got.Name)
	}
	if got.Kind != otlpSpanKindInternal {
		t.Errorf("Kind = %d, want %d", got.Kind, otlpSpanKindInternal)
	}
	if got.Status.Code != otlpStatusCodeError {
		t.Errorf("Status.Code = %d, want %d (ERROR)", got.Status.Code, otlpStatusCodeError)
	}

	// Resource attrs must include cluster + node identity for
	// downstream peer-MAD.
	rattrs := map[string]string{}
	for _, a := range p.ResourceSpans[0].Resource.Attributes {
		if v, ok := a.Value["stringValue"].(string); ok {
			rattrs[a.Key] = v
		}
	}
	if rattrs[contract.AttrNodeID] != "test-node" {
		t.Errorf("resource.AttrNodeID = %q, want test-node", rattrs[contract.AttrNodeID])
	}
	if rattrs[contract.AttrClusterID] != "test-cluster" {
		t.Errorf("resource.AttrClusterID = %q, want test-cluster", rattrs[contract.AttrClusterID])
	}

	// Span attrs collapse into a key->value map for assertion.
	sattrs := map[string]interface{}{}
	for _, a := range got.Attributes {
		if v, ok := a.Value["stringValue"].(string); ok {
			sattrs[a.Key] = v
			continue
		}
		if v, ok := a.Value["intValue"].(string); ok {
			// JSON numbers > 2^53 are encoded as strings via the
			// OTLP/JSON convention; unmarshal-target type happens
			// to keep them as strings here.
			sattrs[a.Key] = v
			continue
		}
		// fall through: numeric int or other
		sattrs[a.Key] = a.Value
	}
	if sattrs[contract.AttrInferOutlierBucket] != "3x" {
		t.Errorf("AttrInferOutlierBucket = %v, want 3x", sattrs[contract.AttrInferOutlierBucket])
	}
	if sattrs[contract.AttrInferPhase] != "decode" {
		t.Errorf("AttrInferPhase = %v, want decode", sattrs[contract.AttrInferPhase])
	}
	if sattrs[contract.AttrCgroupPathHash] != "abc123" {
		t.Errorf("AttrCgroupPathHash = %v, want abc123", sattrs[contract.AttrCgroupPathHash])
	}
	if sattrs[contract.AttrEventID] != "11111111-2222-3333-4444-555555555555" {
		t.Errorf("AttrEventID mismatch: got %v", sattrs[contract.AttrEventID])
	}
	if sattrs[contract.AttrGenAIRequestModel] != "meta-llama/Llama-3-7b" {
		t.Errorf("AttrGenAIRequestModel = %v", sattrs[contract.AttrGenAIRequestModel])
	}
	if sattrs[contract.AttrGenAISystem] != "vllm" {
		t.Errorf("AttrGenAISystem = %v", sattrs[contract.AttrGenAISystem])
	}
}

func TestPushSpans_EmptyInputIsNoop(t *testing.T) {
	called := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called++
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	exp := NewOTLP(OTLPConfig{Endpoint: srv.URL, Insecure: true})
	if err := exp.PushSpans(nil); err != nil {
		t.Errorf("nil input: %v", err)
	}
	if err := exp.PushSpans([]stats.OutlierSpan{}); err != nil {
		t.Errorf("empty input: %v", err)
	}
	if called != 0 {
		t.Errorf("receiver called %d times, want 0", called)
	}
}

func TestNewTraceAndSpanIDs_DistinctEachCall(t *testing.T) {
	t1, s1 := newTraceAndSpanIDs()
	t2, s2 := newTraceAndSpanIDs()
	if t1 == t2 {
		t.Error("two consecutive trace ids collided")
	}
	if s1 == s2 {
		t.Error("two consecutive span ids collided")
	}
	if len(t1) != 32 || len(t2) != 32 {
		t.Errorf("traceId lengths: %d, %d (want 32)", len(t1), len(t2))
	}
	if len(s1) != 16 || len(s2) != 16 {
		t.Errorf("spanId lengths: %d, %d (want 16)", len(s1), len(s2))
	}
}

func TestTracesURL_AppendsPath(t *testing.T) {
	cases := []struct {
		endpoint string
		insecure bool
		want     string
	}{
		{"localhost:4318", true, "http://localhost:4318/v1/traces"},
		{"otel.example.com:4318", false, "https://otel.example.com:4318/v1/traces"},
		{"http://localhost:4318", true, "http://localhost:4318/v1/traces"},
		{"https://otel.example.com:4318/", false, "https://otel.example.com:4318/v1/traces"},
	}
	for _, tc := range cases {
		exp := NewOTLP(OTLPConfig{Endpoint: tc.endpoint, Insecure: tc.insecure})
		if got := exp.tracesURL(); got != tc.want {
			t.Errorf("tracesURL(%q insecure=%v) = %q, want %q",
				tc.endpoint, tc.insecure, got, tc.want)
		}
	}
}

// silence unused-import warnings if the file is later trimmed.
var _ = context.Background
var _ = strings.HasPrefix
