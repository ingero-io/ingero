package health

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/contract"
)

var emitterNow = time.Date(2026, 4, 16, 12, 0, 0, 0, time.UTC)

// recordingServer captures incoming OTLP requests for assertion.
type recordingServer struct {
	t      *testing.T
	server *httptest.Server
	mu     sync.Mutex
	bodies [][]byte
	status int
}

func newRecordingServer(t *testing.T, status int) *recordingServer {
	t.Helper()
	rs := &recordingServer{t: t, status: status}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/metrics", func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		rs.mu.Lock()
		rs.bodies = append(rs.bodies, body)
		st := rs.status
		rs.mu.Unlock()
		w.WriteHeader(st)
	})
	rs.server = httptest.NewServer(mux)
	t.Cleanup(rs.server.Close)
	return rs
}

func (rs *recordingServer) decodeLast(t *testing.T) otlpPayload {
	t.Helper()
	rs.mu.Lock()
	defer rs.mu.Unlock()
	if len(rs.bodies) == 0 {
		t.Fatal("no bodies recorded")
	}
	var p otlpPayload
	if err := json.Unmarshal(rs.bodies[len(rs.bodies)-1], &p); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return p
}

func (rs *recordingServer) setStatus(s int) {
	rs.mu.Lock()
	rs.status = s
	rs.mu.Unlock()
}

func (rs *recordingServer) count() int {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	return len(rs.bodies)
}

func baseEmitterCfg(endpoint string) EmitterConfig {
	c := DefaultEmitterConfig()
	c.Endpoint = endpoint
	c.ClusterID = "test-cluster"
	c.NodeID = "test-node"
	c.WorkloadType = "training"
	c.Insecure = true
	return c
}

func sampleScore() Score {
	return Score{
		Value:        0.82,
		Throughput:   0.9,
		Compute:      0.85,
		Memory:       0.75,
		CPU:          0.77,
		WorkloadType: "training",
		Timestamp:    emitterNow,
	}
}

func TestEmitterConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*EmitterConfig)
		wantErr bool
	}{
		{"valid", func(c *EmitterConfig) {}, false},
		{"empty_endpoint", func(c *EmitterConfig) { c.Endpoint = "" }, true},
		{"whitespace_endpoint", func(c *EmitterConfig) { c.Endpoint = "   " }, true},
		{"empty_cluster", func(c *EmitterConfig) { c.ClusterID = "" }, true},
		{"empty_node", func(c *EmitterConfig) { c.NodeID = "" }, true},
		{"invalid_utf8_node", func(c *EmitterConfig) { c.NodeID = string([]byte{0xff, 0xfe, 0xfd}) }, true},
		{"sub_second_interval", func(c *EmitterConfig) { c.PushInterval = 100 * time.Millisecond }, true},
		{"sub_100ms_timeout", func(c *EmitterConfig) { c.Timeout = 50 * time.Millisecond }, true},
		{"timeout_exceeds_interval", func(c *EmitterConfig) {
			c.PushInterval = 2 * time.Second
			c.Timeout = 5 * time.Second
		}, true},
		{"zero_threshold", func(c *EmitterConfig) { c.FailureThreshold = 0 }, true},
		{"tls_partial", func(c *EmitterConfig) { c.TLS.CACertPath = "/tmp/ca.pem" }, true},
		{"negative_rank", func(c *EmitterConfig) { c.NodeRank = -1 }, true},
		{"negative_world", func(c *EmitterConfig) { c.WorldSize = -1 }, true},
		{"rank_ge_world", func(c *EmitterConfig) { c.WorldSize = 4; c.NodeRank = 4 }, true},
		{"header_cr_key", func(c *EmitterConfig) { c.Headers = map[string]string{"X-Evil\r": "v"} }, true},
		{"header_lf_value", func(c *EmitterConfig) { c.Headers = map[string]string{"X": "v\nAuth: bad"} }, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := baseEmitterCfg("fleet:4318")
			tc.mutate(&c)
			err := c.Validate()
			if tc.wantErr && err == nil {
				t.Fatal("expected error")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected: %v", err)
			}
		})
	}
}

func TestNewEmitter_RejectsInvalidConfig(t *testing.T) {
	_, err := NewEmitter(EmitterConfig{}, nil)
	if err == nil {
		t.Fatal("expected error on zero config")
	}
}

func TestBuildURL(t *testing.T) {
	cases := []struct {
		endpoint string
		insecure bool
		want     string
	}{
		{"fleet:4318", true, "http://fleet:4318/v1/metrics"},
		{"fleet:4318", false, "https://fleet:4318/v1/metrics"},
		{"http://fleet:4318", false, "http://fleet:4318/v1/metrics"},
		{"https://fleet:8080/", true, "https://fleet:8080/v1/metrics"},
		// Already has the target path — not duplicated.
		{"https://fleet:8080/v1/metrics", false, "https://fleet:8080/v1/metrics"},
		{"fleet:4318/v1/metrics", true, "http://fleet:4318/v1/metrics"},
		// Trailing slashes stripped before appending.
		{"fleet:4318/", true, "http://fleet:4318/v1/metrics"},
		{"fleet:4318///", true, "http://fleet:4318/v1/metrics"},
		// Non-default path preserved, path appended.
		{"https://fleet:8080/api", false, "https://fleet:8080/api/v1/metrics"},
		// IPv6.
		{"[::1]:4318", true, "http://[::1]:4318/v1/metrics"},
	}
	for _, c := range cases {
		got, err := buildURL(c.endpoint, c.insecure, "")
		if err != nil {
			t.Errorf("buildURL(%q, %v) unexpected error: %v", c.endpoint, c.insecure, err)
			continue
		}
		if got != c.want {
			t.Errorf("buildURL(%q, %v) = %q, want %q", c.endpoint, c.insecure, got, c.want)
		}
	}
}

func TestBuildURL_ClusterIDQueryParam(t *testing.T) {
	got, err := buildURL("fleet:4318", false, "prod-cluster")
	if err != nil {
		t.Fatal(err)
	}
	want := "https://fleet:4318/v1/metrics?cluster_id=prod-cluster"
	if got != want {
		t.Errorf("buildURL with clusterID = %q, want %q", got, want)
	}
	// Empty cluster_id should not add query param.
	got2, err := buildURL("fleet:4318", false, "")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(got2, "cluster_id") {
		t.Errorf("empty clusterID should not add query param: %q", got2)
	}
}

func TestBuildURL_Errors(t *testing.T) {
	cases := []string{
		"",
		"   ",
		"ftp://fleet:4318", // unsupported scheme
		"://invalid",       // malformed
	}
	for _, c := range cases {
		if _, err := buildURL(c, false, ""); err == nil {
			t.Errorf("buildURL(%q) expected error, got nil", c)
		}
	}
}

// AC1-AC5: all 8 contract metrics emitted in a single push with correct
// resource + data-point attributes.
func TestPush_EmitsAllContractMetrics(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	e, err := NewEmitter(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("NewEmitter: %v", err)
	}
	if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", true); err != nil {
		t.Fatalf("Push: %v", err)
	}
	p := rs.decodeLast(t)

	if len(p.ResourceMetrics) != 1 {
		t.Fatalf("want 1 resourceMetrics block, got %d", len(p.ResourceMetrics))
	}
	rm := p.ResourceMetrics[0]

	// Resource attributes: node.id + cluster.id.
	assertAttr(t, rm.Resource.Attributes, contract.AttrNodeID, "test-node")
	assertAttr(t, rm.Resource.Attributes, contract.AttrClusterID, "test-cluster")

	if len(rm.ScopeMetrics) != 1 {
		t.Fatalf("want 1 scopeMetrics block, got %d", len(rm.ScopeMetrics))
	}
	metrics := rm.ScopeMetrics[0].Metrics

	// All 8 metric names present.
	wantNames := []string{
		contract.MetricHealthScore,
		contract.MetricThroughputRatio,
		contract.MetricComputeEfficiency,
		contract.MetricMemoryHeadroom,
		contract.MetricCPUAvailability,
		contract.MetricDegradationWarning,
		contract.MetricDetectionMode,
		contract.MetricFleetReachable,
	}
	names := make(map[string]bool, len(metrics))
	for _, m := range metrics {
		names[m.Name] = true
	}
	for _, want := range wantNames {
		if !names[want] {
			t.Errorf("missing metric %q", want)
		}
	}
}

// AC1: health_score has the correct value and node_state attribute.
func TestPush_HealthScoreShape(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	e, _ := NewEmitter(cfg, nil)
	sc := sampleScore()
	if err := e.Push(context.Background(), emitterNow, sc, StateIdle, "fleet-cached", false); err != nil {
		t.Fatalf("Push: %v", err)
	}
	p := rs.decodeLast(t)
	m := findMetric(t, p, contract.MetricHealthScore)
	if m.Gauge == nil || len(m.Gauge.DataPoints) != 1 {
		t.Fatalf("health_score gauge missing or wrong dp count")
	}
	dp := m.Gauge.DataPoints[0]
	if dp.AsDouble == nil || *dp.AsDouble != sc.Value {
		t.Fatalf("health_score AsDouble = %v, want %v", dp.AsDouble, sc.Value)
	}
	assertAttr(t, dp.Attributes, contract.AttrNodeState, "idle")
	assertAttr(t, dp.Attributes, contract.AttrWorkloadType, "training")
}

// Degradation warning on => int 1.
func TestPush_DegradationWarningInt(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", true)
	p := rs.decodeLast(t)
	m := findMetric(t, p, contract.MetricDegradationWarning)
	dp := m.Gauge.DataPoints[0]
	if dp.AsInt == nil || *dp.AsInt != 1 {
		t.Fatalf("degradation_warning AsInt = %v, want 1", dp.AsInt)
	}
}

// Detection mode emitted with the string "mode" attribute per contract.
func TestPush_DetectionModeAttribute(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "local-baseline", false)
	p := rs.decodeLast(t)
	m := findMetric(t, p, contract.MetricDetectionMode)
	dp := m.Gauge.DataPoints[0]
	assertAttr(t, dp.Attributes, contract.AttrDetectionMode, "local-baseline")
	if dp.AsInt == nil || *dp.AsInt != 1 {
		t.Fatalf("detection_mode AsInt = %v, want 1", dp.AsInt)
	}
}

// AC3: world_size and node_rank attach when WorldSize > 0.
func TestPush_OptionalWorldSizeAttrs(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	cfg.WorldSize = 8
	cfg.NodeRank = 3
	e, _ := NewEmitter(cfg, nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	p := rs.decodeLast(t)
	m := findMetric(t, p, contract.MetricHealthScore)
	dp := m.Gauge.DataPoints[0]
	assertAttrInt(t, dp.Attributes, contract.AttrWorldSize, 8)
	assertAttrInt(t, dp.Attributes, contract.AttrNodeRank, 3)
}

// AC7: consecutive failures flip FleetReachable to false at threshold,
// and a successful push RESETS the counter so subsequent failures must
// re-trip the threshold from zero (not from 1).
func TestPush_FailureCountersTripFleetReachable(t *testing.T) {
	rs := newRecordingServer(t, http.StatusInternalServerError)
	cfg := baseEmitterCfg(rs.server.URL)
	cfg.FailureThreshold = 3
	e, _ := NewEmitter(cfg, nil)

	ctx := context.Background()
	if !e.FleetReachable() {
		t.Fatal("initially should be reachable")
	}
	for i := 0; i < 2; i++ {
		_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false)
		if !e.FleetReachable() {
			t.Fatalf("reachable flipped early at failure %d", i+1)
		}
	}
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false)
	if e.FleetReachable() {
		t.Fatal("reachable should be false after 3 consecutive failures")
	}

	// A successful push resets the counter to zero.
	rs.setStatus(http.StatusOK)
	if err := e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false); err != nil {
		t.Fatalf("push after recovery: %v", err)
	}
	if !e.FleetReachable() {
		t.Fatal("reachable should reset after successful push")
	}

	// After recovery: 2 more failures must NOT trip reachable (counter
	// should be 2, not threshold).
	rs.setStatus(http.StatusInternalServerError)
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false)
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false)
	if !e.FleetReachable() {
		t.Fatal("counter did not fully reset — 2 post-recovery failures tripped reachable")
	}
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false)
	if e.FleetReachable() {
		t.Fatal("reachable should trip after 3rd failure")
	}
}

// 4xx/5xx responses must NOT be retried (would flood the server).
// Network-class errors ARE retried once before counting as a failure.
func TestPush_5xxNotRetried(t *testing.T) {
	rs := newRecordingServer(t, http.StatusInternalServerError)
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	if got := rs.count(); got != 1 {
		t.Fatalf("server saw %d requests, want 1 (no retry on 5xx)", got)
	}
}

func TestPush_NetworkErrorRetriedOnce(t *testing.T) {
	// First attempt: unroutable address -> network error.
	// Second attempt: also network error since address is permanently
	// unroutable. We verify the retry happens by checking push latency —
	// the jittered 200ms backoff between attempts means total push time
	// is at least ~160ms (200ms - 20% jitter). A single-attempt would
	// return in a few ms.
	e, _ := NewEmitter(baseEmitterCfg("http://127.0.0.1:1/"), nil)
	start := time.Now()
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected error")
	}
	if elapsed < 100*time.Millisecond {
		t.Fatalf("push returned too quickly (%s) — retry backoff likely skipped", elapsed)
	}
}

// AC7: a push failure does not panic and returns a wrapped error.
func TestPush_ErrorDoesNotCrash(t *testing.T) {
	e, _ := NewEmitter(baseEmitterCfg("http://127.0.0.1:1/"), nil) // port 1 = reserved
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	if err == nil {
		t.Fatal("expected error for unreachable endpoint")
	}
	if _, errors := e.Stats(); errors != 1 {
		t.Fatalf("errors count = %d, want 1", errors)
	}
}

// AC7 reachability-on-non-2xx: the fleet_reachable metric reflects the
// emitter's current belief.
func TestPush_ReachableMetricReflectsState(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	cfg.FailureThreshold = 2
	e, _ := NewEmitter(cfg, nil)
	ctx := context.Background()

	// First push — reachable = 1.
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false)
	m := findMetric(t, rs.decodeLast(t), contract.MetricFleetReachable)
	if m.Gauge.DataPoints[0].AsInt == nil || *m.Gauge.DataPoints[0].AsInt != 1 {
		t.Fatalf("reachable = %v, want 1", m.Gauge.DataPoints[0].AsInt)
	}

	// Now fail twice — reachable flips.
	rs.setStatus(http.StatusInternalServerError)
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false)
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false)
	if e.FleetReachable() {
		t.Fatal("should be unreachable")
	}

	// Next push still happens; the server records it but the payload
	// reflects fleet_reachable=0.
	rs.setStatus(http.StatusOK)
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet-cached", false)
	// After this successful push the counter resets; the emitter's
	// FleetReachable view at the moment of this push is still "false"
	// until the success is recorded, so the payload we just sent should
	// read 0. Verify.
	m = findMetric(t, rs.decodeLast(t), contract.MetricFleetReachable)
	if m.Gauge.DataPoints[0].AsInt == nil || *m.Gauge.DataPoints[0].AsInt != 0 {
		t.Fatalf("reachable during recovering push = %v, want 0", m.Gauge.DataPoints[0].AsInt)
	}
}

// Concurrency: many goroutines pushing simultaneously must not race.
func TestPush_Concurrent(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	var wg sync.WaitGroup
	var total atomic.Int32
	for g := 0; g < 10; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false); err == nil {
					total.Add(1)
				}
			}
		}()
	}
	wg.Wait()
	if total.Load() != 200 {
		t.Fatalf("successful pushes = %d, want 200", total.Load())
	}
	pushes, errors := e.Stats()
	if pushes != 200 || errors != 0 {
		t.Fatalf("Stats = (%d, %d), want (200, 0)", pushes, errors)
	}
	if rs.count() != 200 {
		t.Fatalf("server recorded %d bodies, want 200", rs.count())
	}
}

// Context cancellation propagates — Push returns quickly.
func TestPush_RespectsContextCancel(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false)
	if err == nil {
		t.Fatal("expected error for cancelled ctx")
	}
}

// Piggyback headers on the push response flow into the ThresholdCache.
func TestPush_PiggybackHeadersFlowToCache(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(contract.HeaderThreshold, "0.83")
		w.Header().Set(contract.HeaderQuorumMet, "true")
		w.WriteHeader(http.StatusOK)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	cfg := baseEmitterCfg(srv.URL)
	cfg.ThresholdCache = cache
	e, _ := NewEmitter(cfg, nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)

	snap, ok := cache.Get()
	if !ok {
		t.Fatal("cache empty after piggyback")
	}
	if snap.Value != 0.83 || !snap.QuorumMet {
		t.Fatalf("cache = %+v, want 0.83/true", snap)
	}
	if !cache.PiggybackAvailable() {
		t.Fatal("piggyback should be available")
	}
}

// Piggyback parsing happens on non-2xx responses too — Fleet middleware
// may attach headers even on error responses.
func TestPush_PiggybackHeadersOnErrorResponse(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(contract.HeaderThreshold, "0.77")
		w.Header().Set(contract.HeaderQuorumMet, "true")
		w.WriteHeader(http.StatusServiceUnavailable)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	cfg := baseEmitterCfg(srv.URL)
	cfg.ThresholdCache = cache
	e, _ := NewEmitter(cfg, nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)

	snap, ok := cache.Get()
	if !ok || snap.Value != 0.77 {
		t.Fatalf("cache should carry headers from error response: snap=%+v ok=%v", snap, ok)
	}
}

// No headers on response => piggyback marked unavailable.
func TestPush_NoHeadersFlagsPiggybackUnavailable(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cache := NewThresholdCache()
	cfg := baseEmitterCfg(rs.server.URL)
	cfg.ThresholdCache = cache
	e, _ := NewEmitter(cfg, nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	if cache.PiggybackAvailable() {
		t.Fatal("piggyback should be unavailable when headers absent")
	}
}

// Headers from EmitterConfig are attached.
func TestPush_CustomHeaders(t *testing.T) {
	var captured http.Header
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/metrics", func(w http.ResponseWriter, r *http.Request) {
		captured = r.Header.Clone()
		w.WriteHeader(http.StatusOK)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cfg := baseEmitterCfg(srv.URL)
	cfg.Headers = map[string]string{"Authorization": "Bearer secret"}
	e, _ := NewEmitter(cfg, nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	if captured.Get("Authorization") != "Bearer secret" {
		t.Fatalf("header missing: %v", captured)
	}
	if captured.Get("Content-Type") != "application/json" {
		t.Fatalf("content-type wrong: %v", captured.Get("Content-Type"))
	}
}

// ---------- helpers ----------

func findMetric(t *testing.T, p otlpPayload, name string) otlpMetric {
	t.Helper()
	for _, rm := range p.ResourceMetrics {
		for _, sm := range rm.ScopeMetrics {
			for _, m := range sm.Metrics {
				if m.Name == name {
					return m
				}
			}
		}
	}
	t.Fatalf("metric %q not found in payload", name)
	return otlpMetric{}
}

func assertAttr(t *testing.T, attrs []otlpKV, key, want string) {
	t.Helper()
	for _, a := range attrs {
		if a.Key == key {
			if a.Value.StringValue == nil || *a.Value.StringValue != want {
				t.Fatalf("attr %q = %v, want %q", key, a.Value.StringValue, want)
			}
			return
		}
	}
	t.Fatalf("attr %q missing", key)
}

func assertAttrInt(t *testing.T, attrs []otlpKV, key string, want int64) {
	t.Helper()
	for _, a := range attrs {
		if a.Key == key {
			if a.Value.IntValue == nil || *a.Value.IntValue != want {
				t.Fatalf("attr %q (int) = %v, want %d", key, a.Value.IntValue, want)
			}
			return
		}
	}
	t.Fatalf("attr %q missing", key)
}

// 429 with Retry-After in seconds surfaces as a typed RetryAfterError
// so the loop can delay the next tick by the server-requested amount.
func TestPush_429RetryAfterSeconds(t *testing.T) {
	rs := newRetryAfterServer(t, http.StatusTooManyRequests, "7")
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	if err == nil {
		t.Fatal("expected RetryAfterError, got nil")
	}
	ra := AsRetryAfter(err)
	if ra == nil {
		t.Fatalf("expected RetryAfterError, got %T: %v", err, err)
	}
	if ra.StatusCode != http.StatusTooManyRequests {
		t.Errorf("StatusCode=%d, want 429", ra.StatusCode)
	}
	if ra.Delay != 7*time.Second {
		t.Errorf("Delay=%s, want 7s", ra.Delay)
	}
}

// 503 with Retry-After as HTTP-date. Covers the other RFC 7231 form.
func TestPush_503RetryAfterHTTPDate(t *testing.T) {
	future := time.Now().Add(20 * time.Second).UTC().Format(http.TimeFormat)
	rs := newRetryAfterServer(t, http.StatusServiceUnavailable, future)
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	ra := AsRetryAfter(err)
	if ra == nil {
		t.Fatalf("expected RetryAfterError, got %v", err)
	}
	if ra.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("StatusCode=%d, want 503", ra.StatusCode)
	}
	if ra.Delay < 10*time.Second || ra.Delay > 25*time.Second {
		t.Errorf("Delay=%s out of expected range ~20s", ra.Delay)
	}
}

// 429 WITHOUT Retry-After falls through to the plain status-error path.
func TestPush_429NoRetryAfterIsPlainError(t *testing.T) {
	rs := newRetryAfterServer(t, http.StatusTooManyRequests, "")
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	if err == nil {
		t.Fatal("expected error")
	}
	if AsRetryAfter(err) != nil {
		t.Errorf("plain 429 (no header) should NOT produce RetryAfterError; got %v", err)
	}
}

// 400 with Retry-After is ignored; the helper only consults the header
// for 429 and 503.
func TestPush_400RetryAfterIsPlainError(t *testing.T) {
	rs := newRetryAfterServer(t, http.StatusBadRequest, "30")
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	if AsRetryAfter(err) != nil {
		t.Errorf("400 with Retry-After should NOT produce RetryAfterError; got %v", err)
	}
}

func newRetryAfterServer(t *testing.T, status int, retryAfter string) *recordingServer {
	t.Helper()
	rs := &recordingServer{t: t, status: status}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/metrics", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		rs.mu.Lock()
		rs.bodies = append(rs.bodies, body)
		rs.mu.Unlock()
		if retryAfter != "" {
			w.Header().Set("Retry-After", retryAfter)
		}
		w.WriteHeader(status)
	})
	rs.server = httptest.NewServer(mux)
	t.Cleanup(rs.server.Close)
	return rs
}

// A server that hijacks the connection and slams it shut with
// SO_LINGER=0 before writing a response. Go surfaces this as either
// "connection reset by peer" or "EOF"; both count as network-class
// errors the emitter's retry loop must handle.
func TestPush_TCPResetIsRetriedAsNetworkError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("response writer is not a Hijacker")
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			t.Fatalf("hijack: %v", err)
		}
		if tc, ok := conn.(*net.TCPConn); ok {
			// SO_LINGER=0 turns Close() into an abortive RST instead of
			// a graceful FIN-then-wait.
			_ = tc.SetLinger(0)
		}
		_ = conn.Close()
	}))
	defer srv.Close()

	e, _ := NewEmitter(baseEmitterCfg(srv.URL), nil)
	start := time.Now()
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected error from RST; got nil")
	}
	if AsRetryAfter(err) != nil {
		t.Fatalf("RST must NOT look like a Retry-After error; got %v", err)
	}
	if elapsed < 100*time.Millisecond {
		t.Errorf("push returned in %s — retry backoff likely skipped", elapsed)
	}
}

// Blackhole dial: point the emitter at a no-longer-listening port and
// assert the push fails inside cfg.Timeout + slack. Guards against a
// future change that forgets to set a dialer timeout or relies on
// kernel default SYN retries (~63 s on Linux).
func TestPush_BlackholeDialRespectsTimeout(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr := ln.Addr().String()
	ln.Close()

	cfg := baseEmitterCfg("http://" + addr)
	cfg.PushInterval = 2 * time.Second
	cfg.Timeout = 500 * time.Millisecond
	e, _ := NewEmitter(cfg, nil)

	start := time.Now()
	perr := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false)
	elapsed := time.Since(start)

	if perr == nil {
		t.Fatal("expected error")
	}
	// Budget: 2x Timeout (first attempt) + 200ms jitter + 2x Timeout
	// (retry) + 500ms slack.
	if elapsed > 5*time.Second {
		t.Errorf("push elapsed %s > 5s budget; dialer timeout drift?", elapsed)
	}
}
