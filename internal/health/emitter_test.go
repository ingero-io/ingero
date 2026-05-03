package health

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
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
	if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", true, nil); err != nil {
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
	if err := e.Push(context.Background(), emitterNow, sc, StateIdle, "fleet-cached", false, nil); err != nil {
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
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", true, nil)
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
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "local-baseline", false, nil)
	p := rs.decodeLast(t)
	m := findMetric(t, p, contract.MetricDetectionMode)
	dp := m.Gauge.DataPoints[0]
	assertAttr(t, dp.Attributes, contract.AttrDetectionMode, "local-baseline")
	if dp.AsInt == nil || *dp.AsInt != 1 {
		t.Fatalf("detection_mode AsInt = %v, want 1", dp.AsInt)
	}
}

// AC3: world_size and node_rank attach as RESOURCE attributes (stable
// per-agent identity, not per-data-point) when WorldSize > 0. v0.11
// moves these from data-point attrs to resource attrs to follow OTEL
// convention for identity-shaped attributes; consumers that read
// resource scope inherit them on every metric automatically.
func TestPush_OptionalWorldSizeAttrs(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	cfg.WorldSize = 8
	cfg.NodeRank = 3
	e, _ := NewEmitter(cfg, nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
	p := rs.decodeLast(t)
	if len(p.ResourceMetrics) != 1 {
		t.Fatalf("want 1 resourceMetrics block, got %d", len(p.ResourceMetrics))
	}
	rm := p.ResourceMetrics[0]
	assertAttrInt(t, rm.Resource.Attributes, contract.AttrWorldSize, 8)
	assertAttrInt(t, rm.Resource.Attributes, contract.AttrNodeRank, 3)

	// Verify they're NOT on data-points (the old, deprecated location).
	m := findMetric(t, p, contract.MetricHealthScore)
	dp := m.Gauge.DataPoints[0]
	for _, kv := range dp.Attributes {
		if kv.Key == contract.AttrWorldSize || kv.Key == contract.AttrNodeRank {
			t.Errorf("attr %q must not appear on data-point in v0.11 (resource-only)", kv.Key)
		}
	}
}

// world_size/node_rank must be ABSENT from the resource block when
// WorldSize is 0 (the default for non-distributed deployments).
func TestPush_NoWorldSizeAttrsWhenZero(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	// cfg.WorldSize stays 0
	e, _ := NewEmitter(cfg, nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
	p := rs.decodeLast(t)
	rm := p.ResourceMetrics[0]
	for _, kv := range rm.Resource.Attributes {
		if kv.Key == contract.AttrWorldSize || kv.Key == contract.AttrNodeRank {
			t.Errorf("attr %q must not appear when WorldSize=0", kv.Key)
		}
	}
}

// EmitStragglerEvent must also carry world_size/node_rank on the
// resource block when configured (per-agent identity is consistent
// across the regular push and the straggler-edge push).
func TestEmitStragglerEvent_ResourceCarriesRankWorldSize(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	cfg.WorldSize = 4
	cfg.NodeRank = 1
	e, _ := NewEmitter(cfg, nil)
	ev := StragglerEvent{
		NodeID:         "test-node",
		ClusterID:      "test-cluster",
		Score:          0.42,
		Threshold:      0.6,
		DetectionMode:  "fleet",
		DominantSignal: "throughput",
		Timestamp:      emitterNow,
	}
	_ = e.EmitStragglerEvent(context.Background(), ev, true)
	p := rs.decodeLast(t)
	if len(p.ResourceMetrics) != 1 {
		t.Fatalf("want 1 resourceMetrics block, got %d", len(p.ResourceMetrics))
	}
	rm := p.ResourceMetrics[0]
	assertAttrInt(t, rm.Resource.Attributes, contract.AttrWorldSize, 4)
	assertAttrInt(t, rm.Resource.Attributes, contract.AttrNodeRank, 1)
}

// v0.11 cost-of-problem support gauges: ingero.node.info (gpu_model +
// gpu_count attrs) appears when both fields are configured;
// ingero.node.world_size always emits with the configured value.
func TestPush_CostGaugesEmittedWhenConfigured(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	cfg.GPUModel = "NVIDIA GH200 480GB"
	cfg.GPUCount = 1
	cfg.WorldSize = 8
	cfg.NodeRank = 3
	e, _ := NewEmitter(cfg, nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
	p := rs.decodeLast(t)

	info := findMetric(t, p, contract.MetricNodeInfo)
	if info.Gauge == nil || len(info.Gauge.DataPoints) != 1 {
		t.Fatalf("MetricNodeInfo missing or has wrong shape")
	}
	if info.Gauge.DataPoints[0].AsInt == nil || *info.Gauge.DataPoints[0].AsInt != 1 {
		t.Errorf("MetricNodeInfo value should be 1; got %+v", info.Gauge.DataPoints[0])
	}
	assertAttr(t, info.Gauge.DataPoints[0].Attributes, contract.AttrGPUModel, "NVIDIA GH200 480GB")
	assertAttrInt(t, info.Gauge.DataPoints[0].Attributes, contract.AttrGPUCount, 1)

	ws := findMetric(t, p, contract.MetricNodeWorldSize)
	if ws.Gauge == nil || len(ws.Gauge.DataPoints) != 1 {
		t.Fatalf("MetricNodeWorldSize missing or has wrong shape")
	}
	if ws.Gauge.DataPoints[0].AsInt == nil || *ws.Gauge.DataPoints[0].AsInt != 8 {
		t.Errorf("MetricNodeWorldSize value should be 8; got %+v", ws.Gauge.DataPoints[0])
	}
}

// ingero.node.info is suppressed when the operator does not pass
// gpu_model / gpu_count (e.g. no nvidia-smi in the environment).
// ingero.node.world_size still emits with value=0 to make the absence
// of distributed-training affirmative on the wire.
func TestPush_CostGauges_NodeInfoOmittedWithoutGPU(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	// GPUModel + GPUCount left zero
	e, _ := NewEmitter(cfg, nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
	p := rs.decodeLast(t)
	if len(p.ResourceMetrics) == 0 {
		t.Fatal("no resource metrics")
	}
	for _, m := range p.ResourceMetrics[0].ScopeMetrics[0].Metrics {
		if m.Name == contract.MetricNodeInfo {
			t.Errorf("MetricNodeInfo must be omitted when GPUModel/GPUCount are unset")
		}
	}
	// world_size always emits, even at zero.
	ws := findMetric(t, p, contract.MetricNodeWorldSize)
	if ws.Gauge == nil || len(ws.Gauge.DataPoints) != 1 {
		t.Fatalf("MetricNodeWorldSize missing")
	}
	if ws.Gauge.DataPoints[0].AsInt == nil || *ws.Gauge.DataPoints[0].AsInt != 0 {
		t.Errorf("MetricNodeWorldSize value should be 0 when not distributed; got %+v", ws.Gauge.DataPoints[0])
	}
}

// EmitStragglerEvent attaches the per-event UUID as the
// `ingero.event.id` data-point attribute so consumers correlate the
// OTLP push with the parallel UDS message.
func TestEmitStragglerEvent_AttachesEventID(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	ev := StragglerEvent{
		NodeID:    "test-node",
		ClusterID: "test-cluster",
		Score:     0.42,
		Threshold: 0.6,
		Timestamp: emitterNow,
		EventID:   "00000000-0000-0000-0000-000000000abc",
	}
	_ = e.EmitStragglerEvent(context.Background(), ev, true)
	p := rs.decodeLast(t)
	m := findMetric(t, p, contract.MetricStragglerEvent)
	dp := m.Gauge.DataPoints[0]
	assertAttr(t, dp.Attributes, contract.AttrEventID, ev.EventID)
}

// v0.13 Slice A: cgroup_path_hash is emitted on the health_score data
// point, resolved once at NewEmitter via the package-level resolver.
// Fleet groups MAD thresholds by this attribute; absence folds into the
// legacy cluster-wide bucket. The resolver is stubbed here so the test
// does not depend on the real /proc/self/cgroup contents (which differ
// between Linux CI, WSL, macOS, and Windows).
func TestPush_HealthScoreCarriesCgroupPathHash(t *testing.T) {
	const stubHash = "abc123def4567890"
	prev := cgroupPathHashResolver
	cgroupPathHashResolver = func() (string, error) { return stubHash, nil }
	t.Cleanup(func() { cgroupPathHashResolver = prev })

	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	e, err := NewEmitter(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("NewEmitter: %v", err)
	}
	if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil); err != nil {
		t.Fatalf("Push: %v", err)
	}
	p := rs.decodeLast(t)
	m := findMetric(t, p, contract.MetricHealthScore)
	if m.Gauge == nil || len(m.Gauge.DataPoints) != 1 {
		t.Fatalf("health_score gauge missing or wrong dp count")
	}
	dp := m.Gauge.DataPoints[0]
	assertAttr(t, dp.Attributes, contract.AttrCgroupPathHash, stubHash)

	// Hash is fixed-width 16 hex chars per the contract definition on
	// contract.AttrCgroupPathHash. Assert the stub honored that shape so
	// future drift in the production resolver is caught by the same test.
	for _, kv := range dp.Attributes {
		if kv.Key == contract.AttrCgroupPathHash {
			if kv.Value.StringValue == nil {
				t.Fatalf("cgroup_path_hash value is nil")
			}
			if got := len(*kv.Value.StringValue); got != 16 {
				t.Fatalf("cgroup_path_hash length = %d, want 16", got)
			}
		}
	}
}

// Resolver failure path: emitter must not panic, must emit empty string
// for the attribute, and must continue pushing normally. Mirrors the
// "no /proc available" case (macOS dev, hostile sandbox).
func TestPush_HealthScoreCgroupPathHashEmptyOnResolverError(t *testing.T) {
	prev := cgroupPathHashResolver
	cgroupPathHashResolver = func() (string, error) {
		return "", errors.New("simulated /proc unavailable")
	}
	t.Cleanup(func() { cgroupPathHashResolver = prev })

	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	e, err := NewEmitter(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("NewEmitter: %v", err)
	}
	if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil); err != nil {
		t.Fatalf("Push: %v", err)
	}
	p := rs.decodeLast(t)
	m := findMetric(t, p, contract.MetricHealthScore)
	dp := m.Gauge.DataPoints[0]
	assertAttr(t, dp.Attributes, contract.AttrCgroupPathHash, "")
}

// EventID attribute is omitted when StragglerEvent.EventID is empty.
func TestEmitStragglerEvent_OmitsEmptyEventID(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	ev := StragglerEvent{
		NodeID:    "test-node",
		ClusterID: "test-cluster",
		Timestamp: emitterNow,
	}
	_ = e.EmitStragglerEvent(context.Background(), ev, true)
	p := rs.decodeLast(t)
	m := findMetric(t, p, contract.MetricStragglerEvent)
	dp := m.Gauge.DataPoints[0]
	for _, kv := range dp.Attributes {
		if kv.Key == contract.AttrEventID {
			t.Errorf("AttrEventID must be absent when EventID is empty")
		}
	}
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
		_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false, nil)
		if !e.FleetReachable() {
			t.Fatalf("reachable flipped early at failure %d", i+1)
		}
	}
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false, nil)
	if e.FleetReachable() {
		t.Fatal("reachable should be false after 3 consecutive failures")
	}

	// A successful push resets the counter to zero.
	rs.setStatus(http.StatusOK)
	if err := e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false, nil); err != nil {
		t.Fatalf("push after recovery: %v", err)
	}
	if !e.FleetReachable() {
		t.Fatal("reachable should reset after successful push")
	}

	// After recovery: 2 more failures must NOT trip reachable (counter
	// should be 2, not threshold).
	rs.setStatus(http.StatusInternalServerError)
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false, nil)
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false, nil)
	if !e.FleetReachable() {
		t.Fatal("counter did not fully reset — 2 post-recovery failures tripped reachable")
	}
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false, nil)
	if e.FleetReachable() {
		t.Fatal("reachable should trip after 3rd failure")
	}
}

// 4xx/5xx responses must NOT be retried (would flood the server).
// Network-class errors ARE retried once before counting as a failure.
func TestPush_5xxNotRetried(t *testing.T) {
	rs := newRecordingServer(t, http.StatusInternalServerError)
	e, _ := NewEmitter(baseEmitterCfg(rs.server.URL), nil)
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false, nil)
	m := findMetric(t, rs.decodeLast(t), contract.MetricFleetReachable)
	if m.Gauge.DataPoints[0].AsInt == nil || *m.Gauge.DataPoints[0].AsInt != 1 {
		t.Fatalf("reachable = %v, want 1", m.Gauge.DataPoints[0].AsInt)
	}

	// Now fail twice — reachable flips.
	rs.setStatus(http.StatusInternalServerError)
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false, nil)
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false, nil)
	if e.FleetReachable() {
		t.Fatal("should be unreachable")
	}

	// Next push still happens; the server records it but the payload
	// reflects fleet_reachable=0.
	rs.setStatus(http.StatusOK)
	_ = e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet-cached", false, nil)
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
				if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil); err == nil {
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
	err := e.Push(ctx, emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)

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
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)

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
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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
	_ = e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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
	err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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
	perr := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil)
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

// ============================================================================
// Multi-replica failover tests (added 2026-04-26)
// ============================================================================
//
// These tests cover Fleet pod selection / failover behavior in multi-replica
// deployments. They use a test-only http.RoundTripper injected into the
// emitter's http.Client to simulate multi-backend routing without depending
// on a custom net.Resolver. See `ingero-fleet/docs/ARCHITECTURE.md`
// "Multi-replica behavior" section for the public design context.

// failoverRT is a test RoundTripper: fails the first request (simulating a
// dead Fleet pod) and forwards every subsequent request to the fallback URL.
type failoverRT struct {
	fallbackURL string
	callCount   int32
}

func (r *failoverRT) RoundTrip(req *http.Request) (*http.Response, error) {
	n := atomic.AddInt32(&r.callCount, 1)
	if n == 1 {
		return nil, &net.OpError{Op: "dial", Err: errors.New("connection refused (simulated)")}
	}
	u, err := url.Parse(r.fallbackURL)
	if err != nil {
		return nil, err
	}
	newReq := req.Clone(req.Context())
	newReq.URL.Scheme = u.Scheme
	newReq.URL.Host = u.Host
	return http.DefaultTransport.RoundTrip(newReq)
}

// randomBackendRT is a test RoundTripper: picks one of two backends at
// random for each request and tracks per-backend hit counts.
type randomBackendRT struct {
	serverA, serverB *httptest.Server
	mu               sync.Mutex
	rng              *rand.Rand
	countA, countB   int32
}

func (r *randomBackendRT) RoundTrip(req *http.Request) (*http.Response, error) {
	r.mu.Lock()
	pick := r.rng.Intn(2)
	r.mu.Unlock()

	var target string
	if pick == 0 {
		target = r.serverA.URL
		atomic.AddInt32(&r.countA, 1)
	} else {
		target = r.serverB.URL
		atomic.AddInt32(&r.countB, 1)
	}
	u, err := url.Parse(target)
	if err != nil {
		return nil, err
	}
	newReq := req.Clone(req.Context())
	newReq.URL.Scheme = u.Scheme
	newReq.URL.Host = u.Host
	return http.DefaultTransport.RoundTrip(newReq)
}

// TestPush_RedialAfterFailure_PicksFreshBackend verifies that on a network
// error to one backend, the emitter's retry triggers a fresh RoundTrip
// (which a real http.Transport would back with a fresh DNS lookup), and
// the second attempt succeeds against the fallback backend.
func TestPush_RedialAfterFailure_PicksFreshBackend(t *testing.T) {
	var fallbackHits int32
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackHits, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer fallback.Close()

	rt := &failoverRT{fallbackURL: fallback.URL}

	cfg := baseEmitterCfg(fallback.URL)
	cfg.Timeout = 500 * time.Millisecond
	e, err := NewEmitter(cfg, nil)
	if err != nil {
		t.Fatalf("NewEmitter: %v", err)
	}
	e.(*httpEmitter).client.Transport = rt

	if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil); err != nil {
		t.Fatalf("push should have succeeded after retry: %v", err)
	}
	if calls := atomic.LoadInt32(&rt.callCount); calls != 2 {
		t.Errorf("expected 2 RoundTrip calls (initial fail + retry succeed), got %d", calls)
	}
	if hits := atomic.LoadInt32(&fallbackHits); hits != 1 {
		t.Errorf("expected fallback to receive 1 successful request, got %d", hits)
	}
}

// TestPush_ManyAgentInstances_DistributeAcrossBackends verifies that when
// N emitter instances each push once via a randomly-routing RoundTripper,
// hits distribute roughly uniformly across backends. Models the "50 agents,
// 2 Fleet pods, ~25/25 split" property.
//
// Tolerance: with N=50 and p=0.5, stddev is ~3.5; we accept any split
// within [10, 40] (well outside any plausible variance).
func TestPush_ManyAgentInstances_DistributeAcrossBackends(t *testing.T) {
	serverA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer serverA.Close()
	serverB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer serverB.Close()

	rt := &randomBackendRT{
		serverA: serverA,
		serverB: serverB,
		rng:     rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	cfg := baseEmitterCfg(serverA.URL)

	const N = 50
	for i := 0; i < N; i++ {
		e, err := NewEmitter(cfg, nil)
		if err != nil {
			t.Fatalf("NewEmitter %d: %v", i, err)
		}
		e.(*httpEmitter).client.Transport = rt
		if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil); err != nil {
			t.Errorf("push %d: %v", i, err)
		}
	}

	a := atomic.LoadInt32(&rt.countA)
	b := atomic.LoadInt32(&rt.countB)
	if a+b != N {
		t.Errorf("expected %d total pushes across backends, got A=%d B=%d (sum %d)", N, a, b, a+b)
	}
	if a < 10 || a > 40 {
		t.Errorf("distribution skewed: A=%d B=%d (expected ~25/25, accepting [10,40] each)", a, b)
	}
}

// TestEmitterTransport_HasStickyConnectionConfig verifies that the emitter's
// HTTP transport is configured for connection-pool stickiness: one idle
// connection per host with a 30s idle timeout. With 10s push intervals these
// settings keep the agent bound to one Fleet pod in steady state, which is
// the design intent for multi-replica HA without an L7 LB. Connection-level
// stickiness itself can't be exercised through a custom RoundTripper (it
// lives in *http.Transport's pool), so this is a configuration-validation
// test.
func TestEmitterTransport_HasStickyConnectionConfig(t *testing.T) {
	cfg := baseEmitterCfg("http://127.0.0.1:9999")
	e, err := NewEmitter(cfg, nil)
	if err != nil {
		t.Fatalf("NewEmitter: %v", err)
	}
	transport, ok := e.(*httpEmitter).client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("expected *http.Transport, got %T", e.(*httpEmitter).client.Transport)
	}
	if transport.MaxIdleConnsPerHost != 1 {
		t.Errorf("MaxIdleConnsPerHost = %d, want 1 (sticky pool behavior)", transport.MaxIdleConnsPerHost)
	}
	if transport.IdleConnTimeout != 30*time.Second {
		t.Errorf("IdleConnTimeout = %v, want 30s", transport.IdleConnTimeout)
	}
}

// findSumMetric is the Sum-shaped sibling of findMetric. Per-cgroup
// metrics are emitted as OTLP Sum (cumulative monotonic), not Gauge.
func findSumMetric(t *testing.T, p otlpPayload, name string) otlpMetric {
	t.Helper()
	for _, rm := range p.ResourceMetrics {
		for _, sm := range rm.ScopeMetrics {
			for _, m := range sm.Metrics {
				if m.Name == name && m.Sum != nil {
					return m
				}
			}
		}
	}
	t.Fatalf("Sum metric %q not found in payload", name)
	return otlpMetric{}
}

func samplePerCGroup(hashA, hashB string) []PerCGroupStats {
	return []PerCGroupStats{
		{
			CgroupPathHash:    hashA,
			KernelLaunchCount: 100,
			CPUStallNanos:     5_000_000,
			MemcpyBytesByDir: map[string]int64{
				contract.MemcpyDirectionH2D: 8192,
				contract.MemcpyDirectionD2H: 4096,
			},
		},
		{
			CgroupPathHash:    hashB,
			KernelLaunchCount: 50,
		},
	}
}

// v0.13 Slice B: when perCGroup is non-empty, the emitter emits all
// three per-cgroup metric families as OTel Sum data points labeled by
// cgroup_path_hash. The memcpy metric also carries the direction
// attribute.
func TestPush_PerCGroupMetricsEmitted(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	e, err := NewEmitter(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("NewEmitter: %v", err)
	}
	stats := samplePerCGroup("hash-a", "hash-b")
	if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, stats); err != nil {
		t.Fatalf("Push: %v", err)
	}
	p := rs.decodeLast(t)

	launch := findSumMetric(t, p, contract.MetricCUDAKernelLaunchTotal)
	if launch.Sum == nil || !launch.Sum.IsMonotonic || launch.Sum.AggregationTemporality != 2 {
		t.Errorf("kernel launch metric shape wrong: %+v", launch.Sum)
	}
	gotHashes := make(map[string]int64)
	for _, dp := range launch.Sum.DataPoints {
		for _, kv := range dp.Attributes {
			if kv.Key == contract.AttrCgroupPathHash && dp.AsInt != nil {
				gotHashes[*kv.Value.StringValue] = *dp.AsInt
			}
		}
	}
	if gotHashes["hash-a"] != 100 || gotHashes["hash-b"] != 50 {
		t.Errorf("kernel launch series = %+v, want hash-a:100 hash-b:50", gotHashes)
	}

	stall := findSumMetric(t, p, contract.MetricCPUStallNanosTotal)
	if got := dpAsIntForHash(t, stall, "hash-a"); got != 5_000_000 {
		t.Errorf("cpu_stall hash-a = %d, want 5_000_000", got)
	}

	memcpy := findSumMetric(t, p, contract.MetricCUDAMemcpyBytesTotal)
	memDir := make(map[string]int64)
	for _, dp := range memcpy.Sum.DataPoints {
		var hash, dir string
		for _, kv := range dp.Attributes {
			if kv.Key == contract.AttrCgroupPathHash {
				hash = *kv.Value.StringValue
			}
			if kv.Key == contract.AttrMemcpyDirection {
				dir = *kv.Value.StringValue
			}
		}
		if dp.AsInt != nil {
			memDir[hash+"|"+dir] = *dp.AsInt
		}
	}
	if memDir["hash-a|h2d"] != 8192 || memDir["hash-a|d2h"] != 4096 {
		t.Errorf("memcpy series = %+v, want hash-a|h2d:8192 hash-a|d2h:4096", memDir)
	}
}

// Cumulative semantics: the second push adds its window delta to the
// running total. Pushing 10 then 5 should emit 15 on the second tick.
func TestPush_PerCGroupCountersCumulative(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	e, err := NewEmitter(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("NewEmitter: %v", err)
	}

	first := []PerCGroupStats{{CgroupPathHash: "x", KernelLaunchCount: 10}}
	second := []PerCGroupStats{{CgroupPathHash: "x", KernelLaunchCount: 5}}
	if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, first); err != nil {
		t.Fatalf("Push 1: %v", err)
	}
	if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, second); err != nil {
		t.Fatalf("Push 2: %v", err)
	}
	p := rs.decodeLast(t)
	launch := findSumMetric(t, p, contract.MetricCUDAKernelLaunchTotal)
	if got := dpAsIntForHash(t, launch, "x"); got != 15 {
		t.Errorf("cumulative kernel_launch = %d, want 15 (10 + 5)", got)
	}
}

// A nil / empty perCGroup slice must not produce any per-cgroup metric
// families. This protects existing pre-Slice-B test invariants: the
// payload should look identical to v0.12 when no cgroup data is fed.
func TestPush_PerCGroupNilProducesNoMetrics(t *testing.T) {
	rs := newRecordingServer(t, http.StatusOK)
	cfg := baseEmitterCfg(rs.server.URL)
	e, err := NewEmitter(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Push(context.Background(), emitterNow, sampleScore(), StateActive, "fleet", false, nil); err != nil {
		t.Fatalf("Push: %v", err)
	}
	p := rs.decodeLast(t)
	for _, rm := range p.ResourceMetrics {
		for _, sm := range rm.ScopeMetrics {
			for _, m := range sm.Metrics {
				switch m.Name {
				case contract.MetricCUDAKernelLaunchTotal,
					contract.MetricCPUStallNanosTotal,
					contract.MetricCUDAMemcpyBytesTotal:
					t.Errorf("nil perCGroup leaked metric %q into payload", m.Name)
				}
			}
		}
	}
}

// dpAsIntForHash extracts the AsInt value for a Sum data point whose
// cgroup_path_hash matches. Returns -1 when missing so the caller can
// distinguish "found 0" from "absent" in assertions.
func dpAsIntForHash(t *testing.T, m otlpMetric, hash string) int64 {
	t.Helper()
	if m.Sum == nil {
		return -1
	}
	for _, dp := range m.Sum.DataPoints {
		for _, kv := range dp.Attributes {
			if kv.Key == contract.AttrCgroupPathHash && kv.Value.StringValue != nil && *kv.Value.StringValue == hash {
				if dp.AsInt == nil {
					return -1
				}
				return *dp.AsInt
			}
		}
	}
	return -1
}
