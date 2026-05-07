package health

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/ingero-io/ingero/internal/auth"
	"github.com/ingero-io/ingero/internal/cgroup"
	"github.com/ingero-io/ingero/pkg/contract"
)

// cgroupPathHashResolver is the function used to resolve the agent's own
// cgroup path hash at emitter startup. Overridable for tests to avoid
// depending on the real /proc/self/cgroup contents. Returns the SHA256
// hash of the cgroup path truncated to 16 hex chars, per the contract
// described on contract.AttrCgroupPathHash.
var cgroupPathHashResolver = defaultCgroupPathHash

// defaultCgroupPathHash reads /proc/self/cgroup, resolves the cgroup path,
// and returns its SHA256 truncated to 16 hex chars. Returns an error if
// /proc is not available or the file cannot be read (e.g. on macOS/Windows
// during local development, or inside an environment that hides /proc).
func defaultCgroupPathHash() (string, error) {
	path, err := cgroup.ReadCGroupPath(uint32(os.Getpid()))
	if err != nil {
		return "", err
	}
	return hashCGroupPath(path), nil
}

// ResolveCgroupPathHash exposes the agent's own cgroup_path_hash for
// callers that need to attribute non-emitter telemetry (notably the
// detection-event tracer) to the same cgroup the metrics emitter uses.
// Returns "" on any resolution failure — the same fallback behavior as
// the emitter.
func ResolveCgroupPathHash() string {
	h, err := cgroupPathHashResolver()
	if err != nil {
		return ""
	}
	return h
}

// minPushInterval and minTimeout are conservative lower bounds enforced by
// EmitterConfig.Validate. They prevent pathological configurations that
// would flood Fleet or guarantee every push times out.
const (
	minPushInterval = time.Second
	minTimeout      = 100 * time.Millisecond
)

// EmitterConfig configures the OTLP/HTTP JSON client that pushes health
// scores to Fleet or any compatible OTEL Collector. Zero-value = disabled.
type EmitterConfig struct {
	// Endpoint is the Fleet OTLP/HTTP URL. Accepts bare host:port (e.g.
	// "fleet.example:4318") or a full URL. Required.
	Endpoint string `yaml:"endpoint"`
	// ClusterID labels every push with the cluster (Fleet maintains a
	// separate score map per cluster). Required.
	ClusterID string `yaml:"cluster_id"`
	// NodeID labels every push. Caller typically resolves via the agent's
	// existing identity logic and passes it in. Required.
	NodeID string `yaml:"node_id"`
	// WorkloadType is a free-form label ("training", "inference", "unknown")
	// attached to each data point.
	WorkloadType string `yaml:"workload_type"`
	// PushInterval drives the loop cadence. Default 10s.
	PushInterval time.Duration `yaml:"push_interval"`
	// Timeout on a single OTLP push. Default 5s.
	Timeout time.Duration `yaml:"timeout"`
	// Insecure selects HTTP (vs HTTPS) when Endpoint is bare host:port.
	// Full URLs ignore this flag.
	Insecure bool `yaml:"insecure"`
	// FailureThreshold is the number of consecutive push failures after
	// which FleetReachable() flips to false. Default 3.
	FailureThreshold int `yaml:"failure_threshold"`
	// TLS configures mTLS. All three paths are required together; leaving
	// all three empty disables TLS configuration (falls back to system
	// default TLS if the scheme is https).
	TLS TLSConfig `yaml:"tls"`
	// Headers are added to every request (e.g. Authorization: Bearer ...).
	Headers map[string]string `yaml:"headers"`
	// Optional world_size / node_rank for distributed training labeling. A
	// zero WorldSize suppresses both attributes.
	WorldSize int `yaml:"world_size"`
	NodeRank  int `yaml:"node_rank"`
	// GPUModel and GPUCount drive the v0.11 ingero.node.info gauge,
	// which the cost-of-problem recording-rule layer joins against
	// gpu_rates.yaml. Empty model or zero count suppresses the gauge.
	// Caller (cmd/ingero/main.go fleet-push entry) populates from
	// nvidia-smi at startup.
	GPUModel string `yaml:"gpu_model"`
	GPUCount int    `yaml:"gpu_count"`
	// Provider is the agent-side cloud-provider attribution emitted as
	// the ingero.provider resource attribute (v0.12.3 attribution-(b)).
	// Caller resolves it once at startup via internal/provider.DetectDefault
	// (or operator override) and passes it in. Empty value suppresses
	// the attribute, leaving Fleet's providerlookupprocessor in charge.
	Provider string `yaml:"provider"`
	// ThresholdCache, if non-nil, is updated with the `X-Ingero-Threshold`
	// and `X-Ingero-Quorum-Met` response headers on every push (success
	// or failure). Nil leaves response-header handling disabled.
	ThresholdCache *ThresholdCache `yaml:"-"`
}

// TLSConfig carries filesystem paths for mTLS materials.
type TLSConfig struct {
	CACertPath     string `yaml:"ca_cert"`
	ClientCertPath string `yaml:"client_cert"`
	ClientKeyPath  string `yaml:"client_key"`
}

// DefaultEmitterConfig returns non-endpoint defaults. The caller must set
// Endpoint, ClusterID, and NodeID for the emitter to be usable.
func DefaultEmitterConfig() EmitterConfig {
	return EmitterConfig{
		PushInterval:     10 * time.Second,
		Timeout:          5 * time.Second,
		FailureThreshold: 3,
		WorkloadType:     "unknown",
	}
}

// Validate rejects configurations that would make Push fail every time or
// flood the server with sub-second pushes.
func (c EmitterConfig) Validate() error {
	if strings.TrimSpace(c.Endpoint) == "" {
		return errors.New("emitter.endpoint is required")
	}
	if strings.TrimSpace(c.ClusterID) == "" {
		return errors.New("emitter.cluster_id is required")
	}
	if strings.TrimSpace(c.NodeID) == "" {
		return errors.New("emitter.node_id is required")
	}
	if !utf8.ValidString(c.NodeID) {
		return errors.New("emitter.node_id must be valid UTF-8")
	}
	if c.PushInterval < minPushInterval {
		return fmt.Errorf("emitter.push_interval must be >= %s: got %s", minPushInterval, c.PushInterval)
	}
	if c.Timeout < minTimeout {
		return fmt.Errorf("emitter.timeout must be >= %s: got %s", minTimeout, c.Timeout)
	}
	if c.Timeout > c.PushInterval {
		return fmt.Errorf("emitter.timeout (%s) must be <= push_interval (%s)", c.Timeout, c.PushInterval)
	}
	if c.FailureThreshold <= 0 {
		return errors.New("emitter.failure_threshold must be > 0")
	}
	if c.WorldSize < 0 {
		return fmt.Errorf("emitter.world_size must be >= 0: got %d", c.WorldSize)
	}
	if c.NodeRank < 0 {
		return fmt.Errorf("emitter.node_rank must be >= 0: got %d", c.NodeRank)
	}
	if c.WorldSize > 0 && c.NodeRank >= c.WorldSize {
		return fmt.Errorf("emitter.node_rank (%d) must be < world_size (%d)", c.NodeRank, c.WorldSize)
	}
	// Reject CR/LF in identity fields that propagate to OTLP attributes
	// and gRPC metadata. CR/LF is valid UTF-8 but can cause header injection.
	for _, pair := range []struct{ name, val string }{
		{"cluster_id", c.ClusterID},
		{"node_id", c.NodeID},
		{"workload_type", c.WorkloadType},
	} {
		if strings.ContainsAny(pair.val, "\r\n") {
			return fmt.Errorf("emitter.%s: contains CR or LF", pair.name)
		}
	}
	// Header-injection: reject any key or value containing \r or \n.
	for k, v := range c.Headers {
		if strings.ContainsAny(k, "\r\n") || strings.ContainsAny(v, "\r\n") {
			return fmt.Errorf("emitter.headers[%q]: contains CR or LF", k)
		}
	}
	t := c.TLS
	anySet := t.CACertPath != "" || t.ClientCertPath != "" || t.ClientKeyPath != ""
	allSet := t.CACertPath != "" && t.ClientCertPath != "" && t.ClientKeyPath != ""
	if anySet && !allSet {
		return errors.New("emitter.tls: all three of ca_cert/client_cert/client_key must be set together")
	}
	return nil
}

// Emitter pushes one health snapshot per call. Thread-safe. Tracks
// consecutive failures for FleetReachable().
type Emitter interface {
	// Push emits one OTLP payload carrying the score, state, and per-signal
	// metrics. Returns nil on 2xx, a wrapped error otherwise. Does not
	// block or panic; callers must still respect ctx cancellation. The
	// perCGroup slice carries this push's window deltas for the per-cgroup
	// CUDA + CPU-stall metric families; passing nil or empty means "no
	// per-cgroup data this tick" and the cumulative counters do not move.
	Push(ctx context.Context, now time.Time, score Score, state State, detectionMode string, degradation bool, perCGroup []PerCGroupStats) error
	// EmitStragglerEvent pushes a single `ingero.node.straggler_event`
	// metric with value=1 (isStraggler=true) or value=0 (recovery edge,
	// isStraggler=false). Attributes on the data point carry threshold,
	// score, and dominant_signal for drill-down. Used for edge-triggered
	// notifications alongside the regular Push cadence.
	EmitStragglerEvent(ctx context.Context, ev StragglerEvent, isStraggler bool) error
	// FleetReachable returns false after FailureThreshold consecutive Push
	// failures; resets to true on the first successful Push.
	FleetReachable() bool
	// Stats returns cumulative counters.
	Stats() (pushes, errors int64)
}

// httpEmitter is the production Emitter. OTLP/HTTP JSON only - no gRPC,
// following the existing internal/export/otlp.go pattern.
type httpEmitter struct {
	cfg    EmitterConfig
	url    string
	client *http.Client
	log    *slog.Logger

	// cgroupPathHash is the SHA256-truncated-16 hash of the agent's own
	// /proc/self/cgroup path, resolved once at NewEmitter and emitted on
	// every health_score data point so Fleet can group MAD by cohort.
	// Empty string when resolution failed (e.g. /proc unavailable);
	// downstream Fleet folds that into the legacy cluster-wide bucket.
	cgroupPathHash string

	mu                  sync.Mutex
	consecutiveFailures int
	rng                 *rand.Rand // guarded by mu

	pushes atomic.Int64
	errors atomic.Int64

	// counters holds the running cumulative totals for the per-cgroup
	// Sum metrics. Each Push adds the window delta to the appropriate
	// map and emits the resulting cumulative value, matching OTLP's
	// AGGREGATION_TEMPORALITY_CUMULATIVE contract.
	counters counterState
}

// counterState carries cumulative per-series totals across pushes.
// Guarded by counters.mu so concurrent Push() callers do not race.
type counterState struct {
	mu           sync.Mutex
	kernelLaunch map[string]int64       // cgroup_path_hash -> total launches
	cpuStall     map[string]int64       // cgroup_path_hash -> total off-CPU nanos
	memcpyBytes  map[memcpyCounterKey]int64
}

type memcpyCounterKey struct {
	Hash      string
	Direction string
}

// NewEmitter constructs an httpEmitter. Returns an error if cfg is invalid
// or TLS material cannot be loaded.
func NewEmitter(cfg EmitterConfig, log *slog.Logger) (Emitter, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if log == nil {
		log = slog.Default()
	}

	endpointURL, err := buildURL(cfg.Endpoint, cfg.Insecure, cfg.ClusterID)
	if err != nil {
		return nil, fmt.Errorf("emitter.endpoint: %w", err)
	}

	// Build a Transport that re-resolves DNS periodically so agents
	// rebalance across Fleet replicas behind a headless Service. The
	// default http.DefaultTransport pool sticks to whatever IP the first
	// dial resolved; with 5s push intervals and 30s idle timeout, most
	// pushes redial, which picks up new A/AAAA records.
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.IdleConnTimeout = 30 * time.Second
	tr.MaxIdleConnsPerHost = 1
	client := &http.Client{Timeout: cfg.Timeout, Transport: tr}
	if cfg.TLS.CACertPath != "" {
		tlsCfg, tlsErr := LoadTLSConfig(cfg.TLS)
		if tlsErr != nil {
			return nil, fmt.Errorf("emitter.tls: %w", tlsErr)
		}
		trTLS := tr.Clone()
		trTLS.TLSClientConfig = tlsCfg
		client.Transport = trTLS
	}

	// Resolve the agent's own cgroup path hash once at startup. On any
	// failure (no /proc, permission denied, host process), we emit the
	// empty string on the wire, which Fleet folds into the legacy
	// cluster-wide bucket. WARN once so operators see the fallback.
	cgroupHash, cgroupErr := cgroupPathHashResolver()
	if cgroupErr != nil {
		log.Warn("cgroup path hash resolution failed; emitting empty value on health_score",
			"err", cgroupErr.Error())
		cgroupHash = ""
	}

	return &httpEmitter{
		cfg:            cfg,
		url:            endpointURL,
		client:         client,
		log:            log,
		cgroupPathHash: cgroupHash,
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
		counters: counterState{
			kernelLaunch: make(map[string]int64),
			cpuStall:     make(map[string]int64),
			memcpyBytes:  make(map[memcpyCounterKey]int64),
		},
	}, nil
}

// Push emits one OTLP payload. The attempt sequence is:
//  1. primary Do()
//  2. if primary returned a network-class error (not 4xx/5xx), sleep a
//     short jittered delay and retry once.
//
// A 4xx/5xx response is not retried — Fleet itself rejected the payload,
// so spamming isn't useful. Only the SECOND failure (if any) increments
// the consecutive-failure counter, so a brief network blip does not
// falsely trip FleetReachable.
func (e *httpEmitter) Push(ctx context.Context, now time.Time, score Score, state State, detectionMode string, degradation bool, perCGroup []PerCGroupStats) error {
	payload := e.buildPayload(now, score, state, detectionMode, degradation, perCGroup)
	body, err := json.Marshal(payload)
	if err != nil {
		e.recordFailure()
		return fmt.Errorf("emitter: marshal: %w", err)
	}

	// First attempt.
	statusErr, netErr := e.doPush(ctx, body)
	if statusErr == nil && netErr == nil {
		e.recordSuccess()
		return nil
	}
	if statusErr != nil {
		// 4xx/5xx — do not retry, do not flood the server.
		e.recordFailure()
		return statusErr
	}

	// Network-class error: one retry after jittered backoff.
	if ctx.Err() != nil {
		e.recordFailure()
		return fmt.Errorf("emitter: push: %w", netErr)
	}
	delay := e.jitter(200 * time.Millisecond)
	select {
	case <-ctx.Done():
		e.recordFailure()
		return fmt.Errorf("emitter: push: %w", ctx.Err())
	case <-time.After(delay):
	}
	statusErr2, netErr2 := e.doPush(ctx, body)
	if statusErr2 == nil && netErr2 == nil {
		e.recordSuccess()
		return nil
	}
	e.recordFailure()
	if statusErr2 != nil {
		return statusErr2
	}
	e.log.Debug("fleet push failed after retry", "first", netErr.Error(), "second", netErr2.Error())
	return fmt.Errorf("emitter: push: %w", netErr2)
}

// doPush performs a single HTTP attempt. Returns (statusErr, netErr) —
// exactly one is non-nil on failure, both nil on 2xx. On any response
// received from the server (including non-2xx), the threshold cache is
// updated from response headers if configured.
func (e *httpEmitter) doPush(ctx context.Context, body []byte) (statusErr error, netErr error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("emitter: new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range e.cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	// Threshold piggyback: parse headers from any response the server
	// returned (2xx or error). A non-2xx may still carry fresh threshold
	// values via middleware.
	if e.cfg.ThresholdCache != nil {
		e.cfg.ThresholdCache.ParseAndSetHTTPHeaders(resp.Header, time.Now())
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// If the server set Retry-After on a 429/503, surface it to the
		// caller so the loop can delay the next tick. Missing header
		// falls through to the existing generic status-error path; the
		// loop then uses its normal push-interval tick.
		if resp.StatusCode == http.StatusTooManyRequests ||
			resp.StatusCode == http.StatusServiceUnavailable {
			if d, ok := parseRetryAfterHeader(resp.Header.Get("Retry-After")); ok {
				return &RetryAfterError{StatusCode: resp.StatusCode, Delay: d}, nil
			}
		}
		return fmt.Errorf("emitter: push rejected: %d %s", resp.StatusCode, resp.Status), nil
	}
	return nil, nil
}

func (e *httpEmitter) jitter(base time.Duration) time.Duration {
	e.mu.Lock()
	defer e.mu.Unlock()
	// Jitter ±20% around base.
	pct := e.rng.Float64()*0.4 - 0.2
	return base + time.Duration(float64(base)*pct)
}

// EmitStragglerEvent implements the Emitter interface.
func (e *httpEmitter) EmitStragglerEvent(ctx context.Context, ev StragglerEvent, isStraggler bool) error {
	payload := e.buildStragglerEventPayload(ev, isStraggler)
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("emitter: marshal straggler: %w", err)
	}
	statusErr, netErr := e.doPush(ctx, body)
	if statusErr == nil && netErr == nil {
		return nil
	}
	// Straggler events don't participate in FleetReachable accounting —
	// they ride on a best-effort channel separate from the main push
	// cadence (which IS what owns reachability semantics).
	if statusErr != nil {
		return statusErr
	}
	return fmt.Errorf("emitter: straggler push: %w", netErr)
}

func (e *httpEmitter) buildStragglerEventPayload(ev StragglerEvent, isStraggler bool) otlpPayload {
	timeNano := fmt.Sprintf("%d", ev.Timestamp.UnixNano())
	var value int64
	if isStraggler {
		value = 1
	}
	attrs := []otlpKV{
		{Key: contract.AttrDetectionMode, Value: otlpStr(string(ev.DetectionMode))},
		{Key: contract.AttrThreshold, Value: otlpDbl(ev.Threshold)},
		{Key: contract.AttrScore, Value: otlpDbl(ev.Score)},
		{Key: contract.AttrDominantSignal, Value: otlpStr(ev.DominantSignal)},
	}
	if ev.EventID != "" {
		attrs = append(attrs, otlpKV{Key: contract.AttrEventID, Value: otlpStr(ev.EventID)})
	}
	resAttrs := []otlpKV{
		{Key: contract.AttrNodeID, Value: otlpStr(ev.NodeID)},
		{Key: contract.AttrClusterID, Value: otlpStr(ev.ClusterID)},
	}
	if e.cfg.WorldSize > 0 {
		resAttrs = append(resAttrs,
			otlpKV{Key: contract.AttrWorldSize, Value: otlpInt(int64(e.cfg.WorldSize))},
			otlpKV{Key: contract.AttrNodeRank, Value: otlpInt(int64(e.cfg.NodeRank))},
		)
	}
	return otlpPayload{
		ResourceMetrics: []otlpResourceMetricsBlock{{
			Resource: otlpResourceBlock{
				Attributes: resAttrs,
			},
			ScopeMetrics: []otlpScopeMetricsBlock{{
				Scope:   otlpScopeBlock{Name: "ingero.health", Version: "0.10.0"},
				Metrics: []otlpMetric{intGauge(contract.MetricStragglerEvent, timeNano, value, attrs)},
			}},
		}},
	}
}

func (e *httpEmitter) FleetReachable() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.consecutiveFailures < e.cfg.FailureThreshold
}

func (e *httpEmitter) Stats() (int64, int64) {
	return e.pushes.Load(), e.errors.Load()
}

func (e *httpEmitter) recordSuccess() {
	e.pushes.Add(1)
	e.mu.Lock()
	e.consecutiveFailures = 0
	e.mu.Unlock()
}

func (e *httpEmitter) recordFailure() {
	e.errors.Add(1)
	e.mu.Lock()
	e.consecutiveFailures++
	e.mu.Unlock()
}

// buildPayload constructs the full OTLP/HTTP JSON body for one push. The
// shape and attribute keys are driven by pkg/contract.
func (e *httpEmitter) buildPayload(now time.Time, score Score, state State, mode string, degradation bool, perCGroup []PerCGroupStats) otlpPayload {
	timeNano := fmt.Sprintf("%d", now.UnixNano())

	// Per-data-point attributes (per-metric, can vary). world_size/node_rank
	// are stable per-agent identity and live on the resource block instead
	// (see below); placing them only on the resource avoids duplication and
	// matches OTEL convention for identity-shaped attributes.
	dpAttrs := []otlpKV{
		{Key: contract.AttrNodeState, Value: otlpStr(string(state))},
		{Key: contract.AttrWorkloadType, Value: otlpStr(e.cfg.WorkloadType)},
	}

	// health_score carries the cgroup_path_hash so Fleet can group MAD
	// thresholds by cohort. The per-signal sub-gauges share the same
	// state/workload attrs but do not need cgroup attribution (Fleet's
	// threshold pipeline only consumes the composite score). The hash
	// is resolved once at NewEmitter and cached on the emitter.
	healthDpAttrs := append([]otlpKV{}, dpAttrs...)
	healthDpAttrs = append(healthDpAttrs,
		otlpKV{Key: contract.AttrCgroupPathHash, Value: otlpStr(e.cgroupPathHash)},
	)

	// Score + per-signal gauges.
	var metrics []otlpMetric
	metrics = append(metrics,
		dblGauge(contract.MetricHealthScore, timeNano, score.Value, healthDpAttrs),
		dblGauge(contract.MetricThroughputRatio, timeNano, score.Throughput, dpAttrs),
		dblGauge(contract.MetricComputeEfficiency, timeNano, score.Compute, dpAttrs),
		dblGauge(contract.MetricMemoryHeadroom, timeNano, score.Memory, dpAttrs),
		dblGauge(contract.MetricCPUAvailability, timeNano, score.CPU, dpAttrs),
	)

	degrade := int64(0)
	if degradation {
		degrade = 1
	}
	metrics = append(metrics, intGauge(contract.MetricDegradationWarning, timeNano, degrade, dpAttrs))

	// Detection mode: Gauge int 1 with mode attribute (per contract).
	modeAttrs := append([]otlpKV{}, otlpKV{Key: contract.AttrDetectionMode, Value: otlpStr(mode)})
	metrics = append(metrics, intGauge(contract.MetricDetectionMode, timeNano, 1, modeAttrs))

	// FleetReachable as int gauge (this value reflects the emitter's own
	// belief at the moment of the push).
	reachable := int64(1)
	if !e.FleetReachable() {
		reachable = 0
	}
	metrics = append(metrics, intGauge(contract.MetricFleetReachable, timeNano, reachable, nil))

	// v0.11 cost-of-problem support gauges. ingero.node.info lights up
	// only when the operator has supplied gpu_model / gpu_count (caller
	// reads them from nvidia-smi at startup); ingero.node.world_size
	// always emits with the configured value (zero is meaningful — it
	// signals a non-distributed deployment).
	if e.cfg.GPUModel != "" && e.cfg.GPUCount > 0 {
		nodeInfoAttrs := []otlpKV{
			{Key: contract.AttrGPUModel, Value: otlpStr(e.cfg.GPUModel)},
			{Key: contract.AttrGPUCount, Value: otlpInt(int64(e.cfg.GPUCount))},
		}
		metrics = append(metrics, intGauge(contract.MetricNodeInfo, timeNano, 1, nodeInfoAttrs))
	}
	metrics = append(metrics, intGauge(contract.MetricNodeWorldSize, timeNano, int64(e.cfg.WorldSize), nil))

	if len(perCGroup) > 0 {
		metrics = append(metrics, e.buildPerCGroupMetrics(timeNano, perCGroup)...)
	}

	return otlpPayload{
		ResourceMetrics: []otlpResourceMetricsBlock{{
			Resource: otlpResourceBlock{
				Attributes: e.resourceAttrs(),
			},
			ScopeMetrics: []otlpScopeMetricsBlock{{
				Scope:   otlpScopeBlock{Name: "ingero.health", Version: "0.10.0"},
				Metrics: metrics,
			}},
		}},
	}
}

// resourceAttrs builds the OTLP resource-attribute set carried on every
// push. Includes the stable per-agent identity (node_id, cluster_id) and
// optionally world_size + node_rank when configured. The whole list is
// stable for the lifetime of the emitter; callers may rely on it being
// identical across consecutive pushes.
func (e *httpEmitter) resourceAttrs() []otlpKV {
	attrs := []otlpKV{
		{Key: contract.AttrNodeID, Value: otlpStr(e.cfg.NodeID)},
		{Key: contract.AttrClusterID, Value: otlpStr(e.cfg.ClusterID)},
	}
	if e.cfg.WorldSize > 0 {
		attrs = append(attrs,
			otlpKV{Key: contract.AttrWorldSize, Value: otlpInt(int64(e.cfg.WorldSize))},
			otlpKV{Key: contract.AttrNodeRank, Value: otlpInt(int64(e.cfg.NodeRank))},
		)
	}
	if e.cfg.Provider != "" {
		attrs = append(attrs, otlpKV{Key: contract.AttrProvider, Value: otlpStr(e.cfg.Provider)})
	}
	return attrs
}

// buildURL normalizes endpoint into a full OTLP/HTTP metrics URL. Accepts:
//   - bare host:port (applies http/https based on insecure)
//   - full URL with or without path
//   - URL whose path already includes /v1/metrics (not duplicated)
//   - IPv6 host forms [::1]:4318
//
// Returns an error on malformed URL or unsupported scheme.
func buildURL(endpoint string, insecure bool, clusterID string) (string, error) {
	endpoint = strings.TrimSpace(endpoint)
	if endpoint == "" {
		return "", errors.New("empty endpoint")
	}

	var u *url.URL
	var err error
	if strings.Contains(endpoint, "://") {
		u, err = url.Parse(endpoint)
		if err != nil {
			return "", fmt.Errorf("parse: %w", err)
		}
	} else {
		// Bare host:port. Synthesize the scheme. url.Parse with a leading
		// "//" gives us a network-authority component that handles IPv6.
		scheme := "https"
		if insecure {
			scheme = "http"
		}
		u, err = url.Parse(scheme + "://" + endpoint)
		if err != nil {
			return "", fmt.Errorf("parse: %w", err)
		}
	}

	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("unsupported scheme %q", u.Scheme)
	}
	if u.Host == "" {
		return "", errors.New("missing host")
	}

	// Path canonicalization: strip trailing slash, then append /v1/metrics
	// unless it's already present.
	p := strings.TrimRight(u.Path, "/")
	if !strings.HasSuffix(p, "/v1/metrics") {
		p += "/v1/metrics"
	}
	u.Path = p

	// Append cluster_id as a query parameter so the Fleet middleware can
	// identify the cluster at the HTTP layer (before parsing the OTLP
	// payload) and inject piggyback threshold headers into the response.
	if clusterID != "" {
		q := u.Query()
		q.Set(contract.ParamClusterID, clusterID)
		u.RawQuery = q.Encode()
	}

	return u.String(), nil
}

// LoadTLSConfig reads mTLS material from disk and returns a *tls.Config
// usable for both the emitter's OTLP push client and the threshold-API GET
// poller. tls.LoadX509KeyPair already verifies that the private key matches
// the client certificate, so a mismatched pair fails at config time rather
// than at first handshake.
func LoadTLSConfig(t TLSConfig) (*tls.Config, error) {
	caPEM, err := os.ReadFile(t.CACertPath)
	if err != nil {
		return nil, fmt.Errorf("read ca_cert: %w", err)
	}
	if len(strings.TrimSpace(string(caPEM))) == 0 {
		return nil, fmt.Errorf("ca_cert %q: file is empty", t.CACertPath)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caPEM) {
		return nil, fmt.Errorf("ca_cert %q: no valid certificates", t.CACertPath)
	}
	cert, err := auth.LoadTLSKeyPair(t.ClientCertPath, t.ClientKeyPath)
	if err != nil {
		return nil, fmt.Errorf("load client keypair: %w", err)
	}
	return &tls.Config{
		RootCAs:      pool,
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
	}, nil
}

// ---------- OTLP/HTTP JSON shapes ----------
//
// Minimal structural duplication of internal/export/otlp.go — local types
// keep this package self-contained and avoid coupling the health emitter
// to the stats-engine exporter. Field names match the OTLP/HTTP JSON spec.

type otlpPayload struct {
	ResourceMetrics []otlpResourceMetricsBlock `json:"resourceMetrics"`
}

type otlpResourceMetricsBlock struct {
	Resource     otlpResourceBlock       `json:"resource"`
	ScopeMetrics []otlpScopeMetricsBlock `json:"scopeMetrics"`
}

type otlpResourceBlock struct {
	Attributes []otlpKV `json:"attributes"`
}

type otlpScopeMetricsBlock struct {
	Scope   otlpScopeBlock `json:"scope"`
	Metrics []otlpMetric   `json:"metrics"`
}

type otlpScopeBlock struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type otlpMetric struct {
	Name  string       `json:"name"`
	Unit  string       `json:"unit,omitempty"`
	Gauge *otlpData    `json:"gauge,omitempty"`
	Sum   *otlpSumData `json:"sum,omitempty"`
}

type otlpData struct {
	DataPoints []otlpDP `json:"dataPoints"`
}

// otlpSumData is the OTLP Sum metric carrier. The agent emits cumulative
// monotonic counters: aggregationTemporality=2 (CUMULATIVE),
// isMonotonic=true. Consumers (Prometheus, Datadog) compute rates.
type otlpSumData struct {
	DataPoints             []otlpDP `json:"dataPoints"`
	AggregationTemporality int      `json:"aggregationTemporality"`
	IsMonotonic            bool     `json:"isMonotonic"`
}

type otlpDP struct {
	Attributes   []otlpKV `json:"attributes,omitempty"`
	TimeUnixNano string   `json:"timeUnixNano"`
	AsDouble     *float64 `json:"asDouble,omitempty"`
	AsInt        *int64   `json:"asInt,omitempty"`
}

type otlpKV struct {
	Key   string   `json:"key"`
	Value otlpVal  `json:"value"`
}

type otlpVal struct {
	StringValue *string  `json:"stringValue,omitempty"`
	IntValue    *int64   `json:"intValue,omitempty"`
	DoubleValue *float64 `json:"doubleValue,omitempty"`
}

func otlpStr(s string) otlpVal { return otlpVal{StringValue: &s} }
func otlpInt(i int64) otlpVal  { return otlpVal{IntValue: &i} }
func otlpDbl(f float64) otlpVal { return otlpVal{DoubleValue: &f} }

func dblGauge(name, timeNano string, v float64, attrs []otlpKV) otlpMetric {
	return otlpMetric{
		Name: name,
		Unit: "1",
		Gauge: &otlpData{
			DataPoints: []otlpDP{{Attributes: attrs, TimeUnixNano: timeNano, AsDouble: &v}},
		},
	}
}

func intGauge(name, timeNano string, v int64, attrs []otlpKV) otlpMetric {
	return otlpMetric{
		Name: name,
		Unit: "1",
		Gauge: &otlpData{
			DataPoints: []otlpDP{{Attributes: attrs, TimeUnixNano: timeNano, AsInt: &v}},
		},
	}
}

// intSum constructs a cumulative monotonic int64 Sum metric with the
// given data points. AggregationTemporality=2 is CUMULATIVE per the
// OTLP proto (1 = DELTA). Empty unit lets Fleet / Prometheus pick a
// rendering default; specific units may be added when a downstream
// dashboard demands them.
func intSum(name, unit string, dps []otlpDP) otlpMetric {
	return otlpMetric{
		Name: name,
		Unit: unit,
		Sum: &otlpSumData{
			DataPoints:             dps,
			AggregationTemporality: 2,
			IsMonotonic:            true,
		},
	}
}

// buildPerCGroupMetrics adds the window deltas in perCGroup to the
// emitter's cumulative counter state and returns one Sum data point per
// (cgroup_path_hash, op) for kernel-launch + cpu-stall, and per
// (cgroup_path_hash, direction) for memcpy. Series are emitted only
// when the cumulative total is non-zero, so a zero-workload cgroup that
// never had a launch does not pollute the metric stream.
func (e *httpEmitter) buildPerCGroupMetrics(timeNano string, perCGroup []PerCGroupStats) []otlpMetric {
	e.counters.mu.Lock()
	for _, s := range perCGroup {
		if s.KernelLaunchCount > 0 {
			e.counters.kernelLaunch[s.CgroupPathHash] += s.KernelLaunchCount
		} else if _, ok := e.counters.kernelLaunch[s.CgroupPathHash]; !ok {
			// Touch the hash so a series exists at zero on the first
			// tick for a freshly-discovered cgroup; downstream rate()
			// queries do not need to wait for a non-zero increment.
			e.counters.kernelLaunch[s.CgroupPathHash] = 0
		}
		if s.CPUStallNanos > 0 {
			e.counters.cpuStall[s.CgroupPathHash] += s.CPUStallNanos
		} else if _, ok := e.counters.cpuStall[s.CgroupPathHash]; !ok {
			e.counters.cpuStall[s.CgroupPathHash] = 0
		}
		for dir, bytes := range s.MemcpyBytesByDir {
			key := memcpyCounterKey{Hash: s.CgroupPathHash, Direction: dir}
			e.counters.memcpyBytes[key] += bytes
		}
	}

	launchDP := make([]otlpDP, 0, len(e.counters.kernelLaunch))
	// Per-iteration scoping (Go 1.22+) makes &v safe across distinct cgroups.
	for hash, total := range e.counters.kernelLaunch {
		v := total
		launchDP = append(launchDP, otlpDP{
			Attributes:   []otlpKV{{Key: contract.AttrCgroupPathHash, Value: otlpStr(hash)}},
			TimeUnixNano: timeNano,
			AsInt:        &v,
		})
	}
	stallDP := make([]otlpDP, 0, len(e.counters.cpuStall))
	for hash, total := range e.counters.cpuStall {
		v := total
		stallDP = append(stallDP, otlpDP{
			Attributes:   []otlpKV{{Key: contract.AttrCgroupPathHash, Value: otlpStr(hash)}},
			TimeUnixNano: timeNano,
			AsInt:        &v,
		})
	}
	memcpyDP := make([]otlpDP, 0, len(e.counters.memcpyBytes))
	for key, total := range e.counters.memcpyBytes {
		v := total
		memcpyDP = append(memcpyDP, otlpDP{
			Attributes: []otlpKV{
				{Key: contract.AttrCgroupPathHash, Value: otlpStr(key.Hash)},
				{Key: contract.AttrMemcpyDirection, Value: otlpStr(key.Direction)},
			},
			TimeUnixNano: timeNano,
			AsInt:        &v,
		})
	}
	e.counters.mu.Unlock()

	out := make([]otlpMetric, 0, 3)
	if len(launchDP) > 0 {
		out = append(out, intSum(contract.MetricCUDAKernelLaunchTotal, "1", launchDP))
	}
	if len(stallDP) > 0 {
		out = append(out, intSum(contract.MetricCPUStallNanosTotal, "ns", stallDP))
	}
	if len(memcpyDP) > 0 {
		out = append(out, intSum(contract.MetricCUDAMemcpyBytesTotal, "By", memcpyDP))
	}
	return out
}
