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

	"github.com/ingero-io/ingero/pkg/contract"
)

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
	// block or panic; callers must still respect ctx cancellation.
	Push(ctx context.Context, now time.Time, score Score, state State, detectionMode string, degradation bool) error
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

// httpEmitter is the production Emitter. OTLP/HTTP JSON only — no gRPC,
// following the existing internal/export/otlp.go pattern.
type httpEmitter struct {
	cfg    EmitterConfig
	url    string
	client *http.Client
	log    *slog.Logger

	mu                  sync.Mutex
	consecutiveFailures int
	rng                 *rand.Rand // guarded by mu

	pushes atomic.Int64
	errors atomic.Int64
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
		tlsCfg, tlsErr := loadTLSConfig(cfg.TLS)
		if tlsErr != nil {
			return nil, fmt.Errorf("emitter.tls: %w", tlsErr)
		}
		trTLS := tr.Clone()
		trTLS.TLSClientConfig = tlsCfg
		client.Transport = trTLS
	}

	return &httpEmitter{
		cfg:    cfg,
		url:    endpointURL,
		client: client,
		log:    log,
		rng:    rand.New(rand.NewSource(time.Now().UnixNano())),
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
func (e *httpEmitter) Push(ctx context.Context, now time.Time, score Score, state State, detectionMode string, degradation bool) error {
	payload := e.buildPayload(now, score, state, detectionMode, degradation)
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
	return otlpPayload{
		ResourceMetrics: []otlpResourceMetricsBlock{{
			Resource: otlpResourceBlock{
				Attributes: []otlpKV{
					{Key: contract.AttrNodeID, Value: otlpStr(ev.NodeID)},
					{Key: contract.AttrClusterID, Value: otlpStr(ev.ClusterID)},
				},
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
func (e *httpEmitter) buildPayload(now time.Time, score Score, state State, mode string, degradation bool) otlpPayload {
	timeNano := fmt.Sprintf("%d", now.UnixNano())

	// Per-data-point attributes (per-metric, can vary).
	dpAttrs := []otlpKV{
		{Key: contract.AttrNodeState, Value: otlpStr(string(state))},
		{Key: contract.AttrWorkloadType, Value: otlpStr(e.cfg.WorkloadType)},
	}
	if e.cfg.WorldSize > 0 {
		dpAttrs = append(dpAttrs,
			otlpKV{Key: contract.AttrWorldSize, Value: otlpInt(int64(e.cfg.WorldSize))},
			otlpKV{Key: contract.AttrNodeRank, Value: otlpInt(int64(e.cfg.NodeRank))},
		)
	}

	// Score + per-signal gauges.
	var metrics []otlpMetric
	metrics = append(metrics,
		dblGauge(contract.MetricHealthScore, timeNano, score.Value, dpAttrs),
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

	return otlpPayload{
		ResourceMetrics: []otlpResourceMetricsBlock{{
			Resource: otlpResourceBlock{
				Attributes: []otlpKV{
					{Key: contract.AttrNodeID, Value: otlpStr(e.cfg.NodeID)},
					{Key: contract.AttrClusterID, Value: otlpStr(e.cfg.ClusterID)},
				},
			},
			ScopeMetrics: []otlpScopeMetricsBlock{{
				Scope:   otlpScopeBlock{Name: "ingero.health", Version: "0.10.0"},
				Metrics: metrics,
			}},
		}},
	}
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

// loadTLSConfig reads mTLS material from disk. tls.LoadX509KeyPair already
// verifies that the private key matches the client certificate, so a
// mismatched pair fails at config time rather than at first handshake.
func loadTLSConfig(t TLSConfig) (*tls.Config, error) {
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
	cert, err := tls.LoadX509KeyPair(t.ClientCertPath, t.ClientKeyPath)
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
	Name  string    `json:"name"`
	Unit  string    `json:"unit,omitempty"`
	Gauge *otlpData `json:"gauge,omitempty"`
}

type otlpData struct {
	DataPoints []otlpDP `json:"dataPoints"`
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
