// Package export provides OTEL-compatible metric and trace export.
//
// Architecture: parallel consumers of the Stats Engine snapshot.
// OTLP and Prometheus are OPTIONAL — disabled by default, enabled via
// --otlp <endpoint> or --prometheus <addr> flags.
//
// Call chain: watch.go calls export.OTLP.Push(snap) every ExportInterval →
//   OTLP exporter serializes metrics as OTLP/HTTP JSON →
//   HTTP POST to <endpoint>/v1/metrics
//
// OTEL semantic conventions used:
//   gpu.cuda.operation.duration  — per-op latency percentiles (microseconds)
//   gpu.cuda.operation.count     — per-op event counts
//   system.cpu.utilization       — system CPU ratio (0-1)
//   system.memory.utilization    — system memory ratio (0-1)
//   ingero.anomaly.count         — anomaly event count
package export

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/ingero-io/ingero/internal/stats"
)

// OTLPConfig configures the OTLP exporter.
type OTLPConfig struct {
	Endpoint       string // e.g., "localhost:4318" (HTTP) or "localhost:4317" (gRPC)
	Protocol       string // "http" (default) or "grpc" (future)
	Insecure       bool
	ExportInterval int // seconds between pushes (default 10)
	Headers        map[string]string

	// DebugLog is called for debug messages when set (typically cli.debugf).
	DebugLog func(format string, args ...any)
}

// OTLPExporter pushes metrics to an OTLP-compatible receiver via HTTP JSON.
//
// Uses the OTLP/HTTP JSON protocol (POST to /v1/metrics with JSON body).
// Compatible with: OpenTelemetry Collector, Grafana Alloy, Grafana Cloud,
// Datadog Agent, New Relic, any OTLP-compatible receiver.
//
// Zero external dependencies — uses only net/http and encoding/json.
type OTLPExporter struct {
	config      OTLPConfig
	client      *http.Client
	mu          sync.Mutex
	pushes      int64
	errors      int64
	warnedOnce  bool // first error logged to stderr (even without --debug)
}

// NewOTLP creates a new OTLP exporter. Returns nil if endpoint is empty.
func NewOTLP(cfg OTLPConfig) *OTLPExporter {
	if cfg.Endpoint == "" {
		return nil
	}
	if cfg.ExportInterval <= 0 {
		cfg.ExportInterval = 10
	}
	if cfg.Protocol == "" {
		cfg.Protocol = "http"
	}
	return &OTLPExporter{
		config: cfg,
		client: &http.Client{Timeout: 10 * time.Second},
	}
}

// Start begins the periodic export loop. Blocks until ctx is cancelled.
func (e *OTLPExporter) Start(ctx context.Context) error {
	if e == nil {
		return nil
	}
	e.debugf("OTLP: exporter started, endpoint=%s, protocol=%s, interval=%ds",
		e.config.Endpoint, e.config.Protocol, e.config.ExportInterval)
	<-ctx.Done()
	e.debugf("OTLP: exporter stopped (%d pushes, %d errors)", e.pushes, e.errors)
	return nil
}

// Push sends a stats snapshot as OTLP metrics via HTTP JSON.
// Called every ExportInterval seconds from the watch loop.
func (e *OTLPExporter) Push(snap *stats.Snapshot) error {
	if e == nil {
		return nil
	}

	payload := e.buildMetricsPayload(snap)

	body, err := json.Marshal(payload)
	if err != nil {
		e.mu.Lock()
		e.errors++
		e.mu.Unlock()
		e.debugf("OTLP: marshal error: %v", err)
		return fmt.Errorf("OTLP marshal: %w", err)
	}

	url := e.metricsURL()
	e.debugf("OTLP: pushing %d bytes to %s (%d metrics)",
		len(body), url, countMetrics(payload))

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		e.mu.Lock()
		e.errors++
		e.mu.Unlock()
		return fmt.Errorf("OTLP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range e.config.Headers {
		req.Header.Set(k, v)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		e.mu.Lock()
		e.errors++
		if !e.warnedOnce {
			e.warnedOnce = true
			// Log first error to stderr even without --debug, so the user
			// knows OTLP export is failing (not silently swallowed).
			fmt.Fprintf(os.Stderr, "  Warning: OTLP push to %s failed: %v\n", e.config.Endpoint, err)
			fmt.Fprintf(os.Stderr, "  Subsequent OTLP errors will only appear with --debug.\n")
		}
		e.mu.Unlock()
		e.debugf("OTLP: push failed: %v", err)
		return fmt.Errorf("OTLP push: %w", err)
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)

	e.mu.Lock()
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		e.pushes++
		e.mu.Unlock()
		e.debugf("OTLP: push OK (%d %s)", resp.StatusCode, resp.Status)
		return nil
	}
	e.errors++
	e.mu.Unlock()
	e.debugf("OTLP: push rejected: %d %s", resp.StatusCode, resp.Status)
	return fmt.Errorf("OTLP push: %d %s", resp.StatusCode, resp.Status)
}

// Interval returns the export interval in seconds.
func (e *OTLPExporter) Interval() int {
	if e == nil {
		return 10
	}
	return e.config.ExportInterval
}

// Stats returns push/error counts for diagnostics.
func (e *OTLPExporter) Stats() (pushes, errors int64) {
	if e == nil {
		return 0, 0
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.pushes, e.errors
}

// metricsURL builds the full URL for the OTLP/HTTP metrics endpoint.
func (e *OTLPExporter) metricsURL() string {
	endpoint := e.config.Endpoint
	scheme := "https"
	if e.config.Insecure {
		scheme = "http"
	}
	// If the endpoint already has a scheme, use it as-is.
	if strings.HasPrefix(endpoint, "http://") || strings.HasPrefix(endpoint, "https://") {
		return strings.TrimRight(endpoint, "/") + "/v1/metrics"
	}
	return fmt.Sprintf("%s://%s/v1/metrics", scheme, endpoint)
}

func (e *OTLPExporter) debugf(format string, args ...any) {
	if e.config.DebugLog != nil {
		e.config.DebugLog(format, args...)
	}
}

// ---------------------------------------------------------------------------
// OTLP/HTTP JSON payload construction
//
// OTLP defines a protobuf schema for metrics, but the HTTP transport accepts
// JSON encoding of the same schema. We construct the JSON directly using maps
// to avoid importing the protobuf definitions (zero-dependency approach).
//
// Reference: https://opentelemetry.io/docs/specs/otlp/#otlphttp
// ---------------------------------------------------------------------------

type otlpPayload struct {
	ResourceMetrics []otlpResourceMetrics `json:"resourceMetrics"`
}

type otlpResourceMetrics struct {
	Resource     otlpResource      `json:"resource"`
	ScopeMetrics []otlpScopeMetrics `json:"scopeMetrics"`
}

type otlpResource struct {
	Attributes []otlpKeyValue `json:"attributes"`
}

type otlpScopeMetrics struct {
	Scope   otlpScope    `json:"scope"`
	Metrics []otlpMetric `json:"metrics"`
}

type otlpScope struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type otlpMetric struct {
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	Unit        string    `json:"unit,omitempty"`
	Gauge       *otlpData `json:"gauge,omitempty"`
	Sum         *otlpData `json:"sum,omitempty"`
}

type otlpData struct {
	DataPoints              []otlpDataPoint `json:"dataPoints"`
	AggregationTemporality  int             `json:"aggregationTemporality,omitempty"` // 1=delta, 2=cumulative
	IsMonotonic             bool            `json:"isMonotonic,omitempty"`
}

type otlpDataPoint struct {
	Attributes     []otlpKeyValue `json:"attributes,omitempty"`
	TimeUnixNano   string         `json:"timeUnixNano"`
	AsDouble       *float64       `json:"asDouble,omitempty"`
	AsInt          *int64         `json:"asInt,omitempty"`
}

type otlpKeyValue struct {
	Key   string     `json:"key"`
	Value otlpValue  `json:"value"`
}

type otlpValue struct {
	StringValue *string `json:"stringValue,omitempty"`
	IntValue    *int64  `json:"intValue,omitempty"`
	DoubleValue *float64 `json:"doubleValue,omitempty"`
}

func stringVal(s string) otlpValue { return otlpValue{StringValue: &s} }

func (e *OTLPExporter) buildMetricsPayload(snap *stats.Snapshot) otlpPayload {
	nowNano := fmt.Sprintf("%d", time.Now().UnixNano())

	var metrics []otlpMetric

	// System metrics.
	if snap.System != nil {
		metrics = append(metrics,
			gaugeMetric("system.cpu.utilization", "System CPU utilization ratio", "1",
				nowNano, snap.System.CPUPercent/100, nil),
			gaugeMetric("system.memory.utilization", "System memory utilization ratio", "1",
				nowNano, snap.System.MemUsedPct/100, nil),
			gaugeMetricInt("system.memory.usage.available", "Available memory", "By",
				nowNano, snap.System.MemAvailMB*1024*1024, nil),
			gaugeMetric("system.cpu.load_average.1m", "1-minute load average", "1",
				nowNano, snap.System.LoadAvg1, nil),
		)
	}

	// Per-operation metrics.
	for _, op := range snap.Ops {
		source := op.Source.String()

		labels := []otlpKeyValue{
			{Key: "source", Value: stringVal(source)},
			{Key: "operation", Value: stringVal(op.Op)},
		}

		// Duration percentiles (gauge, microseconds).
		for _, pct := range []struct {
			name string
			val  time.Duration
		}{
			{"p50", op.P50},
			{"p95", op.P95},
			{"p99", op.P99},
		} {
			pctLabels := append(append([]otlpKeyValue{}, labels...), otlpKeyValue{
				Key: "percentile", Value: stringVal(pct.name),
			})
			metrics = append(metrics, gaugeMetric(
				"gpu.cuda.operation.duration",
				"CUDA operation latency percentile",
				"us",
				nowNano,
				float64(pct.val.Microseconds()),
				pctLabels,
			))
		}

		// Event count (cumulative sum).
		metrics = append(metrics, sumMetric(
			"gpu.cuda.operation.count",
			"Total events per operation",
			"{event}",
			nowNano,
			op.Count,
			labels,
		))
	}

	// Anomaly count.
	metrics = append(metrics, sumMetric(
		"ingero.anomaly.count",
		"Total anomaly events",
		"{event}",
		nowNano,
		int64(snap.AnomalyEvents),
		nil,
	))

	return otlpPayload{
		ResourceMetrics: []otlpResourceMetrics{{
			Resource: otlpResource{
				Attributes: []otlpKeyValue{
					{Key: "service.name", Value: stringVal("ingero")},
					{Key: "service.version", Value: stringVal("0.8.0")},
				},
			},
			ScopeMetrics: []otlpScopeMetrics{{
				Scope: otlpScope{
					Name:    "ingero",
					Version: "0.8.0",
				},
				Metrics: metrics,
			}},
		}},
	}
}

func gaugeMetric(name, desc, unit, timeNano string, value float64, attrs []otlpKeyValue) otlpMetric {
	return otlpMetric{
		Name:        name,
		Description: desc,
		Unit:        unit,
		Gauge: &otlpData{
			DataPoints: []otlpDataPoint{{
				Attributes:   attrs,
				TimeUnixNano: timeNano,
				AsDouble:     &value,
			}},
		},
	}
}

func gaugeMetricInt(name, desc, unit, timeNano string, value int64, attrs []otlpKeyValue) otlpMetric {
	return otlpMetric{
		Name:        name,
		Description: desc,
		Unit:        unit,
		Gauge: &otlpData{
			DataPoints: []otlpDataPoint{{
				Attributes:   attrs,
				TimeUnixNano: timeNano,
				AsInt:        &value,
			}},
		},
	}
}

func sumMetric(name, desc, unit, timeNano string, value int64, attrs []otlpKeyValue) otlpMetric {
	return otlpMetric{
		Name:        name,
		Description: desc,
		Unit:        unit,
		Sum: &otlpData{
			DataPoints: []otlpDataPoint{{
				Attributes:   attrs,
				TimeUnixNano: timeNano,
				AsInt:        &value,
			}},
			AggregationTemporality: 2, // cumulative
			IsMonotonic:            true,
		},
	}
}

func countMetrics(p otlpPayload) int {
	n := 0
	for _, rm := range p.ResourceMetrics {
		for _, sm := range rm.ScopeMetrics {
			n += len(sm.Metrics)
		}
	}
	return n
}
