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

func int64Ptr(v int64) *int64 { return &v }

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

	// NCCL collective data points (v0.12.0+). One nccl.collective.duration_ms
	// gauge per captured event with the standard nccl.* attribute set.
	// Fleet's ncclprocessor consumes these and derives peer_lag_ms.
	//
	// v0.12.1: data points with IsBarrier=true are emitted as
	// `nccl.collective.barrier_wait_ms` instead of duration_ms. The
	// agent-side correlator at internal/cli/nccl_barrier.go produces
	// these by joining NCCL collective uretprobe events with the next
	// cudaStreamSynchronize on the same (pid, stream). Pre-fix this
	// dispatch used a stringly-typed "barrier_wait:" prefix on OpType
	// which was vulnerable to in-band sentinel collisions.
	for _, p := range snap.NCCLDataPoints {
		ts := fmt.Sprintf("%d", p.TimestampUnixNano)
		attrs := []otlpKeyValue{
			{Key: "nccl.op_type", Value: stringVal(p.OpType)},
			{Key: "nccl.comm_id_hash", Value: stringVal(p.CommIDHash)},
			{Key: "nccl.rank", Value: otlpValue{IntValue: int64Ptr(int64(p.Rank))}},
			{Key: "nccl.nranks", Value: otlpValue{IntValue: int64Ptr(int64(p.NRanks))}},
			{Key: "nccl.datatype", Value: otlpValue{IntValue: int64Ptr(int64(p.Datatype))}},
			{Key: "nccl.reduce_op", Value: otlpValue{IntValue: int64Ptr(int64(p.ReduceOp))}},
		}
		// v0.12.2: peer_rank attribute only on ncclSend/ncclRecv
		// (collectives leave PeerRank=0). Topology-mapping for
		// pipeline-parallel workloads.
		if p.PeerRank != 0 {
			attrs = append(attrs, otlpKeyValue{
				Key:   "nccl.peer_rank",
				Value: otlpValue{IntValue: int64Ptr(int64(p.PeerRank))},
			})
		}
		if p.IsBarrier {
			metrics = append(metrics,
				gaugeMetric("nccl.collective.barrier_wait_ms", "Per-collective barrier wait (cudaStreamSynchronize duration after a NCCL collective)", "ms",
					ts, p.DurationMs, attrs),
			)
		} else {
			metrics = append(metrics,
				gaugeMetric("nccl.collective.duration_ms", "Per-collective queue time (entry-to-exit of NCCL uprobe)", "ms",
					ts, p.DurationMs, attrs),
				gaugeMetricInt("nccl.collective.bytes", "Per-collective payload size", "By",
					ts, int64(p.CountBytes), attrs),
			)
		}
	}

	// libnccl process discovery (v0.14 item A). One gpu.nccl.process_loaded
	// gauge=1 per discovered PID, plus gpu.nccl.processes_total per node.
	// The scanner emits an empty slice (not nil) when it has run but
	// found no NCCL processes; the per-node total still emits with
	// value=0 in that case so dashboards can plot "this node has no
	// NCCL workloads right now" alongside positive readings.
	if snap.NCCLProcessReadings != nil {
		for _, r := range snap.NCCLProcessReadings {
			attrs := []otlpKeyValue{
				{Key: "pid", Value: otlpValue{IntValue: int64Ptr(int64(r.PID))}},
				{Key: "comm", Value: stringVal(r.Comm)},
				{Key: "libnccl_path", Value: stringVal(r.LibPath)},
				{Key: "libnccl_version", Value: stringVal(r.LibVersion)},
			}
			metrics = append(metrics,
				gaugeMetricInt("gpu.nccl.process_loaded",
					"NCCL-loaded process discovered on this node (1=present)",
					"1", nowNano, 1, attrs),
			)
		}
		metrics = append(metrics,
			gaugeMetricInt("gpu.nccl.processes_total",
				"Count of NCCL-loaded processes on this node", "1",
				nowNano, int64(len(snap.NCCLProcessReadings)), nil),
		)
	}

	// NVML-poll memfrag heuristic (v0.14 item D, W1 baseline).
	// Polling-based; not the IOCTL-level memfrag tracking that v0.15
	// W1 brings. Four gauges per GPU labelled with gpu.uuid.
	for _, r := range snap.MemFragReadings {
		attrs := []otlpKeyValue{
			{Key: "gpu.uuid", Value: stringVal(r.UUID)},
		}
		metrics = append(metrics,
			gaugeMetricInt("gpu.memory.used", "GPU memory currently allocated (NVML poll)", "By",
				nowNano, r.UsedBytes, attrs),
			gaugeMetricInt("gpu.memory.free", "GPU memory free (NVML poll)", "By",
				nowNano, r.FreeBytes, attrs),
			gaugeMetricInt("gpu.memory.total", "Total GPU memory (NVML poll)", "By",
				nowNano, r.TotalBytes, attrs),
			gaugeMetric("gpu.memory.fragmentation_estimate",
				"Coarse GPU memory fragmentation heuristic from NVML poll [0,1]; v0.15 will replace with IOCTL-level event-driven tracking",
				"1", nowNano, r.FragmentationEstimate, attrs),
		)
	}
	for _, p := range snap.MemFragProcessReadings {
		attrs := []otlpKeyValue{
			{Key: "gpu.uuid", Value: stringVal(p.UUID)},
			{Key: "pid", Value: otlpValue{IntValue: int64Ptr(int64(p.PID))}},
		}
		metrics = append(metrics,
			gaugeMetricInt("gpu.memory.process.allocated_bytes",
				"Per-process GPU memory allocation (NVML compute-apps poll)", "By",
				nowNano, p.UsedBytes, attrs),
		)
	}

	// Per-direction CUDA memcpy aggregates (v0.14 item C). Two
	// metrics per direction: a cumulative counter for byte totals
	// and a per-window gauge for average duration. Direction labels
	// are h2h / h2d / d2h / d2d / default / unknown.
	for _, m := range snap.MemcpyDirReadings {
		attrs := []otlpKeyValue{
			{Key: "direction", Value: stringVal(m.Direction)},
		}
		metrics = append(metrics,
			sumMetric("gpu.memcpy.bytes_total",
				"Cumulative CUDA memcpy bytes by direction (v0.14 item C)",
				"By", nowNano, m.BytesTotal, attrs),
			gaugeMetric("gpu.memcpy.duration_ms",
				"Average per-event CUDA memcpy duration by direction over the last export window",
				"ms", nowNano, m.AverageDurationMs, attrs),
		)
	}

	// NVML clock-throttle reasons (v0.12.10 W2-poller). Four gauges per
	// GPU, labelled with gpu.uuid; value 1 when the bucket is active and
	// 0 when it is not. Polling-based, so a throttle event shorter than
	// the poll interval may be missed; see CHANGELOG for the floor caveat.
	for _, r := range snap.ThrottleReadings {
		attrs := []otlpKeyValue{
			{Key: "gpu.uuid", Value: stringVal(r.UUID)},
		}
		metrics = append(metrics,
			gaugeMetric("gpu.throttle.power_active",
				"GPU clock throttling for power reasons (1=active)", "1",
				nowNano, boolToFloat(r.PowerActive), attrs),
			gaugeMetric("gpu.throttle.thermal_active",
				"GPU clock throttling for thermal reasons (1=active)", "1",
				nowNano, boolToFloat(r.ThermalActive), attrs),
			gaugeMetric("gpu.throttle.sw_active",
				"GPU clock throttling for software-imposed reasons (1=active)", "1",
				nowNano, boolToFloat(r.SWActive), attrs),
			gaugeMetric("gpu.throttle.hw_active",
				"GPU clock throttling for hardware reasons, umbrella (1=active)", "1",
				nowNano, boolToFloat(r.HWActive), attrs),
		)
	}

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

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
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
