// Package export provides OTEL-compatible metric and trace export.
//
// Architecture: parallel consumers of the Stats Engine snapshot.
// OTLP and Prometheus are OPTIONAL - disabled by default, enabled via
// --otlp <endpoint> or --prometheus <addr> flags.
//
// Call chain: watch.go calls export.OTLP.Push(snap) every ExportInterval →
//   OTLP exporter serializes metrics as OTLP/HTTP JSON →
//   HTTP POST to <endpoint>/v1/metrics
//
// OTEL semantic conventions used:
//   gpu.cuda.operation.duration  - per-op latency percentiles (microseconds)
//   gpu.cuda.operation.count     - per-op event counts
//   system.cpu.utilization       - system CPU ratio (0-1)
//   system.memory.utilization    - system memory ratio (0-1)
//   ingero.anomaly.count         - anomaly event count
package export

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/contract"
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
// Zero external dependencies - uses only net/http and encoding/json.
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
	Name                 string                       `json:"name"`
	Description          string                       `json:"description,omitempty"`
	Unit                 string                       `json:"unit,omitempty"`
	Gauge                *otlpData                    `json:"gauge,omitempty"`
	Sum                  *otlpData                    `json:"sum,omitempty"`
	Histogram            *otlpHistogramData           `json:"histogram,omitempty"`
	ExponentialHistogram *otlpExponentialHistogram    `json:"exponentialHistogram,omitempty"`
}

// v0.15 item B: OTLP/JSON histogram + exponential-histogram shapes.
// References:
//   https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding
//   https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/metrics/v1/metrics.proto
//
// Counts are encoded as JSON strings per the OTLP/JSON convention
// (uint64 fields cross the JS-safe-integer boundary).
type otlpHistogramData struct {
	DataPoints             []otlpHistogramPoint `json:"dataPoints"`
	AggregationTemporality int                  `json:"aggregationTemporality,omitempty"`
}

type otlpHistogramPoint struct {
	Attributes        []otlpKeyValue `json:"attributes,omitempty"`
	StartTimeUnixNano string         `json:"startTimeUnixNano,omitempty"`
	TimeUnixNano      string         `json:"timeUnixNano"`
	Count             string         `json:"count"`
	Sum               *float64       `json:"sum,omitempty"`
	BucketCounts      []string       `json:"bucketCounts,omitempty"`
	ExplicitBounds    []float64      `json:"explicitBounds,omitempty"`
	Min               *float64       `json:"min,omitempty"`
	Max               *float64       `json:"max,omitempty"`
}

type otlpExponentialHistogram struct {
	DataPoints             []otlpExponentialHistogramPoint `json:"dataPoints"`
	AggregationTemporality int                             `json:"aggregationTemporality,omitempty"`
}

type otlpExponentialHistogramPoint struct {
	Attributes        []otlpKeyValue              `json:"attributes,omitempty"`
	StartTimeUnixNano string                      `json:"startTimeUnixNano,omitempty"`
	TimeUnixNano      string                      `json:"timeUnixNano"`
	Count             string                      `json:"count"`
	Sum               *float64                    `json:"sum,omitempty"`
	Scale             int32                       `json:"scale"`
	ZeroCount         string                      `json:"zeroCount"`
	Positive          *otlpExponentialHistoBuckets `json:"positive,omitempty"`
	Negative          *otlpExponentialHistoBuckets `json:"negative,omitempty"`
	Min               *float64                    `json:"min,omitempty"`
	Max               *float64                    `json:"max,omitempty"`
}

type otlpExponentialHistoBuckets struct {
	Offset       int32    `json:"offset"`
	BucketCounts []string `json:"bucketCounts,omitempty"`
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
				gaugeMetricInt(contract.MetricGPUNCCLProcessLoaded,
					"NCCL-loaded process discovered on this node (1=present)",
					"1", nowNano, 1, attrs),
			)
		}
		metrics = append(metrics,
			gaugeMetricInt(contract.MetricGPUNCCLProcessesTotal,
				"Count of NCCL-loaded processes on this node", "1",
				nowNano, int64(len(snap.NCCLProcessReadings)), nil),
		)
	}

	// NVML-poll memfrag heuristic. Polling-based; the gauge is a
	// coarse heuristic over (used, free, total). Four gauges per
	// GPU labelled with gpu.uuid.
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
			gaugeMetric(contract.MetricGPUMemoryFragmentation,
				"Coarse GPU memory fragmentation heuristic from NVML poll [0,1]",
				"1", nowNano, r.FragmentationEstimate, attrs),
		)
	}
	for _, p := range snap.MemFragProcessReadings {
		attrs := []otlpKeyValue{
			{Key: "gpu.uuid", Value: stringVal(p.UUID)},
			{Key: "pid", Value: otlpValue{IntValue: int64Ptr(int64(p.PID))}},
		}
		metrics = append(metrics,
			gaugeMetricInt(contract.MetricGPUMemoryProcessAllocated,
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
			sumMetric(contract.MetricGPUMemcpyBytesTotal,
				"Cumulative CUDA memcpy bytes by direction (v0.14 item C)",
				"By", nowNano, m.BytesTotal, attrs),
			histogramMetric(contract.MetricGPUMemcpyDurationMS,
				"Per-event CUDA memcpy duration by direction (v0.15 item C; replaces v0.14 per-window-average gauge)",
				"ms", nowNano, "", m.DurationHistogram, attrs),
		)
	}

	// v0.16.3 inference exporter surface. Three groups of metrics:
	//
	//   1. Per-workload baseline shape: histogram, mean, p95.
	//      Attributes: cgroup_path_hash, pid, stream_handle, phase.
	//   2. Engine-level cumulative counters: outlier total per bucket,
	//      throttle-at-outlier per bucket, workloads tracked.
	//   3. Sampler observability: degraded gauge, degradations total,
	//      cause label.
	//
	// All emit only when the snapshot carries non-empty inference data
	// (i.e. --inference is engaged AND the engine has at least one
	// warmed workload). Pre-v0.16.3 collectors that don't recognize
	// these names ignore them; consumers that do receive a complete
	// view of the agent's inference baseline state.
	for _, w := range snap.InferWorkloads {
		attrs := []otlpKeyValue{
			{Key: contract.AttrCgroupPathHash, Value: stringVal(w.CGroupHash)},
			{Key: "pid", Value: otlpValue{IntValue: int64Ptr(int64(w.PID))}},
			{Key: contract.AttrInferStreamHandle, Value: stringVal(strconv.FormatUint(w.StreamHandle, 10))},
			{Key: contract.AttrInferPhase, Value: stringVal(w.Phase)},
		}
		metrics = append(metrics,
			histogramMetric(contract.MetricInferStepDurationNs,
				"Per-workload inference step duration distribution (cumulative)",
				"ns", nowNano, "", w.Histogram, attrs),
			gaugeMetric(contract.MetricInferBaselineMeanNs,
				"Per-workload inference step EMA mean", "ns",
				nowNano, w.MeanNs, attrs),
			gaugeMetric(contract.MetricInferBaselineP95Ns,
				"Per-workload inference step P² p95 estimate", "ns",
				nowNano, w.P95Ns, attrs),
		)
	}
	if len(snap.InferWorkloads) > 0 || snap.InferStats.WorkloadsTracked > 0 ||
		len(snap.InferStats.OutliersTotal) > 0 || snap.InferSampler.DegradationsTotal > 0 {
		metrics = append(metrics, gaugeMetricInt(
			contract.MetricInferWorkloadsTracked,
			"Distinct (cgroup,pid,stream,phase) workloads currently tracked by the infer engine",
			"1", nowNano, int64(snap.InferStats.WorkloadsTracked), nil))
	}
	for bucket, count := range snap.InferStats.OutliersTotal {
		attrs := []otlpKeyValue{
			{Key: contract.AttrInferOutlierBucket, Value: stringVal(bucket)},
		}
		metrics = append(metrics, sumMetric(
			contract.MetricInferOutlierTotal,
			"Cumulative inference step-duration outliers per bucket",
			"1", nowNano, int64(count), attrs))
	}
	for bucket, count := range snap.InferStats.ThrottleAtOutlier {
		attrs := []otlpKeyValue{
			{Key: contract.AttrInferOutlierBucket, Value: stringVal(bucket)},
		}
		metrics = append(metrics, sumMetric(
			contract.MetricInferThrottleActiveTotal,
			"Cumulative inference outliers observed while NVML throttle reasons were active",
			"1", nowNano, int64(count), attrs))
	}
	if snap.InferSampler.DegradationsTotal > 0 || snap.InferSampler.Degraded {
		samplerAttrs := []otlpKeyValue{}
		if snap.InferSampler.LastCause != "" {
			samplerAttrs = append(samplerAttrs, otlpKeyValue{
				Key: contract.AttrInferSamplerCause, Value: stringVal(snap.InferSampler.LastCause),
			})
		}
		metrics = append(metrics,
			gaugeMetric(contract.MetricInferSamplerDegraded,
				"Inference sampler degraded state (1=admitting 100% of events)", "1",
				nowNano, boolToFloat(snap.InferSampler.Degraded), samplerAttrs),
			sumMetric(contract.MetricInferSamplerDegradationsTotal,
				"Cumulative inference sampler flip-to-degraded transitions", "1",
				nowNano, int64(snap.InferSampler.DegradationsTotal), samplerAttrs),
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

// histogramMetric builds an OTLP Histogram metric from a frozen
// stats.HistogramSnapshot. v0.15 item B. Cumulative temporality
// (matches the Sum/Gauge convention elsewhere in this exporter).
//
// Empty snapshots (HasObservation=false) still emit a zero-count
// data point so consumers see the metric is wired even before the
// first observation.
func histogramMetric(name, desc, unit, timeNano, startNano string, snap stats.HistogramSnapshot, attrs []otlpKeyValue) otlpMetric {
	bounds := append([]float64(nil), snap.ExplicitBounds...)
	bucketStrs := make([]string, len(snap.BucketCounts))
	for i, c := range snap.BucketCounts {
		bucketStrs[i] = uintToStr(c)
	}
	pt := otlpHistogramPoint{
		Attributes:        attrs,
		StartTimeUnixNano: startNano,
		TimeUnixNano:      timeNano,
		Count:             uintToStr(snap.Count),
		BucketCounts:      bucketStrs,
		ExplicitBounds:    bounds,
	}
	if snap.Count > 0 {
		s := snap.Sum
		pt.Sum = &s
	}
	if snap.HasObservation {
		mn := snap.Min
		mx := snap.Max
		pt.Min = &mn
		pt.Max = &mx
	}
	return otlpMetric{
		Name:        name,
		Description: desc,
		Unit:        unit,
		Histogram: &otlpHistogramData{
			DataPoints:             []otlpHistogramPoint{pt},
			AggregationTemporality: 2, // cumulative
		},
	}
}

// exponentialHistogramMetric emits an OTLP ExponentialHistogram.
// v0.15 item B. Producer responsible for the scale + bucket layout;
// this function is the wire-format encoder only.
//
// Reference: opentelemetry-proto metrics v1, message ExponentialHistogram.
// Counts encoded as JSON strings per OTLP/JSON spec.
func exponentialHistogramMetric(name, desc, unit, timeNano, startNano string,
	count uint64, sum float64, scale int32, zeroCount uint64,
	posOffset int32, posCounts []uint64,
	negOffset int32, negCounts []uint64,
	min, max float64, hasMinMax bool,
	attrs []otlpKeyValue,
) otlpMetric {
	posBuckets := uintsToStrs(posCounts)
	negBuckets := uintsToStrs(negCounts)
	pt := otlpExponentialHistogramPoint{
		Attributes:        attrs,
		StartTimeUnixNano: startNano,
		TimeUnixNano:      timeNano,
		Count:             uintToStr(count),
		Scale:             scale,
		ZeroCount:         uintToStr(zeroCount),
	}
	if count > 0 {
		s := sum
		pt.Sum = &s
	}
	if len(posBuckets) > 0 {
		pt.Positive = &otlpExponentialHistoBuckets{Offset: posOffset, BucketCounts: posBuckets}
	}
	if len(negBuckets) > 0 {
		pt.Negative = &otlpExponentialHistoBuckets{Offset: negOffset, BucketCounts: negBuckets}
	}
	if hasMinMax {
		mn := min
		mx := max
		pt.Min = &mn
		pt.Max = &mx
	}
	return otlpMetric{
		Name:        name,
		Description: desc,
		Unit:        unit,
		ExponentialHistogram: &otlpExponentialHistogram{
			DataPoints:             []otlpExponentialHistogramPoint{pt},
			AggregationTemporality: 2, // cumulative
		},
	}
}

func uintToStr(u uint64) string {
	return strconv.FormatUint(u, 10)
}

func uintsToStrs(xs []uint64) []string {
	out := make([]string, len(xs))
	for i, x := range xs {
		out[i] = uintToStr(x)
	}
	return out
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
