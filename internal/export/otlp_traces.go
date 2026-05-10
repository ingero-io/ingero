package export

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/contract"
)

// OTLP /v1/traces wire-format types. Mirrors the metric-side types
// in otlp.go but for the spans pipeline. References:
//
//	https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding
//	https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/trace/v1/trace.proto
//
// Counts and times cross the JS-safe-integer boundary so they're
// encoded as JSON strings per OTLP/JSON convention.
type otlpTracesPayload struct {
	ResourceSpans []otlpResourceSpans `json:"resourceSpans"`
}

type otlpResourceSpans struct {
	Resource   otlpResource     `json:"resource"`
	ScopeSpans []otlpScopeSpans `json:"scopeSpans"`
}

type otlpScopeSpans struct {
	Scope otlpScope  `json:"scope"`
	Spans []otlpSpan `json:"spans"`
}

// SPAN_KIND_INTERNAL = 1. The agent's outlier spans are not network
// operations; they describe an internal observation about the
// workload, which matches "internal" semantics in the OTLP spec.
const otlpSpanKindInternal = 1

// Status code values per the OTel spec: UNSET=0, OK=1, ERROR=2.
// Outliers always set ERROR so a backend's "find error spans" filter
// surfaces every classified outlier.
const otlpStatusCodeError = 2

type otlpSpan struct {
	TraceID           string         `json:"traceId"`
	SpanID            string         `json:"spanId"`
	ParentSpanID      string         `json:"parentSpanId,omitempty"`
	Name              string         `json:"name"`
	Kind              int            `json:"kind"`
	StartTimeUnixNano string         `json:"startTimeUnixNano"`
	EndTimeUnixNano   string         `json:"endTimeUnixNano"`
	Attributes        []otlpKeyValue `json:"attributes,omitempty"`
	Status            otlpSpanStatus `json:"status"`
}

type otlpSpanStatus struct {
	Code    int    `json:"code"`
	Message string `json:"message,omitempty"`
}

// PushSpans serializes the given outlier spans as OTLP/JSON and POSTs
// to /v1/traces. Empty input returns nil (no-op). Errors are
// counted but not propagated to the caller's hot path; mirrors the
// Push() error-counting shape so a transient receiver outage doesn't
// cascade.
func (e *OTLPExporter) PushSpans(spans []stats.OutlierSpan) error {
	if e == nil || len(spans) == 0 {
		return nil
	}

	payload := e.buildSpansPayload(spans)

	body, err := json.Marshal(payload)
	if err != nil {
		e.mu.Lock()
		e.errors++
		e.mu.Unlock()
		e.debugf("OTLP traces: marshal error: %v", err)
		return fmt.Errorf("OTLP traces marshal: %w", err)
	}

	url := e.tracesURL()
	e.debugf("OTLP traces: pushing %d bytes to %s (%d spans)", len(body), url, len(spans))

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		e.mu.Lock()
		e.errors++
		e.mu.Unlock()
		return fmt.Errorf("OTLP traces request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range e.config.Headers {
		req.Header.Set(k, v)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		e.mu.Lock()
		e.errors++
		e.mu.Unlock()
		e.debugf("OTLP traces: push failed: %v", err)
		return fmt.Errorf("OTLP traces push: %w", err)
	}
	defer resp.Body.Close()

	e.mu.Lock()
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		e.pushes++
		e.mu.Unlock()
		e.debugf("OTLP traces: push OK (%d %s)", resp.StatusCode, resp.Status)
		return nil
	}
	e.errors++
	e.mu.Unlock()
	e.debugf("OTLP traces: push rejected: %d %s", resp.StatusCode, resp.Status)
	return fmt.Errorf("OTLP traces push: %d %s", resp.StatusCode, resp.Status)
}

// tracesURL builds the URL for the /v1/traces endpoint. Mirrors
// metricsURL's scheme/path handling.
func (e *OTLPExporter) tracesURL() string {
	endpoint := e.config.Endpoint
	scheme := "https"
	if e.config.Insecure {
		scheme = "http"
	}
	if strings.HasPrefix(endpoint, "http://") || strings.HasPrefix(endpoint, "https://") {
		return strings.TrimRight(endpoint, "/") + "/v1/traces"
	}
	return fmt.Sprintf("%s://%s/v1/traces", scheme, endpoint)
}

// buildSpansPayload encodes a slice of OutlierSpan into the OTLP
// trace shape. Resource attributes mirror the metric path so spans
// and metrics carry identical (node, cluster, provider) identity for
// downstream join.
func (e *OTLPExporter) buildSpansPayload(spans []stats.OutlierSpan) otlpTracesPayload {
	resourceAttrs := []otlpKeyValue{
		{Key: "service.name", Value: stringVal("ingero")},
		{Key: "service.version", Value: stringVal("0.8.0")},
	}
	if e.config.NodeID != "" {
		resourceAttrs = append(resourceAttrs, otlpKeyValue{
			Key: contract.AttrNodeID, Value: stringVal(e.config.NodeID),
		})
	}
	if e.config.ClusterID != "" {
		resourceAttrs = append(resourceAttrs, otlpKeyValue{
			Key: contract.AttrClusterID, Value: stringVal(e.config.ClusterID),
		})
	}
	if e.config.Provider != "" {
		resourceAttrs = append(resourceAttrs, otlpKeyValue{
			Key: contract.AttrProvider, Value: stringVal(e.config.Provider),
		})
	}

	out := make([]otlpSpan, 0, len(spans))
	for _, s := range spans {
		out = append(out, outlierToOTLPSpan(s))
	}

	return otlpTracesPayload{
		ResourceSpans: []otlpResourceSpans{{
			Resource: otlpResource{Attributes: resourceAttrs},
			ScopeSpans: []otlpScopeSpans{{
				Scope: otlpScope{Name: "ingero", Version: "0.8.0"},
				Spans: out,
			}},
		}},
	}
}

// outlierToOTLPSpan builds the wire-shape Span from one OutlierSpan.
// Span name "infer.outlier.{bucket}" identifies the kind of outlier
// at a glance; the workload-key attributes let an operator filter or
// group across spans without needing external metadata.
func outlierToOTLPSpan(s stats.OutlierSpan) otlpSpan {
	traceID, spanID := newTraceAndSpanIDs()

	attrs := []otlpKeyValue{
		{Key: contract.AttrCgroupPathHash, Value: stringVal(s.CGroupHash)},
		{Key: "pid", Value: otlpValue{IntValue: int64Ptr(int64(s.PID))}},
		{Key: contract.AttrInferStreamHandle, Value: stringVal(strconv.FormatUint(s.StreamHandle, 10))},
		{Key: contract.AttrInferPhase, Value: stringVal(s.Phase)},
		{Key: contract.AttrInferOutlierBucket, Value: stringVal(s.Bucket)},
		{Key: "ingero.infer.step_duration_ns", Value: otlpValue{IntValue: int64Ptr(s.StepDurationNs)}},
		{Key: "ingero.infer.baseline_p95_ns", Value: otlpValue{IntValue: int64Ptr(s.BaselineP95Ns)}},
		{Key: "ingero.infer.baseline_mean_ns", Value: otlpValue{IntValue: int64Ptr(s.BaselineMeanNs)}},
	}
	if s.EventID != "" {
		attrs = append(attrs, otlpKeyValue{
			Key: contract.AttrEventID, Value: stringVal(s.EventID),
		})
	}
	if s.KernelFingerprint != 0 {
		attrs = append(attrs, otlpKeyValue{
			Key:   contract.AttrInferKernelFingerprint,
			Value: stringVal(strconv.FormatUint(s.KernelFingerprint, 16)),
		})
	}
	if s.MemfragEvents != 0 {
		attrs = append(attrs, otlpKeyValue{
			Key:   "ingero.infer.memfrag_events_in_step",
			Value: otlpValue{IntValue: int64Ptr(int64(s.MemfragEvents))},
		})
	}
	if s.ThrottleReasons != 0 {
		// uint64 throttle bitmap; encode as decimal string so the
		// value survives the JS-safe-integer boundary that bare
		// JSON numbers don't.
		attrs = append(attrs, otlpKeyValue{
			Key:   "ingero.infer.throttle_reasons",
			Value: stringVal(strconv.FormatUint(s.ThrottleReasons, 10)),
		})
	}
	if len(s.KVCacheTopAllocAgesMs) > 0 {
		// Surface the oldest live alloc age as a single int attribute -
		// that's the dominant signal for stale-KV-cache; the full
		// distribution lives on the histogram metric.
		attrs = append(attrs, otlpKeyValue{
			Key:   "ingero.infer.kvcache.oldest_alloc_age_ms",
			Value: otlpValue{IntValue: int64Ptr(int64(s.KVCacheTopAllocAgesMs[0]))},
		})
	}
	if s.ModelName != "" {
		attrs = append(attrs, otlpKeyValue{
			Key: contract.AttrGenAIRequestModel, Value: stringVal(s.ModelName),
		})
	}
	if s.EngineSystem != "" {
		attrs = append(attrs, otlpKeyValue{
			Key: contract.AttrGenAISystem, Value: stringVal(s.EngineSystem),
		})
	}

	return otlpSpan{
		TraceID:           traceID,
		SpanID:            spanID,
		Name:              "infer.outlier." + s.Bucket,
		Kind:              otlpSpanKindInternal,
		StartTimeUnixNano: strconv.FormatInt(s.StepStart.UnixNano(), 10),
		EndTimeUnixNano:   strconv.FormatInt(s.StepEnd.UnixNano(), 10),
		Attributes:        attrs,
		Status: otlpSpanStatus{
			Code:    otlpStatusCodeError,
			Message: "step duration exceeded baseline " + s.Bucket,
		},
	}
}

// newTraceAndSpanIDs returns a fresh (32-hex traceId, 16-hex spanId)
// pair using crypto/rand. Each outlier gets a self-contained trace;
// see OutlierSpan doc comment for the design rationale.
//
// On rand.Read failure the function falls back to a fixed all-zeros
// id pair rather than failing the emit - the receiver will likely
// dedup on the all-zeros id but the span body still carries the
// useful attributes, and a flapping CSPRNG isn't a reason to drop
// outlier visibility.
func newTraceAndSpanIDs() (string, string) {
	var traceBytes [16]byte
	var spanBytes [8]byte
	_, _ = rand.Read(traceBytes[:])
	_, _ = rand.Read(spanBytes[:])
	return hex.EncodeToString(traceBytes[:]), hex.EncodeToString(spanBytes[:])
}

