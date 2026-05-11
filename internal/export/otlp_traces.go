package export

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/internal/version"
	"github.com/ingero-io/ingero/pkg/contract"
	"github.com/ingero-io/ingero/pkg/events"
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
//
// ctx is honored for both the request and the body read so a shutdown
// signal mid-push aborts the HTTP round trip instead of waiting for
// the client's own 10s timeout.
func (e *OTLPExporter) PushSpans(ctx context.Context, spans []stats.OutlierSpan) error {
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

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
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
	// Drain the body before close so net/http can return the underlying
	// connection to the keepalive pool. Without this, every span push
	// tears down its TCP+TLS state and the next push pays a full
	// handshake (50-200ms instead of ~5ms warm-pool).
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

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
		{Key: "service.version", Value: stringVal(version.Version())},
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
				Scope: otlpScope{Name: "ingero", Version: version.Version()},
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
	traceID, spanID := newTraceAndSpanIDs(s.PID, s.StreamHandle, s.StepStart.UnixNano())

	attrs := []otlpKeyValue{
		{Key: contract.AttrCgroupPathHash, Value: stringVal(s.CGroupHash)},
		{Key: contract.AttrPID, Value: otlpValue{IntValue: int64Ptr(int64(s.PID))}},
		{Key: contract.AttrInferStreamHandle, Value: stringVal(strconv.FormatUint(s.StreamHandle, 10))},
		{Key: contract.AttrInferPhase, Value: stringVal(s.Phase)},
		{Key: contract.AttrInferOutlierBucket, Value: stringVal(s.Bucket)},
		{Key: contract.AttrInferStepDurationNs, Value: otlpValue{IntValue: int64Ptr(s.StepDurationNs)}},
		{Key: contract.AttrInferBaselineP95Ns, Value: otlpValue{IntValue: int64Ptr(s.BaselineP95Ns)}},
		{Key: contract.AttrInferBaselineMeanNs, Value: otlpValue{IntValue: int64Ptr(s.BaselineMeanNs)}},
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
			Key:   contract.AttrInferMemfragEventsInStep,
			Value: otlpValue{IntValue: int64Ptr(int64(s.MemfragEvents))},
		})
	}
	if s.ThrottleReasons != 0 {
		// uint64 throttle bitmap; encode as decimal string so the
		// value survives the JS-safe-integer boundary that bare
		// JSON numbers don't.
		attrs = append(attrs, otlpKeyValue{
			Key:   contract.AttrInferThrottleReasons,
			Value: stringVal(strconv.FormatUint(s.ThrottleReasons, 10)),
		})
	}
	if len(s.KVCacheTopAllocAgesMs) > 0 {
		// Surface the oldest live alloc age as a single int attribute -
		// that's the dominant signal for stale-KV-cache; the full
		// distribution lives on the histogram metric.
		attrs = append(attrs, otlpKeyValue{
			Key:   contract.AttrInferKVCacheOldestAllocAgeMs,
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
// On rand.Read failure (rare on Linux; possible under sandboxes that
// deny /dev/urandom), the function falls back to a deterministic
// FNV-128a digest of the (pid, streamHandle, tsNanos) workload key
// with two distinct salt bytes for the trace and span. The historical
// fallback was an all-zeros id pair, which Tempo / OTel collectors
// dedup on - so every outlier emitted during a CSPRNG outage
// collapsed into a single downstream row. The deterministic fallback
// keeps each outlier distinguishable as long as the workload key
// moves forward in time.
func newTraceAndSpanIDs(pid uint32, streamHandle uint64, tsNanos int64) (string, string) {
	var traceBytes [16]byte
	var spanBytes [8]byte
	_, errT := rand.Read(traceBytes[:])
	_, errS := rand.Read(spanBytes[:])
	if errT != nil {
		// Salt 0x02 keeps the TraceID digest distinct from the EventID
		// digest in internal/infer (salt 0x01) and the SpanID below
		// (salt 0x03), all using the same workload key.
		traceBytes = events.DeterministicID(pid, streamHandle, tsNanos, 0x02)
	}
	if errS != nil {
		spanDigest := events.DeterministicID(pid, streamHandle, tsNanos, 0x03)
		copy(spanBytes[:], spanDigest[:8])
	}
	return hex.EncodeToString(traceBytes[:]), hex.EncodeToString(spanBytes[:])
}

