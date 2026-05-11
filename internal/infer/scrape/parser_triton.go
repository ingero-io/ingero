package scrape

import "strings"

// tritonMetricMap maps NVIDIA Triton Inference Server Prometheus
// metric names to OTel GenAI semconv. Source: Triton metrics docs
// (accessed 2026-05). Triton's metric set is generic (per-model
// histograms, queue depth, success/failure counters); we map the
// LLM-relevant subset and ignore vision/embedding-only metrics.
var tritonMetricMap = map[string]string{
	"nv_inference_request_duration_us": "gen_ai.client.operation.duration",
	"nv_inference_queue_duration_us":   "gen_ai.server.queue.duration",
	"nv_inference_compute_input_duration_us":  "gen_ai.server.request.duration.prefill",
	"nv_inference_compute_output_duration_us": "gen_ai.server.request.duration.decode",
	"nv_inference_request_success":     "gen_ai.client.operation.success_total",
	"nv_inference_request_failure":     "gen_ai.client.operation.failure_total",
}

// TritonParser implements Parser for Triton's /metrics output.
type TritonParser struct{}

func (TritonParser) Parse(body []byte) ([]ScrapedSample, error) {
	lines := parsePromBody(body)
	out := make([]ScrapedSample, 0, len(lines))
	for _, pl := range lines {
		base, canonical, kind, ok := mapTritonLine(pl.Name)
		if !ok {
			continue
		}
		switch kind {
		case SampleHistogram:
			out = append(out, histogramSampleFromLine(pl, base, canonical))
		default:
			out = append(out, ScrapedSample{
				CanonicalName: canonical,
				EngineName:    base,
				Kind:          kind,
				Value:         pl.Value,
				Labels:        pl.Labels,
			})
		}
	}
	return out, nil
}

func mapTritonLine(name string) (string, string, SampleKind, bool) {
	for _, suf := range []string{"_bucket", "_sum", "_count"} {
		if strings.HasSuffix(name, suf) {
			base := strings.TrimSuffix(name, suf)
			if canon, ok := tritonMetricMap[base]; ok {
				return base, canon, SampleHistogram, true
			}
			return "", "", 0, false
		}
	}
	if canon, ok := tritonMetricMap[name]; ok {
		// Triton counters end in _success / _failure (not _total);
		// treat them as cumulative monotonic Sums regardless.
		if strings.Contains(name, "request_success") || strings.Contains(name, "request_failure") {
			return name, canon, SampleCounter, true
		}
		return name, canon, SampleGauge, true
	}
	return "", "", 0, false
}
