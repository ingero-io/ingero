package scrape

import "strings"

// vllmMetricMap maps vLLM Prometheus metric base names (without
// _bucket/_sum/_count suffix) to OTel GenAI semconv canonical names.
// Source: https://docs.vllm.ai/en/stable/design/metrics/ (accessed
// 2026-05). NIM passes vLLM metrics through unchanged so the same
// map covers NIM deployments.
var vllmMetricMap = map[string]string{
	// Histograms
	"vllm:time_to_first_token_seconds":  "gen_ai.client.operation.time_to_first_token",
	"vllm:inter_token_latency_seconds":  "gen_ai.server.time_per_output_token",
	"vllm:e2e_request_latency_seconds":  "gen_ai.client.operation.duration",
	"vllm:request_prefill_time_seconds": "gen_ai.server.request.duration.prefill",
	"vllm:request_decode_time_seconds":  "gen_ai.server.request.duration.decode",

	// Counters / gauges
	"vllm:prompt_tokens_total":     "gen_ai.client.token.usage.input",
	"vllm:generation_tokens_total": "gen_ai.client.token.usage.output",
}

// VLLMParser implements Parser for vLLM /metrics output.
type VLLMParser struct{}

// Parse converts vLLM Prometheus exposition into OTel GenAI
// ScrapedSample rows. Unknown metric names are silently dropped —
// we only emit what the canonical map covers, keeping the OTel
// surface intentionally narrow (TTFT, TPOT, prefill/decode latency,
// token counts).
func (VLLMParser) Parse(body []byte) ([]ScrapedSample, error) {
	lines := parsePromBody(body)
	out := make([]ScrapedSample, 0, len(lines))
	for _, pl := range lines {
		base, canonical, kind, ok := mapVLLMLine(pl.Name)
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

// mapVLLMLine returns (baseName, canonicalName, sampleKind, ok). For
// histogram lines it strips the _bucket/_sum/_count suffix before
// looking up the map.
func mapVLLMLine(name string) (string, string, SampleKind, bool) {
	// Histogram-suffix forms.
	for _, suf := range []string{"_bucket", "_sum", "_count"} {
		if strings.HasSuffix(name, suf) {
			base := strings.TrimSuffix(name, suf)
			if canon, ok := vllmMetricMap[base]; ok {
				return base, canon, SampleHistogram, true
			}
			return "", "", 0, false
		}
	}
	// Non-histogram (counter / gauge).
	if canon, ok := vllmMetricMap[name]; ok {
		// vLLM emits *_total for counters; treat as cumulative
		// monotonic Sum. Everything else is a gauge.
		if strings.HasSuffix(name, "_total") {
			return name, canon, SampleCounter, true
		}
		return name, canon, SampleGauge, true
	}
	return "", "", 0, false
}
