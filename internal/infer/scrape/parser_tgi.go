package scrape

import "strings"

// tgiMetricMap maps TGI Prometheus metric base names to OTel GenAI
// semconv canonical names. Source:
// https://huggingface.co/docs/text-generation-inference/reference/metrics
// (accessed 2026-05).
//
// TGI does not expose a dedicated TTFT metric — TTFT is composed
// from queue + validation + first-decode latencies. v0.16.2 maps
// the closest equivalent (request duration) and flags this gap in
// the docs.
var tgiMetricMap = map[string]string{
	"tgi_request_duration":             "gen_ai.client.operation.duration",
	"tgi_request_mean_time_per_token_duration": "gen_ai.server.time_per_output_token",
	"tgi_batch_forward_duration":       "gen_ai.server.request.duration.batch",
	"tgi_request_queue_duration":       "gen_ai.server.queue.duration",
	"tgi_request_validation_duration":  "gen_ai.server.validation.duration",
	"tgi_request_input_length":         "gen_ai.client.token.usage.input",
	"tgi_request_generated_tokens":     "gen_ai.client.token.usage.output",

	// Gauges
	"tgi_batch_current_size":           "gen_ai.server.batch.size",
	"tgi_queue_size":                   "gen_ai.server.queue.size",
}

// TGIParser implements Parser for TGI /metrics output.
type TGIParser struct{}

func (TGIParser) Parse(body []byte) ([]ScrapedSample, error) {
	lines := parsePromBody(body)
	out := make([]ScrapedSample, 0, len(lines))
	for _, pl := range lines {
		base, canonical, kind, ok := mapTGILine(pl.Name)
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

func mapTGILine(name string) (string, string, SampleKind, bool) {
	for _, suf := range []string{"_bucket", "_sum", "_count"} {
		if strings.HasSuffix(name, suf) {
			base := strings.TrimSuffix(name, suf)
			if canon, ok := tgiMetricMap[base]; ok {
				return base, canon, SampleHistogram, true
			}
			return "", "", 0, false
		}
	}
	if canon, ok := tgiMetricMap[name]; ok {
		if strings.HasSuffix(name, "_total") {
			return name, canon, SampleCounter, true
		}
		return name, canon, SampleGauge, true
	}
	return "", "", 0, false
}
