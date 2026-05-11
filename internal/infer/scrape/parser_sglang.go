package scrape

import "strings"

// sglangMetricMap maps SGLang Prometheus metric base names (post
// v0.5.4 prefix change from sglang: to sglang_) to OTel GenAI
// semconv canonical names. Source: SGLang Prometheus Metrics Guide
// (accessed 2026-05).
//
// SGLang is the only one of the four engines that exposes phase-
// labeled latencies natively (phase="prefill"|"decode" on the same
// histogram), so we map a single SGLang histogram to two OTel
// metrics by phase label at emit time.
var sglangMetricMap = map[string]string{
	"sglang_request_latency": "gen_ai.client.operation.duration",
	// SGLang exposes per-phase latency via labels on the same metric;
	// the OTLP exporter splits the emission by the "phase" label.

	// Speculative decoding accept ratio (one of the v0.16.2 use
	// cases; primary in the SGLang exposition).
	"sglang_accepted_draft_tokens_total": "gen_ai.client.speculative.accepted_tokens",
	"sglang_proposed_draft_tokens_total": "gen_ai.client.speculative.proposed_tokens",

	// Token counters
	"sglang_prompt_tokens_total":     "gen_ai.client.token.usage.input",
	"sglang_generation_tokens_total": "gen_ai.client.token.usage.output",
}

// SGLangParser implements Parser for SGLang /metrics output.
type SGLangParser struct{}

func (SGLangParser) Parse(body []byte) ([]ScrapedSample, error) {
	lines := parsePromBody(body)
	out := make([]ScrapedSample, 0, len(lines))
	for _, pl := range lines {
		base, canonical, kind, ok := mapSGLangLine(pl.Name)
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

func mapSGLangLine(name string) (string, string, SampleKind, bool) {
	for _, suf := range []string{"_bucket", "_sum", "_count"} {
		if strings.HasSuffix(name, suf) {
			base := strings.TrimSuffix(name, suf)
			if canon, ok := sglangMetricMap[base]; ok {
				return base, canon, SampleHistogram, true
			}
			return "", "", 0, false
		}
	}
	if canon, ok := sglangMetricMap[name]; ok {
		if strings.HasSuffix(name, "_total") {
			return name, canon, SampleCounter, true
		}
		return name, canon, SampleGauge, true
	}
	return "", "", 0, false
}
