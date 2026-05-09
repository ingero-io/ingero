// Package scrape pulls Prometheus-format /metrics endpoints from
// auto-detected inference engines (vLLM, TGI, SGLang, Triton) and
// translates the engine-specific metric names into OTel GenAI
// semantic-convention names. The OTel-named output is what the
// agent's OTLP exporter actually emits, so any downstream consumer
// (Datadog, Grafana, Honeycomb, Arize) sees a unified TTFT/TPOT
// surface regardless of which engine is running locally.
package scrape

import (
	"math"
	"strconv"
	"strings"
)

// SampleKind classifies a parsed metric line by the OTel metric type
// that should carry it. v0.16.2 emits everything as Histogram
// observations or Gauge values; the OTLP exporter (in
// internal/export) handles the temporality.
type SampleKind int

const (
	SampleHistogram SampleKind = iota // _bucket / _sum / _count triple
	SampleGauge                       // single value
	SampleCounter                     // monotonic counter; agent emits as cumulative Sum
)

// ScrapedSample is one normalized metric observation pulled from an
// engine's /metrics endpoint and mapped to OTel GenAI semconv.
//
// CanonicalName is the semconv metric name (e.g.
// "gen_ai.client.operation.time_to_first_token"). EngineName is the
// raw Prometheus name from the engine ("vllm:time_to_first_token_seconds")
// — preserved for trace logs and for parser-fixture tests.
//
// Histograms unfold into multiple ScrapedSample rows: one per
// bucket (with kind=SampleHistogram and Bucket le=N), plus _sum
// and _count rows. The OTLP exporter re-stitches them.
type ScrapedSample struct {
	CanonicalName string
	EngineName    string
	Kind          SampleKind
	Value         float64
	Labels        map[string]string
	// Bucket is set for SampleHistogram bucket rows. The "le" upper
	// bound (or +Inf as math.Inf(1)). Zero for non-bucket rows.
	Bucket float64
	IsSum  bool // true for the _sum row of a histogram
	IsCount bool // true for the _count row of a histogram
}

// Parser converts an engine's Prometheus exposition body into a
// slice of OTel-mapped ScrapedSample rows.
type Parser interface {
	Parse(body []byte) ([]ScrapedSample, error)
}

// promLine is one parsed Prometheus exposition row. We keep the
// parser minimal — full Prometheus exposition parsers exist as
// separate Go modules but pulling one in for this single use case
// is over-spec. The format is well-defined (RFC-style) and our
// parser handles the subset every engine emits.
type promLine struct {
	Name   string
	Labels map[string]string
	Value  float64
}

// parsePromBody splits Prometheus exposition format into ordered
// (name, labels, value) tuples. Comment lines (#) and empty lines
// are skipped. Returns nil on first parse error after the line
// number, but the caller treats partial parse as best-effort.
func parsePromBody(body []byte) []promLine {
	out := make([]promLine, 0, 64)
	for _, raw := range strings.Split(string(body), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// Format: NAME[{LABELS}] VALUE [TIMESTAMP]
		// Find labels block.
		var name, labelStr, rest string
		if i := strings.IndexByte(line, '{'); i >= 0 {
			j := strings.IndexByte(line[i:], '}')
			if j < 0 {
				continue // malformed
			}
			name = strings.TrimSpace(line[:i])
			labelStr = line[i+1 : i+j]
			rest = strings.TrimSpace(line[i+j+1:])
		} else {
			fields := strings.Fields(line)
			if len(fields) < 2 {
				continue
			}
			name = fields[0]
			rest = fields[1]
		}
		// Parse value.
		valStr := rest
		if sp := strings.IndexByte(rest, ' '); sp >= 0 {
			valStr = rest[:sp]
		}
		v, err := strconv.ParseFloat(valStr, 64)
		if err != nil {
			// Some engines emit "+Inf" / "-Inf" — handle these.
			switch valStr {
			case "+Inf", "Inf":
				v = posInf
			case "-Inf":
				v = negInf
			default:
				continue
			}
		}
		// NaN parses without error but is not a useful sample;
		// drop it so the OTLP exporter doesn't see garbage.
		if math.IsNaN(v) {
			continue
		}
		labels := parseLabels(labelStr)
		out = append(out, promLine{Name: name, Labels: labels, Value: v})
	}
	return out
}

func parseLabels(s string) map[string]string {
	if s == "" {
		return nil
	}
	out := make(map[string]string, 4)
	// Labels are k="v" comma-separated. Values may contain commas
	// in escaped form (\,), but engines we cover don't do that —
	// we use a simple state machine that handles only the common
	// quoted-string case.
	i := 0
	for i < len(s) {
		// skip whitespace + commas
		for i < len(s) && (s[i] == ' ' || s[i] == ',') {
			i++
		}
		// key
		eq := strings.IndexByte(s[i:], '=')
		if eq < 0 {
			break
		}
		key := strings.TrimSpace(s[i : i+eq])
		i += eq + 1
		if i >= len(s) || s[i] != '"' {
			break
		}
		i++ // open quote
		end := strings.IndexByte(s[i:], '"')
		if end < 0 {
			break
		}
		val := s[i : i+end]
		i += end + 1 // close quote
		out[key] = val
	}
	return out
}

// histogramSampleFromLine converts a Prometheus histogram bucket /
// sum / count line into a ScrapedSample. base is the histogram
// metric name (without the _bucket / _sum / _count suffix) and
// canonical is the OTel GenAI semconv-mapped name (also without
// suffix). The caller has already verified that pl.Name carries
// the expected suffix.
func histogramSampleFromLine(pl promLine, base, canonical string) ScrapedSample {
	s := ScrapedSample{
		CanonicalName: canonical,
		EngineName:    base,
		Kind:          SampleHistogram,
		Value:         pl.Value,
		Labels:        copyMapWithoutLE(pl.Labels),
	}
	switch {
	case strings.HasSuffix(pl.Name, "_bucket"):
		if leStr, ok := pl.Labels["le"]; ok {
			switch leStr {
			case "+Inf", "Inf":
				s.Bucket = posInf
			default:
				if le, err := strconv.ParseFloat(leStr, 64); err == nil {
					s.Bucket = le
				}
			}
		}
	case strings.HasSuffix(pl.Name, "_sum"):
		s.IsSum = true
	case strings.HasSuffix(pl.Name, "_count"):
		s.IsCount = true
	}
	return s
}

func copyMapWithoutLE(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		if k == "le" {
			continue
		}
		out[k] = v
	}
	return out
}

// posInf / negInf live as package-level vars so we can reuse them
// without re-importing math at every site.
var (
	posInf = parseInfHelper("+Inf")
	negInf = parseInfHelper("-Inf")
)

func parseInfHelper(s string) float64 {
	v, _ := strconv.ParseFloat(s, 64)
	return v
}
