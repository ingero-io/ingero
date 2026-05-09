package scrape

import (
	"strings"
	"testing"
)

const vllmFixture = `
# HELP vllm:time_to_first_token_seconds Time to first output token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="llama-3-7b",le="0.05"} 12
vllm:time_to_first_token_seconds_bucket{model_name="llama-3-7b",le="0.1"} 47
vllm:time_to_first_token_seconds_bucket{model_name="llama-3-7b",le="0.5"} 215
vllm:time_to_first_token_seconds_bucket{model_name="llama-3-7b",le="+Inf"} 230
vllm:time_to_first_token_seconds_sum{model_name="llama-3-7b"} 38.4
vllm:time_to_first_token_seconds_count{model_name="llama-3-7b"} 230

# HELP vllm:inter_token_latency_seconds Per-output-token latency.
# TYPE vllm:inter_token_latency_seconds histogram
vllm:inter_token_latency_seconds_bucket{model_name="llama-3-7b",le="0.01"} 1500
vllm:inter_token_latency_seconds_bucket{model_name="llama-3-7b",le="+Inf"} 4200
vllm:inter_token_latency_seconds_sum{model_name="llama-3-7b"} 67.2
vllm:inter_token_latency_seconds_count{model_name="llama-3-7b"} 4200

# HELP vllm:prompt_tokens_total Total prompt tokens.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="llama-3-7b"} 124000

# HELP vllm:generation_tokens_total Total generation tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="llama-3-7b"} 89000

# An unknown metric we should silently drop.
vllm:something_we_dont_map 42
`

func TestVLLMParser_HappyPath(t *testing.T) {
	samples, err := VLLMParser{}.Parse([]byte(vllmFixture))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(samples) == 0 {
		t.Fatal("expected samples, got none")
	}

	var ttftBuckets, ttftSum, ttftCount, itlSum int
	var promptTokens, genTokens float64
	for _, s := range samples {
		switch {
		case s.CanonicalName == "gen_ai.client.operation.time_to_first_token" && s.Kind == SampleHistogram:
			switch {
			case s.IsSum:
				ttftSum++
				if s.Value != 38.4 {
					t.Errorf("ttft sum value = %v, want 38.4", s.Value)
				}
			case s.IsCount:
				ttftCount++
			default:
				ttftBuckets++
			}
		case s.CanonicalName == "gen_ai.server.time_per_output_token" && s.IsSum:
			itlSum++
		case s.CanonicalName == "gen_ai.client.token.usage.input" && s.Kind == SampleCounter:
			promptTokens = s.Value
		case s.CanonicalName == "gen_ai.client.token.usage.output" && s.Kind == SampleCounter:
			genTokens = s.Value
		}
	}
	if ttftBuckets != 4 {
		t.Errorf("ttft buckets = %d, want 4", ttftBuckets)
	}
	if ttftSum != 1 || ttftCount != 1 {
		t.Errorf("ttft sum/count = %d/%d, want 1/1", ttftSum, ttftCount)
	}
	if itlSum != 1 {
		t.Errorf("itl sum = %d, want 1", itlSum)
	}
	if promptTokens != 124000 {
		t.Errorf("prompt_tokens = %v, want 124000", promptTokens)
	}
	if genTokens != 89000 {
		t.Errorf("generation_tokens = %v, want 89000", genTokens)
	}
}

func TestVLLMParser_UnknownMetricSkipped(t *testing.T) {
	samples, _ := VLLMParser{}.Parse([]byte(vllmFixture))
	for _, s := range samples {
		if strings.Contains(s.EngineName, "something_we_dont_map") {
			t.Errorf("unmapped metric leaked into output: %+v", s)
		}
	}
}

const tgiFixture = `
# HELP tgi_request_duration Total time per request.
# TYPE tgi_request_duration histogram
tgi_request_duration_bucket{le="0.5"} 100
tgi_request_duration_bucket{le="1.0"} 350
tgi_request_duration_bucket{le="+Inf"} 400
tgi_request_duration_sum 320.5
tgi_request_duration_count 400

# HELP tgi_queue_size Current queue size.
# TYPE tgi_queue_size gauge
tgi_queue_size 12
`

func TestTGIParser_HappyPath(t *testing.T) {
	samples, err := TGIParser{}.Parse([]byte(tgiFixture))
	if err != nil {
		t.Fatal(err)
	}
	var rd, qs int
	for _, s := range samples {
		if s.CanonicalName == "gen_ai.client.operation.duration" && s.Kind == SampleHistogram {
			rd++
		}
		if s.CanonicalName == "gen_ai.server.queue.size" && s.Kind == SampleGauge {
			qs++
			if s.Value != 12 {
				t.Errorf("queue_size value = %v, want 12", s.Value)
			}
		}
	}
	if rd < 5 {
		t.Errorf("expected request_duration histogram rows >= 5 (3 buckets+sum+count), got %d", rd)
	}
	if qs != 1 {
		t.Errorf("queue_size gauge rows = %d, want 1", qs)
	}
}

const sglangFixture = `
# TYPE sglang_request_latency histogram
sglang_request_latency_bucket{phase="prefill",le="0.5"} 50
sglang_request_latency_bucket{phase="prefill",le="+Inf"} 80
sglang_request_latency_sum{phase="prefill"} 18.5
sglang_request_latency_count{phase="prefill"} 80
sglang_request_latency_bucket{phase="decode",le="0.01"} 1200
sglang_request_latency_bucket{phase="decode",le="+Inf"} 4500
sglang_request_latency_sum{phase="decode"} 9.2
sglang_request_latency_count{phase="decode"} 4500

# TYPE sglang_accepted_draft_tokens_total counter
sglang_accepted_draft_tokens_total{model="qwen2-72b"} 12000
`

func TestSGLangParser_PreservesPhaseLabel(t *testing.T) {
	samples, err := SGLangParser{}.Parse([]byte(sglangFixture))
	if err != nil {
		t.Fatal(err)
	}
	prefill, decode := 0, 0
	for _, s := range samples {
		if s.CanonicalName != "gen_ai.client.operation.duration" {
			continue
		}
		switch s.Labels["phase"] {
		case "prefill":
			prefill++
		case "decode":
			decode++
		}
	}
	if prefill == 0 || decode == 0 {
		t.Errorf("expected per-phase rows: prefill=%d decode=%d", prefill, decode)
	}
}

func TestSGLangParser_SpeculativeAcceptedTotal(t *testing.T) {
	samples, _ := SGLangParser{}.Parse([]byte(sglangFixture))
	found := false
	for _, s := range samples {
		if s.CanonicalName == "gen_ai.client.speculative.accepted_tokens" && s.Kind == SampleCounter {
			found = true
			if s.Value != 12000 {
				t.Errorf("accepted_tokens value = %v, want 12000", s.Value)
			}
		}
	}
	if !found {
		t.Error("speculative accepted_tokens not extracted")
	}
}

const tritonFixture = `
# TYPE nv_inference_request_duration_us histogram
nv_inference_request_duration_us_bucket{model="llama",le="100000"} 50
nv_inference_request_duration_us_bucket{model="llama",le="+Inf"} 75
nv_inference_request_duration_us_sum{model="llama"} 6500000
nv_inference_request_duration_us_count{model="llama"} 75

# TYPE nv_inference_request_success counter
nv_inference_request_success{model="llama"} 75

# TYPE nv_inference_request_failure counter
nv_inference_request_failure{model="llama"} 2
`

func TestTritonParser_SuccessFailureCounters(t *testing.T) {
	samples, err := TritonParser{}.Parse([]byte(tritonFixture))
	if err != nil {
		t.Fatal(err)
	}
	successFound, failureFound := false, false
	for _, s := range samples {
		if s.CanonicalName == "gen_ai.client.operation.success_total" && s.Kind == SampleCounter {
			successFound = true
		}
		if s.CanonicalName == "gen_ai.client.operation.failure_total" && s.Kind == SampleCounter {
			failureFound = true
		}
	}
	if !successFound || !failureFound {
		t.Errorf("counters: success=%v failure=%v", successFound, failureFound)
	}
}

func TestParsePromBody_HandlesInfBuckets(t *testing.T) {
	body := []byte(`metric_bucket{le="+Inf"} 100
metric_bucket{le="-Inf"} 0
metric_value NaN
`)
	got := parsePromBody(body)
	// +Inf and -Inf as label values still parse normally (value is
	// in the bucket count, not the label). The NaN row has NaN as
	// the actual value and is dropped.
	if len(got) != 2 {
		t.Errorf("len = %d, want 2 (NaN value dropped)", len(got))
	}
}

func TestParseLabels_QuotedValues(t *testing.T) {
	got := parseLabels(`model_name="llama-3-7b", phase="decode"`)
	if got["model_name"] != "llama-3-7b" {
		t.Errorf("model_name = %q, want llama-3-7b", got["model_name"])
	}
	if got["phase"] != "decode" {
		t.Errorf("phase = %q, want decode", got["phase"])
	}
}

func TestParseLabels_Empty(t *testing.T) {
	if got := parseLabels(""); got != nil {
		t.Errorf("empty labels should return nil, got %+v", got)
	}
}
