#!/usr/bin/env bash
# vllm_logprobs_amplification.sh - Reproduce vLLM #37343
#
# n_completions=8 + logprobs=20 causes one request to block all others
# for 9+ seconds. Each decode step expands to 8 sequences x full-vocab
# softmax (150K tokens), starving co-scheduled requests.
#
# Prerequisites:
#   pip install vllm==0.17.1
#   python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen2.5-0.5B-Instruct --port 8000 \
#     --gpu-memory-utilization 0.95 --max-model-len 32768 \
#     --enable-prefix-caching
#
# Usage:
#   # Run 3 rounds (default)
#   bash tests/workloads/pathological/vllm_logprobs_amplification.sh
#
#   # Run N rounds
#   bash tests/workloads/pathological/vllm_logprobs_amplification.sh 5
#
# Reference: https://github.com/vllm-project/vllm/issues/37343

set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
ROUNDS="${1:-3}"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# Shared prefix (32 tokens) - all requests start with this
SHARED_PREFIX="You are a helpful AI assistant. Please analyze the following technical document about GPU performance optimization and provide a detailed summary of the key findings."

echo "=== vLLM #37343 Reproduction: logprobs amplification ==="
echo "Server: $VLLM_URL"
echo "Model:  $MODEL"
echo "Rounds: $ROUNDS"
echo ""

# Check server is up
if ! curl -sf "$VLLM_URL/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server not responding at $VLLM_URL"
    echo "Start it with:"
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model $MODEL --port 8000 \\"
    echo "    --gpu-memory-utilization 0.95 --max-model-len 32768 \\"
    echo "    --enable-prefix-caching"
    exit 1
fi

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

for round in $(seq 1 "$ROUNDS"); do
    echo "--- Round $round/$ROUNDS ---"

    # r9: plain victim request (arrives first, should be fast)
    curl -sf "$VLLM_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$SHARED_PREFIX What is GPU memory?\"}],
            \"max_tokens\": 64
        }" \
        -w "\nr9: %{time_total}s (TTFT victim)\n" \
        -o "$TMPDIR/r9_${round}.json" &
    PID_R9=$!

    # Small delay so victim arrives first
    sleep 0.05

    # r06: the amplifier - n=8, logprobs=20 (this blocks everything)
    curl -sf "$VLLM_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$SHARED_PREFIX Explain the concept of CUDA kernel launches and how they relate to GPU utilization in deep learning training.\"}],
            \"max_tokens\": 256,
            \"n\": 8,
            \"logprobs\": true,
            \"top_logprobs\": 20
        }" \
        -w "\nr06: %{time_total}s (amplifier n=8 logprobs=20)\n" \
        -o "$TMPDIR/r06_${round}.json" &
    PID_R06=$!

    # r01, r03, r08: other requests with logprobs (moderate load)
    curl -sf "$VLLM_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$SHARED_PREFIX What causes GPU underutilization?\"}],
            \"max_tokens\": 128,
            \"logprobs\": true,
            \"top_logprobs\": 5
        }" \
        -w "\nr01: %{time_total}s\n" \
        -o "$TMPDIR/r01_${round}.json" &

    curl -sf "$VLLM_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$SHARED_PREFIX How does prefix caching work in vLLM?\"}],
            \"max_tokens\": 128,
            \"logprobs\": true,
            \"top_logprobs\": 5
        }" \
        -w "\nr03: %{time_total}s\n" \
        -o "$TMPDIR/r03_${round}.json" &

    curl -sf "$VLLM_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$SHARED_PREFIX What is the difference between prefill and decode phases?\"}],
            \"max_tokens\": 128,
            \"logprobs\": true,
            \"top_logprobs\": 5
        }" \
        -w "\nr08: %{time_total}s\n" \
        -o "$TMPDIR/r08_${round}.json" &

    # Wait for all requests
    wait
    echo ""
done

echo "=== Done ==="
echo "Round 1 is typically the worst (cold cache)."
echo "Look for r06 taking 5-10+ seconds while r9 should be <200ms."
