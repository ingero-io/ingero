# GPU Investigation Trace Data

Real-world GPU performance investigations traced with Ingero. Each `.db` file is a SQLite database containing CUDA API call timings, host kernel events, and causal chain analysis.

## Databases

| File | Issue | What It Shows |
|------|-------|---------------|
| `pytorch-dataloader-starvation.db` | [pytorch/pytorch#154318](https://github.com/pytorch/pytorch/issues/154318) | PyTorch DataLoader 114x slower than direct indexing. 200K+ context switches, GPU starving for data. |
| `vllm-37343-logprobs-amplification.db` | [vllm-project/vllm#37343](https://github.com/vllm-project/vllm/issues/37343) | vLLM n_completions + logprobs blocks all co-scheduled requests for 11+ seconds. 80% kernel throughput drop. |
| `vllm-37308-hol-blocking.db` | [vllm-project/vllm#37308](https://github.com/vllm-project/vllm/issues/37308) | vLLM head-of-line blocking with prefix caching. 14.5x TTFT regression. Available as [release asset](https://github.com/ingero-io/ingero/releases). |

## Explore with Ingero MCP

Connect any MCP-compatible AI assistant (Claude, Cursor, OpenClaw) to investigate these traces:

```bash
# Start MCP server with a trace database
ingero mcp --db investigations/pytorch-dataloader-starvation.db

# Or via HTTPS for remote access
ingero mcp --db investigations/vllm-37343-logprobs-amplification.db --http :8080
```

Then ask your AI assistant questions like:
- "What are the causal chains in this trace?"
- "Show me the CUDA API latency breakdown per process"
- "Which process had the most context switches?"
- "Run SQL: SELECT op, COUNT(*), AVG(duration_ns)/1000 as avg_us FROM events GROUP BY op ORDER BY avg_us DESC"

## Quick Analysis (no MCP needed)

```bash
# View causal chains
ingero explain --db investigations/pytorch-dataloader-starvation.db --since 5m

# Per-process GPU API breakdown
ingero explain --db investigations/pytorch-dataloader-starvation.db --per-process --since 5m

# Query raw events
ingero query --db investigations/pytorch-dataloader-starvation.db --since 5m --op cudaMemcpyAsync
```

## Environment

All traces captured on TensorDock RTX 4090 (24GB), Ubuntu 22.04, kernel 5.15, NVIDIA driver 570.211.01.
- PyTorch investigation: PyTorch 2.10.0+cu128
- vLLM investigations: vLLM 0.17.1, Qwen/Qwen2.5-0.5B-Instruct with prefix caching
