# GPU Investigation Trace Data

Real-world GPU performance investigations traced with Ingero. Each `.db` file is a SQLite database containing CUDA API call timings, host kernel events, and causal chain analysis.

## Databases

| File | Issue | What It Shows |
|------|-------|---------------|
| `pytorch-dataloader-starvation.db` | [pytorch/pytorch#154318](https://github.com/pytorch/pytorch/issues/154318) | PyTorch DataLoader 114x slower than direct indexing. 200K+ context switches, GPU starving for data. |
| `vllm-37343-logprobs-amplification.db` | [vllm-project/vllm#37343](https://github.com/vllm-project/vllm/issues/37343) | vLLM n_completions + logprobs blocks all co-scheduled requests for 11+ seconds. 80% kernel throughput drop. |
| `pytorch-173382-empty-cache.db` | [pytorch/pytorch#173382](https://github.com/pytorch/pytorch/issues/173382) | `torch.cuda.empty_cache()` not freeing memory. cudaFree p99=1.9ms, cudaMemcpyAsync 98.6x slowdown from CPU scheduling. |
| `vllm-37308-hol-blocking.db` | [vllm-project/vllm#37308](https://github.com/vllm-project/vllm/issues/37308) | vLLM head-of-line blocking with prefix caching. 14.5x TTFT regression. Available as [release asset](https://github.com/ingero-io/ingero/releases). |

## Investigate with AI (copy & paste)

Pick a database and run. Type `/investigate` when prompted to start a guided investigation.

**PyTorch DataLoader starvation** ([pytorch/pytorch#154318](https://github.com/pytorch/pytorch/issues/154318)):
```bash
cat > /tmp/ingero-mcp.json << 'EOF'
{"mcpServers":{"ingero":{"command":"./bin/ingero","args":["mcp","--db","investigations/pytorch-dataloader-starvation.db"]}}}
EOF
ollmcp -m minimax-m2.7:cloud -j /tmp/ingero-mcp.json
```

**PyTorch empty_cache leak** ([pytorch/pytorch#173382](https://github.com/pytorch/pytorch/issues/173382)):
```bash
cat > /tmp/ingero-mcp.json << 'EOF'
{"mcpServers":{"ingero":{"command":"./bin/ingero","args":["mcp","--db","investigations/pytorch-173382-empty-cache.db"]}}}
EOF
ollmcp -m minimax-m2.7:cloud -j /tmp/ingero-mcp.json
```

**vLLM logprobs amplification** ([vllm-project/vllm#37343](https://github.com/vllm-project/vllm/issues/37343)):
```bash
cat > /tmp/ingero-mcp.json << 'EOF'
{"mcpServers":{"ingero":{"command":"./bin/ingero","args":["mcp","--db","investigations/vllm-37343-logprobs-amplification.db"]}}}
EOF
ollmcp -m minimax-m2.7:cloud -j /tmp/ingero-mcp.json
```

Swap `minimax-m2.7:cloud` for any Ollama model (`qwen3.5:cloud`, `llama3.3`, etc.), or use Claude Desktop / Cursor by adding the `mcpServers` block to your MCP config.

### Example questions after `/investigate`

- "What was the core reason for the GPU stall?"
- "Which CUDA operation was hit the hardest?"
- "Show me the causal chains"
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
