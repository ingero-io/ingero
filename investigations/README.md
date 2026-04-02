# GPU Investigation Trace Data

Real-world GPU performance investigations traced with Ingero. Each `.db` file is a SQLite database containing CUDA API call timings, host kernel events, and causal chain analysis.

## Databases

| File | Issue | What It Shows |
|------|-------|---------------|
| `pytorch-dataloader-starvation.db` | [pytorch/pytorch#154318](https://github.com/pytorch/pytorch/issues/154318) | PyTorch DataLoader 114x slower than direct indexing. 200K+ context switches, GPU starving for data. |
| `vllm-37343-logprobs-amplification.db` | [vllm-project/vllm#37343](https://github.com/vllm-project/vllm/issues/37343) | vLLM n_completions + logprobs blocks all co-scheduled requests for 11+ seconds. 80% kernel throughput drop. |
| `pytorch-173382-empty-cache.db` | [pytorch/pytorch#173382](https://github.com/pytorch/pytorch/issues/173382) | `torch.cuda.empty_cache()` not freeing memory. cudaFree p99=1.9ms, cudaMemcpyAsync 98.6x slowdown from CPU scheduling. |
| `vllm-37308-hol-blocking.db` | [vllm-project/vllm#37308](https://github.com/vllm-project/vllm/issues/37308) | vLLM head-of-line blocking with prefix caching. 14.5x TTFT regression. Available as [release asset](https://github.com/ingero-io/ingero/releases). |
| `cuda-graph-cpu-contention.db` | CUDA Graph + CPU contention | `torch.compile` inference with batch size change triggering graph re-capture under CPU contention. 71% graph launch rate drop, 33x cudaLaunchKernel p99 blowup. CUDA Graph lifecycle events (capture, instantiate, launch) with causal correlation. **Captured 2026-04-02 04:38–04:39 UTC** (~50s window, 147K events). When using `--since`, note that event timestamps are from this date — use `--since 24h` or a wide window if querying after the capture date. |

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

**CUDA Graph + CPU contention** (v0.9.0 demo):
```bash
cat > /tmp/ingero-mcp.json << 'EOF'
{"mcpServers":{"ingero":{"command":"./bin/ingero","args":["mcp","--db","investigations/cuda-graph-cpu-contention.db"]}}}
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

> **Note on `--since`:** The `--since` flag filters by wall-clock time relative to *now*, not the capture time. For saved databases, use a wide window (e.g. `--since 8760h` for 1 year) or omit it if your version supports that. The CUDA Graph demo DB was captured on 2026-04-02 ~04:38 UTC.

```bash
# View causal chains
ingero explain --db investigations/pytorch-dataloader-starvation.db --since 8760h

# Per-process GPU API breakdown
ingero explain --db investigations/pytorch-dataloader-starvation.db --per-process --since 8760h

# Query raw events
ingero query --db investigations/pytorch-dataloader-starvation.db --since 8760h --op cudaMemcpyAsync

# CUDA Graph demo — causal chains including graph correlation
ingero explain --db investigations/cuda-graph-cpu-contention.db --since 8760h
```

## Demo Recordings

GIF recordings from the CUDA Graph demo (v0.9.0), located in `docs/assets/`:

| GIF | What It Shows |
|-----|---------------|
| `demo-graph-trace.gif` | Live `ingero trace` on a `torch.compile` workload with CPU contention — graph events streaming alongside CUDA and host events |
| `demo-graph-investigate.gif` | `ingero explain` showing 8 causal chains including CUDA Graph correlation, then `ingero query` for graph capture events |
| `demo-graph-ai-mcp.gif` | Real Claude Code session using Ingero MCP tools (`get_trace_stats`, `get_causal_chains`, `graph_lifecycle`, `graph_frequency`) to diagnose the same trace |

Editable `.cast` source files are in `docs/demo-recordings/`.

## Reproduction

**CUDA Graph demo** (requires any NVIDIA GPU + PyTorch 2.x):
```bash
# 1. Run the demo workload
python tests/workloads/cuda_graph_demo.py &

# 2. Add CPU contention
stress-ng --cpu 2 --timeout 30s &

# 3. Trace with Ingero
sudo ingero trace --pid $(pgrep -f cuda_graph_demo) --db demo.db --duration 30s

# 4. Investigate
sudo ingero explain --db demo.db

# 5. AI investigation (Claude Code + MCP)
claude mcp add -s local ingero -- sudo ingero mcp --db demo.db
claude
# Then ask: "Use ingero tools to investigate this GPU trace"
```

## Environment

All traces captured on TensorDock RTX 4090 (24GB), Ubuntu 22.04, kernel 5.15, NVIDIA driver 570.211.01.
- PyTorch investigation: PyTorch 2.10.0+cu128
- vLLM investigations: vLLM 0.17.1, Qwen/Qwen2.5-0.5B-Instruct with prefix caching
- CUDA Graph demo: EC2 g4dn.xlarge (Tesla T4, 15GB), Ubuntu 24.04, kernel 6.17, NVIDIA 580.126.09, PyTorch 2.10+CUDA 12.0
