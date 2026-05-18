# vLLM

Run the Ingero agent next to a
[vLLM](https://github.com/vllm-project/vllm) inference server on the
same host. The agent attaches eBPF uprobes to vLLM's CUDA and NCCL
calls, so the trace shows where inference time goes.

## Run

```bash
cd examples/integrations/vllm
docker compose up -d
```

`docker-compose.yaml` starts two containers:

- **vllm** - the inference server, OpenAI-compatible API on port 8000.
- **ingero** - the agent running `trace --record --record-all`,
  sharing the host PID namespace so it sees vLLM's processes.

## Why `--record --record-all`

`trace` alone uses selective storage: under a light or healthy
workload it aggregates high-volume CUDA events instead of keeping
every row, so a short demo run shows no individual CUDA ops in
`ingero query`. `--record --record-all` disables that and stores
every event, so the trace shows `cudaLaunchKernel`, `graphLaunch`,
`cudaMemcpyAsync`, and the rest. Pass both flags: `--record-all`
takes effect only when `--record` is also given explicitly. It is
the right setting for an example or a short investigation; the
trade-off is a larger database.

For a long-running production deployment, use `--inference` instead
of `--record --record-all`. That is the v0.16 inference daemon mode:
selective storage with an event sampler, automatic database rollover,
and per-workload step-duration outlier detection. Edit the `ingero`
service `command` in `docker-compose.yaml` to switch.

Send a request, then inspect the trace:

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "facebook/opt-125m", "prompt": "GPU observability is", "max_tokens": 32}'

# Causal analysis
docker compose exec ingero ingero explain --db /var/lib/ingero/vllm.db --since 5m

# CUDA op breakdown from vLLM (cudaLaunchKernel, graphLaunch, cudaMemcpyAsync, ...)
docker compose exec ingero ingero query --db /var/lib/ingero/vllm.db \
  "SELECT source, op, count(*) FROM events GROUP BY source, op ORDER BY 3 DESC"
```

## Requirements

- An NVIDIA GPU, the NVIDIA Container Toolkit, and a host kernel with
  BTF enabled.
- The agent runs `privileged` with `pid: host` and mounts
  `/sys/kernel/debug`, `/sys/kernel/btf`, and `/sys/fs/bpf` - the same
  requirements as any eBPF tool. See the repository README.

## Notes

- The default model is `facebook/opt-125m`, small enough to pull and
  load quickly. Change the `--model` arg for a real workload.
- The agent traces every GPU process on the host, not only vLLM. With
  `pid: host` that is the point: a colocated training job or a second
  model server shows up in the same trace.
- Trace data persists to `./data` on the host. Remove that directory
  to start clean.
