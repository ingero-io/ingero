# What Ingero Detects

Ingero addresses 25 documented GPU problems across training, inference,
and AI agent workloads. The README's "What Ingero detects" section
shows the highest-impact 8; the full list is below.

| # | GPU Problem | Severity | How Ingero Detects It |
|---|-------------|----------|----------------------|
| 1 | NCCL hangs & distributed training deadlocks | CRITICAL | Direct `ncclAllReduce` / `ncclSend` / `ncclRecv` enter/exit uprobes (v0.12.0+) measure per-collective wall time, with rank/`comm_id_hash`/`nranks` correlation. `sched_switch` + TCP-retransmit tracing remain as host-side and network-side cross-checks. |
| 2 | GPU underutilization / data pipeline starvation | CRITICAL | Host scheduler + `cudaStreamSync` + `cudaMemcpy` pipeline bubble diagnosis. Block I/O shows DataLoader disk bottleneck |
| 3 | CUDA OOM & memory fragmentation | CRITICAL | `cudaMalloc`/`cuMemAlloc` allocation pattern tracing. `cudaMallocManaged` adds managed-memory over-subscription detection |
| 4 | Silent data corruption (SDC) | CRITICAL | Anomalous kernel timing as indirect signal (limited) |
| 5 | Inference cost explosion (multi-step agents) | CRITICAL | CUDA API burst/idle patterns per agent session |
| 6 | KV cache pressure & preemption cascades | CRITICAL | `cudaMalloc` patterns + `cudaStreamSync` spikes during preemption. Managed-memory page fault detection |
| 6b | CUDA Graph re-capture latency spikes (vLLM, torch.compile) | HIGH | Graph lifecycle tracing: capture/instantiate/launch rates, pool exhaustion detection, OOM during capture, CPU contention during launch |
| 7 | GPU hardware failures at scale | HIGH | `cudaMemcpy` baseline drift, `sched_switch` frequency anomalies |
| 8 | CPU bottleneck in GPU serving | HIGH | `sched_switch` on inference process + `cudaStreamSync` idle gaps |
| 9 | GPU idle waste during agent tool execution | HIGH | CUDA API silence periods correlated with host process activity. TCP tracing shows "GPU idle during 2s HTTP tool call" |
| 10 | GPU memory leaks in long-running services | HIGH | `cudaMalloc`/`cudaFree` imbalance tracking over time, per-container via cgroup |
| 11 | Mixed precision (AMP) instability | HIGH | Anomalous kernel timing (skipped updates = fast sync) |
| 12 | Goodput loss (training efficiency gap) | HIGH | Scheduler preemption, memcpy latency, pipeline bubbles. Block I/O shows checkpoint write + data read overhead |
| 13 | GPU scheduling & orchestration failures | HIGH | Per-cgroup `sched_switch` latency + orchestrator metadata. v0.12.3 added multi-orchestrator detection: K8s (auto-discovers `nvidia.com/gpu` pods), Slurm (`SLURM_JOB_ID`), ECS (`ECS_CONTAINER_METADATA_URI_V4`/V3), Docker / containerd (cgroup hex match). |
| 14 | Model swapping latency (multi-model agents) | HIGH | `cudaMalloc` + `cudaMemcpy` patterns during model load. Block I/O shows disk→CPU transfer time |
| 15 | CUDA device-side asserts & illegal memory access | MEDIUM | CUDA API call sequence + stack traces before crash |
| 16 | NVIDIA driver / CUDA version incompatibility | MEDIUM | Uprobe attachment failure = library/driver mismatch signal |
| 17 | Thermal throttling & power limit throttling | MEDIUM | Kernel duration trending over time |
| 18 | Noisy neighbor / multi-tenant GPU interference | MEDIUM | Per-cgroup `sched_switch` latency + CUDA API latency correlation. Noisy neighbor detection via cgroup_schedstat |
| 19 | Cold start / model loading latency | MEDIUM | Full cold start sequence via CUDA API timing. Block I/O completes disk→CPU→GPU pipeline |
| 20 | Multi-GPU tensor parallel communication overhead | MEDIUM | Direct NCCL collective uprobes (`ncclAllReduce` / `ncclAllGather` / `ncclReduceScatter`, v0.12.0+) measure barrier-wait time per rank with `comm_id_hash` + `nranks` labels. Host-side `sched_switch` + TCP-retransmit on NCCL ports remain as cross-checks. |
| 21 | RAG pipeline GPU contention | MEDIUM | Per-process CUDA API breakdown (`explain --per-process`): shows which process is hogging GPU time |
| 22 | Checkpoint save/load failures | MEDIUM | Memory spike detection + I/O blocking in `cudaStreamSync`. Block I/O shows actual write latency + NFS timeouts |
| 23 | PCIe bottleneck (KV cache swap, model loading) | MEDIUM | `cudaMemcpy` per-operation tracing with direction/size/duration. `cudaMallocManaged` page migration + Block I/O shows NVMe-PCIe contention |
| 24 | Loss spikes (non-AMP) | LOW-MED | System event correlation with loss timing |
| 25 | Triton Inference Server multi-GPU bugs | LOW-MED | CUDA API tracing on Triton processes |
