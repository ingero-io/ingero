# Troubleshooting + Advanced Configuration

Recurring GPU workload issues that Ingero detects automatically (with
documented fixes), plus operational cheat sheets for the most common
operator questions and reference material for power users.

## Patterns Ingero detects

### CUDA Graph capture fails immediately (cuBLAS lazy initialization)

**Symptom:** `cudaStreamBeginCapture` followed by cuBLAS or cuDNN
calls fails immediately. Errors surface as
`CUBLAS_STATUS_NOT_INITIALIZED`, a failed `cudaStreamEndCapture`, or
an invalid graph handle. In traces, the capture region is abnormally
short (duration < 1ms) and contains no kernel launches.

**Cause:** cuBLAS and cuDNN lazily create their internal handles,
memory pools, and workspace buffers on the first API call. Those
initialization steps invoke CUDA runtime APIs (`cudaMalloc`,
`cudaEventCreate`, and others) that are disallowed inside a stream
capture region. When the first cuBLAS/cuDNN call happens under
capture, the runtime rejects those disallowed calls and the capture
aborts or produces an invalid graph.

**Fix:** Execute 3+ warmup iterations of the work you intend to
capture before calling `cudaStreamBeginCapture`. Warmup forces
cuBLAS/cuDNN to complete lazy initialization outside the capture
context.

```python
# BAD: capture aborts on first cuBLAS call
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = torch.matmul(a, b)

# GOOD: warmup forces cuBLAS initialization outside capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        y = torch.matmul(a, b)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = torch.matmul(a, b)
```

Alternatively, use `torch.cuda.make_graphed_callables()`, which
handles the warmup sequence automatically.

**Automatic detection:** `ingero explain` surfaces this pattern as a
`graph-capture-warmup` causal chain (MEDIUM severity). Run it after
a trace when you suspect CUDA Graph capture issues.

### Python source frames are missing

**Symptom:** Native frames appear in stack traces, but the Python
file, function, and line fields are empty. The trace shows
`[Native]` frames only; no `[Python]` frames interleave with the
CPython eval loop.

**Causes:**
- `kernel.yama.ptrace_scope >= 1` blocks `/proc/[pid]/mem` access,
  which the userspace walker relies on.
- Distro-patched CPython whose struct offsets differ from upstream.
- CPython version older than 3.10 or newer than the supported set
  (3.10, 3.11, 3.12).

**Fix:**

1. Check `ingero check` for the `ptrace_scope` advisory. At level 0
   or 1 the userspace walker works when ingero runs as root or with
   `CAP_SYS_PTRACE` (the `process_vm_readv` fallback handles level 1
   automatically).
2. For hardened systems at `ptrace_scope=2` or `=3`, pass
   `--py-walker=ebpf` to route frame walking into the kernel via
   eBPF. The in-kernel walker reads CPython frame state directly
   from the task's user memory and bypasses the `/proc/[pid]/mem`
   dependency entirely.
3. For distro builds whose offsets differ from upstream, installing
   `python3-dbgsym` lets ingero use DWARF offsets. CPython 3.12
   additionally uses the self-describing `_Py_DebugOffsets` struct
   when present (no debug symbols needed).

### High event drop rates under load

**Symptom:** Table UI footer shows `Events dropped: cuda=N driver=N
...` with nonzero counts, or a `>5% of events dropped` WARN line.
Per-tracer drop counters are visible in the trace output whenever
drops occur.

**Cause:** Ring buffer or userspace channel saturating under
sustained event rates (typically above ~5M events/sec). The driver
or runtime ring buffers fill faster than the userspace reader drains
them.

**Fix options (in order of preference):**

1. Let adaptive sampling kick in: `--sampling-rate 0`. The adaptive
   path escalates the sampling rate under sustained drops and resets
   when the event stream is quiet. No manual tuning required.
2. Increase ring buffer size for the high-throughput probes:
   `--ringbuf-size 32m` (or larger, must be a power of 2). The flag
   applies to cuda/driver/host ring buffers; low-throughput probes
   keep their compiled defaults.
3. For sustained extreme rates, fix sampling: `--sampling-rate 10`
   emits one in every ten events.

Critical events (OOM kills, process exec/exit/fork) flow through a
dedicated smaller ring buffer and are never subject to sampling or
aggregation. They remain visible even under heavy drop conditions on
the main event stream.

## Operational cheat sheet

Tighter symptom-to-fix entries for common operational questions. The
"Patterns Ingero detects" section above has the full context; these
are the cheat-sheet versions.

**Q: My venv workload isn't being traced.**

Multi-library discovery is automatic. Ingero locates every copy of
`libcudart.so` (system install plus venv/conda copies shipped by
`nvidia-cuda-runtime` pip packages) and attaches probes to all of
them. Confirm with `--debug`: you should see `INFO discover: found
libcudart.so path=...` lines for each copy. Force a specific library
with `--cuda-lib /path/to/libcudart.so` if auto-discovery picks the
wrong one.

**Q: Python source frames don't appear in my stack traces.**

Quick checks: `ingero check | grep ptrace_scope`, ensure you're
running as root or with `CAP_SYS_PTRACE`, and try `--py-walker=ebpf`
for hardened systems. CPython 3.12 gets the best experience via the
self-describing `_Py_DebugOffsets` struct (no debug symbols needed).

**Q: Events are being dropped.**

Start by letting adaptive sampling handle it (`--sampling-rate 0`,
which is the recommended default for variable workloads). Tune
`--ringbuf-size` only if the adaptive path isn't enough. OOM,
process exec/exit, and fork events are guaranteed delivery
regardless of drop rates on the main stream.

**Q: How do I reduce ingero's overhead?**

Default overhead target is `<2%` above the workload's baseline
(NFR3). If you're seeing more:

- Disable low-value probes: `--no-io --no-tcp --no-net` turns off
  block I/O, TCP retransmit, and network socket tracers. CUDA and
  host remain.
- Skip stack capture: omit `--stack` (or pass `--stack=false`) if
  you don't need userspace stack traces; it's the most expensive
  per-event cost.
- Use sampling: `--sampling-rate 10` on very high event workloads.
  For occasional-overhead-spike workloads, adaptive
  (`--sampling-rate 0`, default) is usually sufficient.
- Keep the userspace walker (default `--py-walker=auto`) unless you
  need the eBPF path; the eBPF walker adds helper-call cost per
  event.

## Advanced configuration

Reference material for power users. The defaults are tuned for
typical training and inference workloads; only tweak these if you
have a specific reason.

**Ring buffer sizing.** Default sizes reflect expected event rates
(8MB for cuda/driver, 1MB for host, smaller for tcp/net/block-io).
Increase the high-throughput probe buffers if your workload exceeds
~1-5M events/sec sustained. The `--ringbuf-size` flag applies to
the high-throughput probes only; low-throughput probes keep their
compiled defaults.

**Sampling rate semantics.** `0` = adaptive (the recommended default
for variable workloads). `1` = emit every event (deterministic,
useful for reproducibility testing). `N > 1` = per-CPU event
counter; every Nth event is emitted. Does **not** apply to host
probes (`sched_switch`, `mm_page_alloc`, OOM, exec/exit/fork are
never sampled).

**Python walker choice.** `auto` (default) runs the userspace
walker; it supports 3.10/3.11/3.12 and handles `ptrace_scope` up to
level 2 via a `process_vm_readv` fallback. `ebpf` runs the in-kernel
walker; also supports 3.10/3.11/3.12 and additionally works at
`ptrace_scope=3`. `userspace` forces the userspace walker (disables
any automatic promotion).

**Critical events reliability.** OOM, process exec, exit, and fork
events flow through a dedicated 256KB ring buffer independent of
the main 8MB/1MB buffers. They are never sampled, never aggregated,
and the userspace reader blocks rather than drops; critical signals
(needed for fork-inheritance, OOM correlation, orchestrator
remediation) are guaranteed delivery.

## Related docs

- [`commands.md`](commands.md): full per-command flag reference.
- [`otlp.md`](otlp.md): metric-name catalog including the W2-poller
  bit-to-bucket mapping.
- [`stack_tracing.md`](stack_tracing.md): walker selection, JSON
  output examples, `kernel.yama.ptrace_scope` deep dive.
