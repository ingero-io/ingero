# Stack Tracing

Stack tracing is **on by default**: every CUDA / Driver API event captures
the full userspace call chain. Shows **who called cudaMalloc**: from the
CUDA library up through PyTorch, your Python code, and all the way to
`main()`. GPU-measured overhead is **0.4-0.6%** (within noise on RTX 3090
through H100). Disable with `--stack=false` if needed.

```bash
sudo ingero trace --json               # JSON with resolved stack traces (stacks on by default)
sudo ingero trace --debug              # debug output shows resolved frames on stderr
sudo ingero demo --gpu --json          # GPU demo with stack traces (needs sudo)
ingero explain                         # post-hoc causal analysis from DB (no sudo)
sudo ingero trace --stack=false        # disable stacks if needed
```

**Maximum depth**: 64 native frames (eBPF `bpf_get_stack`). This covers
deep call chains from CUDA → cuBLAS/cuDNN → PyTorch C++ → Python
interpreter and up to `main()` / `_start`.

## Python Stack Attribution

For Python workloads (PyTorch, TensorFlow, etc.), Ingero extracts
**CPython frame information** directly from process memory. When a native
frame is inside libpython's eval loop, the corresponding Python source
frames are injected into the stack:

```
[Python] train.py:8 in train_step()
[Python] train.py:13 in main()
[Python] train.py:1 in <module>()
[Native] cublasLtSSSMatmul+0x1d4 (libcublasLt.so.12)
[Native] cublasSgemm_v2+0xa6 (libcublas.so.12)
[Native] (libtorch_cuda.so)
```

Supported Python versions: **3.10, 3.11, 3.12** (covers Ubuntu 22.04
default, conda default, and most production deployments). Version
detection is automatic via `/proc/[pid]/maps`.

### Why you want a Python frame walker

Native stack traces alone stop at `_PyEval_EvalFrameDefault`: the C
function that runs the Python bytecode interpreter. Every frame above
that in "what your code is actually doing" lives in interpreter state
(`PyThreadState`, `_PyInterpreterFrame`, `PyCodeObject`), not in the C
call stack. Without a walker, you see `_PyEval_EvalFrameDefault`
repeated N times, which tells you nothing about which `.py` file
triggered the slow `cuLaunchKernel`.

A Python frame walker reads CPython's own data structures and
reconstructs the source-level call chain (`train.py:train_step`,
`model.py:forward`, ...). That's what lets you answer "which Python
line launched this slow kernel?" instead of "something inside the
interpreter launched it."

Ingero ships **two walker implementations** for this:

- **Userspace walker (default)** runs in the Go process after an event
  arrives. Reads target process memory via `/proc/[pid]/mem` or
  `process_vm_readv`. Simple, flexible, handles the full CPython
  offset fallback chain (`_Py_DebugOffsets` → known-offsets DB →
  DWARF → hardcoded).
- **In-kernel eBPF walker (opt-in)** walks frames from inside the
  kernel probe via `bpf_probe_read_user` helpers. No `/proc/[pid]/mem`
  access needed. Required when `kernel.yama.ptrace_scope=3` (hardened
  systems), and useful when you want frame capture to happen
  synchronously with the CUDA event rather than asynchronously on
  event arrival.

### How to use it

**Default (userspace walker):** Just pass `--stack`: frames appear
automatically for supported Python versions.

```bash
sudo ingero trace --stack --duration 30s
```

You'll see `py_file` / `py_func` / `py_line` fields in JSON output, or
`[Python] <file>:<line> in <func>()` entries in the table/debug view.

**eBPF walker (opt-in):** Pass `--py-walker=ebpf` alongside `--stack`.

```bash
sudo ingero trace --stack --py-walker=ebpf --duration 30s
```

Use the eBPF walker when:

- Your system has `kernel.yama.ptrace_scope=3` (the userspace walker
  can't read process memory there)
- You want guaranteed synchronous frame capture at the exact moment of
  the CUDA event
- You're running on a read-only / hardened host where
  `/proc/[pid]/mem` access is blocked

Stick with the default (`--py-walker=auto`, which resolves to the
userspace walker) when:

- You're on a normal Linux host (ptrace_scope 0, 1, or 2): the
  userspace walker is simpler and has full offset-fallback coverage
  including distro-patched CPython builds
- You care about minimum per-event overhead: the eBPF walker adds BPF
  helper-call cost per emitted event

**Troubleshooting missing frames:** Run `ingero check`: it reports
your `kernel.yama.ptrace_scope` value and tells you what to do if
it's blocking the userspace walker. For CPython 3.12 you'll also
benefit automatically from the self-describing `_Py_DebugOffsets`
struct (no debug symbols needed); for 3.10/3.11 on patched distro
builds, installing the matching `python3.X-dbgsym` package gives
the userspace walker DWARF offsets to fall back on.

## JSON Output with `--stack`

Real output from a PyTorch ResNet-50 training run on A100 SXM4: a
cuBLAS matmul kernel launch captured via Driver API uprobes, with the
full call chain from Python through cuBLAS to the GPU:

```json
{
  "timestamp": "2026-02-25T12:06:24.753983243Z",
  "pid": 11435,
  "tid": 11435,
  "source": "driver",
  "op": "cuLaunchKernel",
  "duration_ns": 10900,
  "duration": "11us",
  "stack": [
    {"ip": "0x0", "py_file": "train.py", "py_func": "train_step", "py_line": 8},
    {"ip": "0x0", "py_file": "train.py", "py_func": "main", "py_line": 13},
    {"ip": "0x0", "py_file": "train.py", "py_func": "<module>", "py_line": 1},
    {"ip": "0x765bb62cfa44", "symbol": "cublasLtSSSMatmul+0x1d4", "file": "libcublasLt.so.12.8.4.1"},
    {"ip": "0x765be7734046", "symbol": "cublasSgemm_v2+0xa6", "file": "libcublas.so.12.8.4.1"},
    {"ip": "0x765c2517fa49", "file": "libtorch_cuda.so"}
  ]
}
```

This kernel launch is invisible to CUDA Runtime profilers: cuBLAS calls
`cuLaunchKernel` directly. Only Ingero's Driver API uprobes capture it.

## Debug Output with `--stack --debug`

```
[DEBUG] stack trace for cuLaunchKernel (PID 11435, TID 11435, 6 frames):
[DEBUG]   [0] [Python] train.py:8 in train_step()
[DEBUG]   [1] [Python] train.py:13 in main()
[DEBUG]   [2] [Python] train.py:1 in <module>()
[DEBUG]   [3] cublasLtSSSMatmul+0x1d4 (libcublasLt.so.12)
[DEBUG]   [4] cublasSgemm_v2+0xa6 (libcublas.so.12)
[DEBUG]   [5] (libtorch_cuda.so)
```
