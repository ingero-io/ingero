# Remediation Protocol

> **Stability: Experimental.** This protocol is part of an active PoC. The
> socket path, wire format, field names, and watchdog mechanism may change
> without notice in future releases. Pin to a specific Ingero version if you
> build on this.

When Ingero runs with `--remediate`, it exposes a Unix Domain Socket (UDS) that
streams real-time GPU memory state and CPU straggler detection signals as
NDJSON. Any external process can connect to this socket and build automated
remediation on top of Ingero's eBPF observability.

## Quick Start

```bash
# Start Ingero with the remediation endpoint
sudo ./bin/ingero trace --remediate

# In another terminal, connect and read the stream
socat - UNIX-CONNECT:/tmp/ingero-remediate.sock
```

## Socket

- **Path:** `/tmp/ingero-remediate.sock` (hardcoded default)
- **Type:** Unix domain socket, stream
- **Connections:** Single consumer at a time. A new connection replaces the
  previous one (the old connection is closed).

## Wire Format

Newline-delimited JSON (NDJSON). Each line is a complete JSON object with a
`"type"` field for dispatch:

```json
{"type":"memory","pid":12345,"allocated_bytes":8589934592,"total_vram":17179869184,"utilization_pct":50.0,"last_alloc_size":268435456,"timestamp_ns":1711180800000000000}
{"type":"straggle","pid":12345,"throughput_drop_pct":45.2,"sched_switch_count":12,"preempting_pids":[6789,6790],"timestamp_ns":1711180800000000000}
```

Consumers should dispatch on the `"type"` field and ignore unknown types for
forward compatibility.

## Message Types

### `memory` — VRAM Memory State

Emitted on every CUDA memory allocation or free observed by the eBPF probes.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"memory"` |
| `pid` | uint32 | Process ID of the GPU workload |
| `allocated_bytes` | uint64 | Current net VRAM allocation balance for this PID |
| `total_vram` | uint64 | Total GPU VRAM in bytes (queried from nvidia-smi at startup) |
| `utilization_pct` | float64 | `allocated_bytes / total_vram * 100` |
| `last_alloc_size` | uint64 | Size of the most recent cudaMalloc/cuMemAlloc call in bytes |
| `timestamp_ns` | int64 | Wall-clock timestamp (Unix nanoseconds) |

**Emission rules:**
- One message per `cudaMalloc`, `cudaFree`, `cuMemAlloc_v2`,
  or `cuMemAllocManaged` call observed by the eBPF probes.

### `straggle` — CPU Scheduling Contention

Emitted when Ingero's straggler detector identifies CPU scheduling interference
correlated with a GPU throughput drop for a process. Both conditions must fire
together to avoid false positives.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"straggle"` |
| `pid` | uint32 | Process ID experiencing scheduling contention |
| `throughput_drop_pct` | float64 | Percentage drop from the EMA throughput baseline |
| `sched_switch_count` | uint32 | Number of `sched_switch` events in the detection interval |
| `preempting_pids` | []uint32 | PIDs of processes that preempted the affected process |
| `timestamp_ns` | int64 | Wall-clock timestamp (Unix nanoseconds) |

**Emission rules:**
- Emitted when both throughput drop and `sched_switch` contention thresholds
  are exceeded for the same PID in the same detection interval.
- Re-emitted periodically while contention persists, so downstream consumers
  that require periodic signals (e.g., within a 2s window) receive updates.

## Common Rules

- Writes use a 50ms deadline. If the consumer is slow, messages are dropped
  (non-blocking — Ingero never stalls).
- If no consumer is connected, messages are silently dropped.

## Watchdog Heartbeat (Optional)

When Ingero loads its eBPF probes, it pins a BPF map at
`/sys/fs/bpf/ingero_watchdog`. A remediation service can write a
`CLOCK_BOOTTIME` nanosecond timestamp to this map at a regular interval
(e.g., every 10ms). The eBPF probes check this timestamp on every
`cudaMalloc`/`cudaFree` — if the heartbeat is stale (>50ms), probes skip
event processing entirely.

This implements a "do no harm" guarantee: if the remediation service crashes,
probes bypass within 50ms and the GPU workload continues unaffected.

**Map details:**
- **Pin path:** `/sys/fs/bpf/ingero_watchdog`
- **Type:** `BPF_MAP_TYPE_ARRAY`, 1 entry
- **Key:** `__u32` (always 0)
- **Value:** `__u64` (nanosecond timestamp from `CLOCK_BOOTTIME`)
- **Write method:** `bpf(BPF_OBJ_GET)` to open the pinned map fd, then
  `bpf(BPF_MAP_UPDATE_ELEM)` to write the timestamp
- **Stale threshold:** 50ms (50,000,000 ns), hardcoded in
  `bpf/common.bpf.h` as `WATCHDOG_STALE_NS`

## Building a Custom Consumer

A minimal consumer in Python:

```python
import json
import socket

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/ingero-remediate.sock")

buf = b""
while True:
    data = sock.recv(4096)
    if not data:
        break
    buf += data
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        msg = json.loads(line)

        if msg["type"] == "memory":
            if msg["utilization_pct"] > 90.0:
                print(f"VRAM WARNING: PID {msg['pid']} at {msg['utilization_pct']:.1f}%")

        elif msg["type"] == "straggle":
            print(f"STRAGGLER: PID {msg['pid']} throughput dropped {msg['throughput_drop_pct']:.1f}%, "
                  f"preempted by {msg['preempting_pids']}")

        else:
            pass  # ignore unknown types for forward compatibility
```
