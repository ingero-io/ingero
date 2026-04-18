# Remediation Protocol

> **Stability: Experimental.** This protocol is part of an active PoC. The
> socket path, wire format, field names, and watchdog mechanism may change
> without notice in future releases. Pin to a specific Ingero version if you
> build on this.

## Contract Version

**Contract version: 1.**

The authoritative machine-readable contract lives in
`internal/remediate/wire_contract_test.go`. That test fails the build if
any field listed in the `required` column below is removed or renamed.
Rules:

- **Stable** types: required fields may not be removed or renamed
  within a contract version. Adding new optional fields is allowed.
  Breaking changes require a contract-version bump.
- **Experimental** types: fields may change between releases. Consumers
  must tolerate missing or added fields.
- **Deprecation:** a stable field is deprecated by adding a successor
  field and marking the old one optional in the contract. It may not be
  removed until the next contract-version bump and one full release
  where the new field is the primary.

| Type | Stability | Required fields | Optional fields |
|---|---|---|---|
| `memory` | stable | `type`, `pid`, `gpu_id`, `allocated_bytes`, `total_vram`, `utilization_pct`, `last_alloc_size`, `timestamp_ns` | `comm` |
| `straggle` | stable | `type`, `pid`, `throughput_drop_pct`, `sched_switch_count`, `preempting_pids`, `timestamp_ns`, `sustained` | `comm` |
| `straggler_state` | experimental | `type`, `node_id`, `cluster_id`, `score`, `threshold`, `detection_mode`, `dominant_signal`, `timestamp` | |
| `straggler_resolved` | experimental | `type`, `node_id`, `cluster_id`, `timestamp` | |

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
| `comm` | string | Kernel-captured process name (optional; omitted when empty) |
| `gpu_id` | uint32 | GPU index (0..N-1) the allocation targets |
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
| `comm` | string | Kernel-captured process name (optional; omitted when empty) |
| `throughput_drop_pct` | float64 | Percentage drop from the EMA throughput baseline |
| `sched_switch_count` | uint32 | Number of `sched_switch` events in the detection interval |
| `preempting_pids` | []uint32 | PIDs of processes that preempted the affected process |
| `timestamp_ns` | int64 | Wall-clock timestamp (Unix nanoseconds) |
| `sustained` | bool | `false` on initial detection, `true` on re-emissions while pressure persists — lets consumers gate "once per episode" remediation |

**Emission rules:**
- Emitted when both throughput drop and `sched_switch` contention thresholds
  are exceeded for the same PID in the same detection interval.
- Re-emitted periodically while contention persists, so downstream consumers
  that require periodic signals (e.g., within a 2s window) receive updates.

### `straggler_state` — Fleet peer-relative straggler (experimental)

Emitted by `ingero fleet-push` when a node's health score crosses the
Fleet-derived threshold downward (healthy → straggler transition). The
score is peer-relative across the cluster; the threshold comes from the
Fleet processor's rolling statistic.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"straggler_state"` |
| `node_id` | string | Stable node identifier (pod hostname by default) |
| `cluster_id` | string | Logical cluster identifier |
| `score` | float64 | Latest health score pushed for this node |
| `threshold` | float64 | Fleet-derived threshold at the moment of transition |
| `detection_mode` | string | One of `calibrating`, `fleet`, `fleet-cached`, `local-cached`, `local-baseline` |
| `dominant_signal` | string | The signal that drove classification (e.g. `throughput`, `compute`) |
| `timestamp` | RFC3339 | Event wall-clock timestamp |

### `straggler_resolved` — straggler → healthy edge (experimental)

Emitted when a previously-straggler node returns above threshold.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"straggler_resolved"` |
| `node_id` | string | Stable node identifier |
| `cluster_id` | string | Logical cluster identifier |
| `timestamp` | RFC3339 | Event wall-clock timestamp |

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
