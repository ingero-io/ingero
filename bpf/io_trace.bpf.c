// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
//
// io_trace.bpf.c — eBPF tracepoints for block I/O request tracing
//
// Traces block device request issue and completion to measure I/O latency.
// Correlates with CUDA events to identify disk bottlenecks (DataLoader,
// checkpoint writes, model loads, NVMe-PCIe contention).
//
// Tracepoints (raw format, portable across 5.15+ kernels):
//   tp/block/block_rq_issue    — request submitted to device driver
//   tp/block/block_rq_complete — request completed by device

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "common.bpf.h"

// Ring buffer for I/O events — 1MB (same as host; I/O events are infrequent
// compared to CUDA: typical production sees 100-10K IOPS vs 50K+ CUDA ops/s).
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1 * 1024 * 1024);
} io_events SEC(".maps");

// In-flight request map: track issue timestamp for latency computation.
// Key = (dev, sector) which uniquely identifies a block request.
struct io_req_key {
	__u32 dev;
	__u32 _pad;
	__u64 sector;
};

struct io_req_val {
	__u64 timestamp_ns;
	__u64 cgroup_id;     // captured at issue time for propagation to complete
	__u32 pid;
	__u32 tid;
	__u32 nr_sector;
	__u8  rwbs;
	__u8  _pad[3];
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 4096);
	__type(key, struct io_req_key);
	__type(value, struct io_req_val);
} io_inflight SEC(".maps");

// Classify rwbs string to operation type.
// rwbs is a char[8] like "R", "W", "WS", "D", "RA", etc.
static __always_inline __u8 classify_rwbs_str(const char *rwbs)
{
	if (rwbs[0] == 'W')
		return IO_OP_WRITE;
	if (rwbs[0] == 'D')
		return IO_OP_DISCARD;
	return IO_OP_READ; // 'R' or anything else
}

// ---- block_rq_issue — request dispatched to driver ----
// Uses trace_event_raw_block_rq: dev, sector, nr_sector, bytes, rwbs[8], comm[16]
SEC("tp/block/block_rq_issue")
int handle_block_rq_issue(struct trace_event_raw_block_rq *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct io_req_key key = {
		.dev = ctx->dev,
		.sector = ctx->sector,
	};

	struct io_req_val val = {
		.timestamp_ns = bpf_ktime_get_ns(),
		.cgroup_id = bpf_get_current_cgroup_id(),
		.pid = pid,
		.tid = tid,
		.nr_sector = ctx->nr_sector,
		.rwbs = classify_rwbs_str(ctx->rwbs),
	};

	bpf_map_update_elem(&io_inflight, &key, &val, BPF_ANY);
	return 0;
}

// ---- block_rq_complete — request completed ----
// Uses trace_event_raw_block_rq_completion: dev, sector, nr_sector, error, rwbs[8]
SEC("tp/block/block_rq_complete")
int handle_block_rq_complete(struct trace_event_raw_block_rq_completion *ctx)
{
	struct io_req_key key = {
		.dev = ctx->dev,
		.sector = ctx->sector,
	};

	struct io_req_val *val = bpf_map_lookup_elem(&io_inflight, &key);
	if (!val)
		return 0;

	__u64 now = bpf_ktime_get_ns();
	__u64 duration_ns = now - val->timestamp_ns;

	struct ingero_io_event *evt = bpf_ringbuf_reserve(&io_events,
		sizeof(struct ingero_io_event), 0);
	if (!evt) {
		bpf_map_delete_elem(&io_inflight, &key);
		return 0;
	}

	evt->hdr.timestamp_ns = now;
	evt->hdr.pid = val->pid;
	evt->hdr.tid = val->tid;
	evt->hdr.source = EVENT_SRC_IO;
	evt->hdr.op = val->rwbs;
	evt->hdr._pad = 0;
	evt->hdr._pad2 = 0;
	evt->hdr.cgroup_id = val->cgroup_id; // captured at issue time (complete runs in IRQ context)
	evt->duration_ns = duration_ns;
	evt->dev = ctx->dev;
	evt->nr_sector = val->nr_sector;
	evt->sector = ctx->sector;
	evt->rwbs = val->rwbs;

	bpf_ringbuf_submit(evt, 0);
	bpf_map_delete_elem(&io_inflight, &key);
	return 0;
}

// Force BTF emission.
const struct ingero_io_event *_unused_io_event_force_btf __attribute__((unused));

char LICENSE[] SEC("license") = "Dual BSD/GPL";
