// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
//
// net_trace.bpf.c — eBPF tracepoints for network socket I/O tracing
//
// Traces socket send/recv syscalls to measure network request latency.
// Correlates with CUDA events to identify "GPU idle during network I/O"
// patterns in inference serving (vLLM, Triton) and tool-calling agents.
//
// Approach: syscall tracepoints on sendto/recvfrom. No HTTP parsing in
// eBPF — just socket-level byte counting and timing.
//
// Tracepoints:
//   tp/syscalls/sys_enter_sendto — socket send entry
//   tp/syscalls/sys_exit_sendto  — socket send return
//   tp/syscalls/sys_enter_recvfrom — socket recv entry
//   tp/syscalls/sys_exit_recvfrom  — socket recv return

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "common.bpf.h"

// Ring buffer — 512KB (socket I/O can be frequent for inference servers).
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 512 * 1024);
} net_events SEC(".maps");

// Per-thread entry state for send/recv duration measurement.
struct net_entry {
	__u64 timestamp_ns;
	__u32 fd;
	__u32 len;     // requested bytes
	__u8  op;      // NET_OP_SEND or NET_OP_RECV
	__u8  _pad[7];
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 8192);
	__type(key, __u32);             // key = TID
	__type(value, struct net_entry);
} net_entry_map SEC(".maps");

// PID filter — when populated, only trace sockets from listed PIDs.
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 256);
	__type(key, __u32);
	__type(value, __u8);
} net_target_pids SEC(".maps");

static __always_inline bool net_is_target(__u32 pid) {
	return bpf_map_lookup_elem(&net_target_pids, &pid) != NULL;
}

static __always_inline bool net_pid_map_empty(void) {
	__u32 zero = 0;
	return bpf_map_lookup_elem(&net_target_pids, &zero) == NULL;
}

// ---- sys_enter_sendto ----
// Args: fd, buff, len, flags, addr, addr_len
SEC("tp/syscalls/sys_enter_sendto")
int handle_sys_enter_sendto(struct trace_event_raw_sys_enter *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	// PID filter: if map is populated, only trace listed PIDs.
	// Empty map means trace all (but that's very noisy — CLI should populate it).
	if (!net_pid_map_empty() && !net_is_target(pid))
		return 0;

	struct net_entry entry = {};
	entry.timestamp_ns = bpf_ktime_get_ns();
	entry.fd = (__u32)ctx->args[0];   // fd
	entry.len = (__u32)ctx->args[2];  // len
	entry.op = NET_OP_SEND;

	bpf_map_update_elem(&net_entry_map, &tid, &entry, BPF_ANY);
	return 0;
}

// ---- sys_exit_sendto ----
SEC("tp/syscalls/sys_exit_sendto")
int handle_sys_exit_sendto(struct trace_event_raw_sys_exit *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct net_entry *entry = bpf_map_lookup_elem(&net_entry_map, &tid);
	if (!entry)
		return 0;

	__s64 ret = ctx->ret;  // bytes sent (or negative errno)

	struct ingero_net_event *evt = bpf_ringbuf_reserve(&net_events,
		sizeof(struct ingero_net_event), 0);
	if (!evt) {
		bpf_map_delete_elem(&net_entry_map, &tid);
		return 0;
	}

	evt->hdr.timestamp_ns = bpf_ktime_get_ns();
	evt->hdr.pid = pid;
	evt->hdr.tid = tid;
	evt->hdr.source = EVENT_SRC_NET;
	evt->hdr.op = NET_OP_SEND;
	evt->hdr._pad = 0;
	evt->hdr._pad2 = 0;
	evt->hdr.cgroup_id = bpf_get_current_cgroup_id();
	evt->fd = entry->fd;
	evt->bytes = ret > 0 ? (__u32)ret : 0;
	evt->direction = NET_OP_SEND;

	bpf_ringbuf_submit(evt, 0);
	bpf_map_delete_elem(&net_entry_map, &tid);
	return 0;
}

// ---- sys_enter_recvfrom ----
SEC("tp/syscalls/sys_enter_recvfrom")
int handle_sys_enter_recvfrom(struct trace_event_raw_sys_enter *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	if (!net_pid_map_empty() && !net_is_target(pid))
		return 0;

	struct net_entry entry = {};
	entry.timestamp_ns = bpf_ktime_get_ns();
	entry.fd = (__u32)ctx->args[0];
	entry.len = (__u32)ctx->args[2];
	entry.op = NET_OP_RECV;

	bpf_map_update_elem(&net_entry_map, &tid, &entry, BPF_ANY);
	return 0;
}

// ---- sys_exit_recvfrom ----
SEC("tp/syscalls/sys_exit_recvfrom")
int handle_sys_exit_recvfrom(struct trace_event_raw_sys_exit *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct net_entry *entry = bpf_map_lookup_elem(&net_entry_map, &tid);
	if (!entry)
		return 0;

	__s64 ret = ctx->ret;

	struct ingero_net_event *evt = bpf_ringbuf_reserve(&net_events,
		sizeof(struct ingero_net_event), 0);
	if (!evt) {
		bpf_map_delete_elem(&net_entry_map, &tid);
		return 0;
	}

	evt->hdr.timestamp_ns = bpf_ktime_get_ns();
	evt->hdr.pid = pid;
	evt->hdr.tid = tid;
	evt->hdr.source = EVENT_SRC_NET;
	evt->hdr.op = NET_OP_RECV;
	evt->hdr._pad = 0;
	evt->hdr._pad2 = 0;
	evt->hdr.cgroup_id = bpf_get_current_cgroup_id();
	evt->fd = entry->fd;
	evt->bytes = ret > 0 ? (__u32)ret : 0;
	evt->direction = NET_OP_RECV;

	bpf_ringbuf_submit(evt, 0);
	bpf_map_delete_elem(&net_entry_map, &tid);
	return 0;
}

// Force BTF emission.
const struct ingero_net_event *_unused_net_event_force_btf __attribute__((unused));

char LICENSE[] SEC("license") = "Dual BSD/GPL";
