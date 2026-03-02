// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
//
// tcp_trace.bpf.c — eBPF tracepoint for TCP retransmission tracing
//
// Traces TCP segment retransmissions to diagnose network issues affecting
// GPU workloads: NCCL hangs ("847 retransmissions to rank 3"), tool call
// attribution ("GPU idle during 2s HTTP call").
//
// Limitation: RDMA/InfiniBand bypasses TCP — no visibility for NCCL over IB.
//
// Tracepoint:
//   tp_btf/tcp_retransmit_skb — TCP segment retransmitted

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "common.bpf.h"

// Ring buffer — 256KB (retransmits are rare events, even under load).
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} tcp_events SEC(".maps");

// ---- tcp_retransmit_skb ----
// Fired when the TCP stack retransmits a segment.
// Args: (const struct sock *sk, const struct sk_buff *skb)
SEC("tp_btf/tcp_retransmit_skb")
int BPF_PROG(handle_tcp_retransmit, const struct sock *sk,
	     const struct sk_buff *skb)
{
	struct ingero_tcp_event *evt = bpf_ringbuf_reserve(&tcp_events,
		sizeof(struct ingero_tcp_event), 0);
	if (!evt)
		return 0;

	__u64 pid_tgid = bpf_get_current_pid_tgid();

	evt->hdr.timestamp_ns = bpf_ktime_get_ns();
	evt->hdr.pid = (__u32)(pid_tgid >> 32);
	evt->hdr.tid = (__u32)pid_tgid;
	evt->hdr.source = EVENT_SRC_TCP;
	evt->hdr.op = TCP_OP_RETRANSMIT;
	evt->hdr._pad = 0;
	evt->hdr._pad2 = 0;
	evt->hdr.cgroup_id = bpf_get_current_cgroup_id();

	// Extract IPv4 addresses from sock_common.
	// For IPv6, these will be 0 (v0.8 scope: IPv4 only).
	evt->saddr = BPF_CORE_READ(sk, __sk_common.skc_rcv_saddr);
	evt->daddr = BPF_CORE_READ(sk, __sk_common.skc_daddr);
	evt->sport = BPF_CORE_READ(sk, __sk_common.skc_num);
	evt->dport = __builtin_bswap16(BPF_CORE_READ(sk, __sk_common.skc_dport));
	evt->state = BPF_CORE_READ(sk, __sk_common.skc_state);

	bpf_ringbuf_submit(evt, 0);
	return 0;
}

// Force BTF emission.
const struct ingero_tcp_event *_unused_tcp_event_force_btf __attribute__((unused));

char LICENSE[] SEC("license") = "Dual BSD/GPL";
