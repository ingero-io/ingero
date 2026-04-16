// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
//
// cuda_graph_trace.bpf.c — eBPF uprobes for CUDA Graph lifecycle tracing
//
// Captures: cudaStreamBeginCapture, cudaStreamEndCapture,
//           cudaGraphInstantiate, cudaGraphLaunch
//
// Separate from cuda_trace.bpf.c per Architecture Decision 7.
// Shares common.bpf.h types but has its own maps and ring buffer.

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "common.bpf.h"

// Ring buffer for graph events (4MB).
// GraphLaunch can fire at 100-1000+/sec in production (vLLM replays).
// At 88 bytes/event (v0.10 with hdr.comm; was 72), 4MB holds ~47K events (~47s at 1000/sec).
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 4 * 1024 * 1024);
} graph_events SEC(".maps");

// Per-TID entry state for graph API calls.
// Separate from cuda_trace's entry_map to avoid key collisions.
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 8192);
	__type(key, __u32);
	__type(value, struct graph_entry_state);
} graph_entry_map SEC(".maps");

// Runtime configuration map — Go writes config, eBPF reads it.
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct ingero_config);
} graph_config_map SEC(".maps");

/*
 * graph_sample_counter: per-CPU event counter for adaptive sampling.
 * Incremented on every event; events are skipped when counter % rate != 0.
 */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u64);
} graph_sample_counter SEC(".maps");

// ---- Helper: sampling gate ----

/*
 * graph_should_sample: returns true if the current event should be emitted
 * under the configured sampling_rate. Rate 0 or 1 = always emit.
 * Rate N > 1 = emit 1 in every N events (per-CPU).
 */
static __always_inline int graph_should_sample(void) {
	__u32 key = 0;
	struct ingero_config *cfg = bpf_map_lookup_elem(&graph_config_map, &key);
	if (!cfg || cfg->sampling_rate <= 1) {
		return 1;
	}
	__u64 *counter = bpf_map_lookup_elem(&graph_sample_counter, &key);
	if (!counter) {
		return 1;  /* safe default — emit on lookup failure */
	}
	__u64 c = __sync_fetch_and_add(counter, 1);
	return (c % cfg->sampling_rate) == 0;
}

// ---- Helper: save graph entry state ----

static __always_inline void graph_save_entry(__u32 tid, __u8 op,
					     __u64 stream, __u64 graph,
					     __u64 exec, __u32 mode)
{
	struct graph_entry_state state = {};
	state.timestamp_ns = bpf_ktime_get_ns();
	state.op = op;
	state.stream_handle = stream;
	state.graph_handle = graph;
	state.exec_handle = exec;
	state.capture_mode = mode;

	bpf_map_update_elem(&graph_entry_map, &tid, &state, BPF_ANY);
}

// ---- Helper: emit graph event ----

static __always_inline void graph_emit_event(__u32 pid, __u32 tid,
					     struct graph_entry_state *entry,
					     __s32 return_code,
					     __u64 graph_handle,
					     __u64 exec_handle)
{
	/* Adaptive sampling: skip this event when rate > 1 and counter % rate != 0. */
	if (!graph_should_sample())
		return;

	struct cuda_graph_event *evt;
	evt = bpf_ringbuf_reserve(&graph_events, sizeof(*evt), 0);
	if (!evt)
		return;

	evt->hdr.timestamp_ns = entry->timestamp_ns;
	evt->hdr.pid = pid;
	evt->hdr.tid = tid;
	evt->hdr.source = EVENT_SRC_CUDA_GRAPH;
	evt->hdr.op = entry->op;
	evt->hdr._pad = 0;
	evt->hdr._pad2 = 0;
	evt->hdr.cgroup_id = bpf_get_current_cgroup_id();
	bpf_get_current_comm(&evt->hdr.comm, sizeof(evt->hdr.comm));
	evt->duration_ns = bpf_ktime_get_ns() - entry->timestamp_ns;
	evt->stream_handle = entry->stream_handle;
	evt->graph_handle = graph_handle ? graph_handle : entry->graph_handle;
	evt->exec_handle = exec_handle ? exec_handle : entry->exec_handle;
	evt->capture_mode = entry->capture_mode;
	evt->return_code = return_code;

	bpf_ringbuf_submit(evt, 0);
}

// ---- cudaStreamBeginCapture uprobes ----
// cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)
// RDI = stream, RSI = mode

SEC("uprobe/cudaStreamBeginCapture")
int uprobe_graph_begin_capture(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 stream = (__u64)PT_REGS_PARM1(ctx);
	__u32 mode = (__u32)PT_REGS_PARM2(ctx);

	graph_save_entry(tid, GRAPH_OP_BEGIN_CAPTURE, stream, 0, 0, mode);
	return 0;
}

SEC("uretprobe/cudaStreamBeginCapture")
int uretprobe_graph_begin_capture(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct graph_entry_state *entry = bpf_map_lookup_elem(&graph_entry_map, &tid);
	if (!entry)
		return 0;

	__s32 ret = (__s32)PT_REGS_RC(ctx);
	graph_emit_event(pid, tid, entry, ret, 0, 0);

	bpf_map_delete_elem(&graph_entry_map, &tid);
	return 0;
}

// ---- cudaStreamEndCapture uprobes ----
// cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)
// RDI = stream, RSI = pGraph (output pointer)

SEC("uprobe/cudaStreamEndCapture")
int uprobe_graph_end_capture(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 stream = (__u64)PT_REGS_PARM1(ctx);

	graph_save_entry(tid, GRAPH_OP_END_CAPTURE, stream, 0, 0, 0);
	return 0;
}

SEC("uretprobe/cudaStreamEndCapture")
int uretprobe_graph_end_capture(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct graph_entry_state *entry = bpf_map_lookup_elem(&graph_entry_map, &tid);
	if (!entry)
		return 0;

	__s32 ret = (__s32)PT_REGS_RC(ctx);
	graph_emit_event(pid, tid, entry, ret, 0, 0);

	bpf_map_delete_elem(&graph_entry_map, &tid);
	return 0;
}

// ---- cudaGraphInstantiate uprobes ----
// cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, ...)
// RDI = pGraphExec (output pointer), RSI = graph

SEC("uprobe/cudaGraphInstantiate")
int uprobe_graph_instantiate(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 graph = (__u64)PT_REGS_PARM2(ctx);

	graph_save_entry(tid, GRAPH_OP_INSTANTIATE, 0, graph, 0, 0);
	return 0;
}

SEC("uretprobe/cudaGraphInstantiate")
int uretprobe_graph_instantiate(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct graph_entry_state *entry = bpf_map_lookup_elem(&graph_entry_map, &tid);
	if (!entry)
		return 0;

	__s32 ret = (__s32)PT_REGS_RC(ctx);
	graph_emit_event(pid, tid, entry, ret, 0, 0);

	bpf_map_delete_elem(&graph_entry_map, &tid);
	return 0;
}

// ---- cudaGraphLaunch uprobes ----
// cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
// RDI = graphExec, RSI = stream

SEC("uprobe/cudaGraphLaunch")
int uprobe_graph_launch(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 exec = (__u64)PT_REGS_PARM1(ctx);
	__u64 stream = (__u64)PT_REGS_PARM2(ctx);

	graph_save_entry(tid, GRAPH_OP_LAUNCH, stream, 0, exec, 0);
	return 0;
}

SEC("uretprobe/cudaGraphLaunch")
int uretprobe_graph_launch(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct graph_entry_state *entry = bpf_map_lookup_elem(&graph_entry_map, &tid);
	if (!entry)
		return 0;

	__s32 ret = (__s32)PT_REGS_RC(ctx);
	graph_emit_event(pid, tid, entry, ret, 0, 0);

	bpf_map_delete_elem(&graph_entry_map, &tid);
	return 0;
}

// Force BTF type emission for bpf2go code generation.
const struct cuda_graph_event *_unused_cuda_graph_event_force_btf __attribute__((unused));
const struct ingero_config *_unused_graph_config_force_btf __attribute__((unused));

char LICENSE[] SEC("license") = "Dual BSD/GPL";
