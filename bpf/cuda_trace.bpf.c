// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
//
// cuda_trace.bpf.c — eBPF uprobes for CUDA Runtime API tracing

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "common.bpf.h"

// Ring buffer for sending events to userspace.
// 8MB: with --stack, events are 576 bytes; at 49K events/sec
// ~14,500 stack events fit (~296ms buffer). Without --stack (56 bytes),
// ~149,000 events fit (~3s buffer). Increased from 2MB after H100
// testing showed 3.5% stack coverage at high event rates.
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 8 * 1024 * 1024);
} events SEC(".maps");

// Runtime configuration — Go writes config, eBPF reads it.
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct ingero_config);
} config_map SEC(".maps");

// entry_state is defined in common.bpf.h (shared with driver_trace.bpf.c).

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 8192);
	__type(key, __u32);             // key = TID
	__type(value, struct entry_state);
} entry_map SEC(".maps");

// ---- Helper functions ----

static __always_inline void save_entry(__u32 tid, __u8 op, __u64 arg0, __u64 arg1)
{
	struct entry_state state = {};
	state.timestamp_ns = bpf_ktime_get_ns();
	state.op = op;
	state.arg0 = arg0;
	state.arg1 = arg1;

	bpf_map_update_elem(&entry_map, &tid, &state, BPF_ANY);
}

// emit_event pushes a completed event to the ring buffer.
// When config.capture_stack is set, emits cuda_event_stack (576 bytes)
// with bpf_get_stack(BPF_F_USER_STACK); otherwise emits cuda_event (56 bytes).
// Go parser distinguishes by record length.
static __always_inline void emit_event(struct pt_regs *ctx,
				       __u32 pid, __u32 tid,
				       struct entry_state *entry,
				       __s32 return_code)
{
	__u64 now = bpf_ktime_get_ns();

	__u32 key = 0;
	struct ingero_config *cfg = bpf_map_lookup_elem(&config_map, &key);
	if (cfg && cfg->capture_stack) {
		struct cuda_event_stack *sevt;
		sevt = bpf_ringbuf_reserve(&events, sizeof(*sevt), 0);
		if (!sevt)
			goto fallback; /* ring full → emit base event without stack */

		sevt->hdr.timestamp_ns = entry->timestamp_ns;
		sevt->hdr.pid = pid;
		sevt->hdr.tid = tid;
		sevt->hdr.source = EVENT_SRC_CUDA;
		sevt->hdr.op = entry->op;
		sevt->hdr._pad = 0;
		sevt->duration_ns = now - entry->timestamp_ns;
		sevt->arg0 = entry->arg0;
		sevt->arg1 = entry->arg1;
		sevt->return_code = return_code;
		sevt->gpu_id = 0;

		long stack_bytes = bpf_get_stack(ctx, sevt->stack_ips,
						 sizeof(sevt->stack_ips),
						 BPF_F_USER_STACK);
		if (stack_bytes > 0)
			sevt->stack_depth = (__u16)(stack_bytes / 8);
		else
			sevt->stack_depth = 0;

		sevt->_stack_pad[0] = 0;
		sevt->_stack_pad[1] = 0;
		sevt->_stack_pad[2] = 0;

		bpf_ringbuf_submit(sevt, 0);
		return;
	}

fallback:;
	struct cuda_event *evt;
	evt = bpf_ringbuf_reserve(&events, sizeof(*evt), 0);
	if (!evt)
		return;

	evt->hdr.timestamp_ns = entry->timestamp_ns;
	evt->hdr.pid = pid;
	evt->hdr.tid = tid;
	evt->hdr.source = EVENT_SRC_CUDA;
	evt->hdr.op = entry->op;
	evt->hdr._pad = 0;
	evt->duration_ns = now - entry->timestamp_ns;
	evt->arg0 = entry->arg0;
	evt->arg1 = entry->arg1;
	evt->return_code = return_code;
	evt->gpu_id = 0;

	bpf_ringbuf_submit(evt, 0);
}

// ---- cudaMalloc uprobes ----
// arg0 = size (bytes requested)

SEC("uprobe/cudaMalloc")
int uprobe_cuda_malloc(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 size = (__u64)PT_REGS_PARM2(ctx);

	save_entry(tid, CUDA_OP_MALLOC, size, 0);
	return 0;
}

SEC("uretprobe/cudaMalloc")
int uretprobe_cuda_malloc(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&entry_map, &tid);
	if (!entry)
		return 0;

	__s32 ret = (__s32)PT_REGS_RC(ctx);
	emit_event(ctx, pid, tid, entry, ret);

	bpf_map_delete_elem(&entry_map, &tid);
	return 0;
}

// ---- cudaFree uprobes ----
// cudaError_t cudaFree(void* devPtr)
// arg0 = devPtr (device pointer being freed — matches cudaMalloc return value)

SEC("uprobe/cudaFree")
int uprobe_cuda_free(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 dev_ptr = (__u64)PT_REGS_PARM1(ctx);

	save_entry(tid, CUDA_OP_FREE, dev_ptr, 0);
	return 0;
}

SEC("uretprobe/cudaFree")
int uretprobe_cuda_free(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&entry_map, &tid);
	if (!entry)
		return 0;

	__s32 ret = (__s32)PT_REGS_RC(ctx);
	emit_event(ctx, pid, tid, entry, ret);

	bpf_map_delete_elem(&entry_map, &tid);
	return 0;
}

// ---- cudaLaunchKernel uprobes ----
// arg0 = GPU kernel function pointer

SEC("uprobe/cudaLaunchKernel")
int uprobe_cuda_launch_kernel(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 func_ptr = (__u64)PT_REGS_PARM1(ctx);

	save_entry(tid, CUDA_OP_LAUNCH_KERNEL, func_ptr, 0);
	return 0;
}

SEC("uretprobe/cudaLaunchKernel")
int uretprobe_cuda_launch_kernel(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&entry_map, &tid);
	if (!entry)
		return 0;

	emit_event(ctx, pid, tid, entry, (__s32)PT_REGS_RC(ctx));
	bpf_map_delete_elem(&entry_map, &tid);
	return 0;
}

// ---- cudaMemcpy uprobes ----
// arg0 = count (bytes), arg1 = kind (direction: 0=H2H, 1=H2D, 2=D2H, 3=D2D, 4=default)

SEC("uprobe/cudaMemcpy")
int uprobe_cuda_memcpy(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 count = (__u64)PT_REGS_PARM3(ctx);
	__u64 kind  = (__u64)PT_REGS_PARM4(ctx);

	save_entry(tid, CUDA_OP_MEMCPY, count, kind);
	return 0;
}

SEC("uretprobe/cudaMemcpy")
int uretprobe_cuda_memcpy(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&entry_map, &tid);
	if (!entry)
		return 0;

	emit_event(ctx, pid, tid, entry, (__s32)PT_REGS_RC(ctx));
	bpf_map_delete_elem(&entry_map, &tid);
	return 0;
}

// ---- cudaStreamSynchronize uprobes ----
// arg0 = stream handle

SEC("uprobe/cudaStreamSynchronize")
int uprobe_cuda_stream_sync(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 stream = (__u64)PT_REGS_PARM1(ctx);

	save_entry(tid, CUDA_OP_STREAM_SYNC, stream, 0);
	return 0;
}

SEC("uretprobe/cudaStreamSynchronize")
int uretprobe_cuda_stream_sync(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&entry_map, &tid);
	if (!entry)
		return 0;

	emit_event(ctx, pid, tid, entry, (__s32)PT_REGS_RC(ctx));
	bpf_map_delete_elem(&entry_map, &tid);
	return 0;
}

// ---- cudaMemcpyAsync uprobes ----
// Uses distinct CUDA_OP_MEMCPY_ASYNC (7) to distinguish from sync cudaMemcpy (4).
// arg0 = count (bytes), arg1 = kind (direction)

SEC("uprobe/cudaMemcpyAsync")
int uprobe_cuda_memcpy_async(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 count = (__u64)PT_REGS_PARM3(ctx);
	__u64 kind  = (__u64)PT_REGS_PARM4(ctx);

	save_entry(tid, CUDA_OP_MEMCPY_ASYNC, count, kind);
	return 0;
}

SEC("uretprobe/cudaMemcpyAsync")
int uretprobe_cuda_memcpy_async(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&entry_map, &tid);
	if (!entry)
		return 0;

	emit_event(ctx, pid, tid, entry, (__s32)PT_REGS_RC(ctx));
	bpf_map_delete_elem(&entry_map, &tid);
	return 0;
}

// ---- cudaDeviceSynchronize uprobes ----

SEC("uprobe/cudaDeviceSynchronize")
int uprobe_cuda_device_sync(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();

	save_entry(tid, CUDA_OP_DEVICE_SYNC, 0, 0);
	return 0;
}

SEC("uretprobe/cudaDeviceSynchronize")
int uretprobe_cuda_device_sync(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&entry_map, &tid);
	if (!entry)
		return 0;

	emit_event(ctx, pid, tid, entry, (__s32)PT_REGS_RC(ctx));
	bpf_map_delete_elem(&entry_map, &tid);
	return 0;
}

// Force BTF type emission for bpf2go code generation.
const struct cuda_event *_unused_cuda_event_force_btf __attribute__((unused));
const struct cuda_event_stack *_unused_cuda_event_stack_force_btf __attribute__((unused));
const struct ingero_config *_unused_config_force_btf __attribute__((unused));

char LICENSE[] SEC("license") = "Dual BSD/GPL";
