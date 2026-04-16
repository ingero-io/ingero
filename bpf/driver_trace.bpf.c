// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
//
// driver_trace.bpf.c — eBPF uprobes for CUDA Driver API tracing (libcuda.so)
//
// The CUDA Driver API (cuLaunchKernel, cuMemcpy*, cuCtxSynchronize) is called
// directly by cuBLAS, cuDNN, and other NVIDIA libraries — bypassing the CUDA
// Runtime API (libcudart.so). Without these probes, kernel launches from
// optimized math libraries are invisible.
//
// Uses the same struct cuda_event output format as cuda_trace.bpf.c, with
// hdr.source = EVENT_SRC_DRIVER to distinguish the source.

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "common.bpf.h"

// Ring buffer for sending driver events to userspace (separate from CUDA runtime).
// 8MB: with --stack (600-byte events, v0.10 +16 for hdr.comm), ~13,900 events buffer.
// Without --stack (80-byte events), ~104,800 events. cuBLAS can fire 17K+ launches/sec.
// Increased from 2MB after H100 testing showed 3.5% stack coverage at high rates.
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 8 * 1024 * 1024);
} driver_events SEC(".maps");

// Runtime configuration map — Go writes config, eBPF reads it.
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct ingero_config);
} driver_config_map SEC(".maps");

// entry_state is defined in common.bpf.h (shared with cuda_trace.bpf.c).

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 8192);
	__type(key, __u32);
	__type(value, struct entry_state);
} driver_entry_map SEC(".maps");

/*
 * driver_sample_counter: per-CPU event counter for adaptive sampling.
 * Incremented on every event; events are skipped when counter % rate != 0.
 */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u64);
} driver_sample_counter SEC(".maps");

// ---- Helpers ----

/*
 * driver_should_sample: returns true if the current event should be emitted
 * under the configured sampling_rate. Rate 0 or 1 = always emit.
 * Rate N > 1 = emit 1 in every N events (per-CPU).
 */
static __always_inline int driver_should_sample(struct ingero_config *cfg) {
	if (!cfg || cfg->sampling_rate <= 1) {
		return 1;
	}
	__u32 zero = 0;
	__u64 *counter = bpf_map_lookup_elem(&driver_sample_counter, &zero);
	if (!counter) {
		return 1;  /* safe default — emit on lookup failure */
	}
	__u64 c = __sync_fetch_and_add(counter, 1);
	return (c % cfg->sampling_rate) == 0;
}

static __always_inline void driver_save_entry(__u32 tid, __u8 op, __u64 arg0, __u64 arg1)
{
	struct entry_state state = {};
	state.timestamp_ns = bpf_ktime_get_ns();
	state.op = op;
	state.arg0 = arg0;
	state.arg1 = arg1;

	bpf_map_update_elem(&driver_entry_map, &tid, &state, BPF_ANY);
}

static __always_inline void driver_emit_event(struct pt_regs *ctx,
					      __u32 pid, __u32 tid,
					      struct entry_state *entry,
					      __s32 return_code)
{
	__u64 now = bpf_ktime_get_ns();

	// Check if stack capture is enabled.
	__u32 key = 0;
	struct ingero_config *cfg = bpf_map_lookup_elem(&driver_config_map, &key);

	/* Adaptive sampling: skip this event when rate > 1 and counter % rate != 0. */
	if (!driver_should_sample(cfg))
		return;

	if (cfg && cfg->capture_stack) {
		struct cuda_event_stack *sevt;
		sevt = bpf_ringbuf_reserve(&driver_events, sizeof(*sevt), 0);
		if (!sevt)
			goto fallback; /* ring full → emit base event without stack */

		sevt->hdr.timestamp_ns = entry->timestamp_ns;
		sevt->hdr.pid = pid;
		sevt->hdr.tid = tid;
		sevt->hdr.source = EVENT_SRC_DRIVER;
		sevt->hdr.op = entry->op;
		sevt->hdr._pad = 0;
		sevt->hdr._pad2 = 0;
		sevt->hdr.cgroup_id = bpf_get_current_cgroup_id();
		bpf_get_current_comm(&sevt->hdr.comm, sizeof(sevt->hdr.comm));
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
	evt = bpf_ringbuf_reserve(&driver_events, sizeof(*evt), 0);
	if (!evt)
		return;

	evt->hdr.timestamp_ns = entry->timestamp_ns;
	evt->hdr.pid = pid;
	evt->hdr.tid = tid;
	evt->hdr.source = EVENT_SRC_DRIVER;
	evt->hdr.op = entry->op;
	evt->hdr._pad = 0;
	evt->hdr._pad2 = 0;
	evt->hdr.cgroup_id = bpf_get_current_cgroup_id();
	bpf_get_current_comm(&evt->hdr.comm, sizeof(evt->hdr.comm));
	evt->duration_ns = now - entry->timestamp_ns;
	evt->arg0 = entry->arg0;
	evt->arg1 = entry->arg1;
	evt->return_code = return_code;
	evt->gpu_id = 0;

	bpf_ringbuf_submit(evt, 0);
}

// ---- cuLaunchKernel ----
// CUresult cuLaunchKernel(CUfunction f, ...)
// arg0 = function handle

SEC("uprobe/cuLaunchKernel")
int uprobe_cu_launch_kernel(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 func_handle = (__u64)PT_REGS_PARM1(ctx);

	driver_save_entry(tid, DRIVER_OP_LAUNCH_KERNEL, func_handle, 0);
	return 0;
}

SEC("uretprobe/cuLaunchKernel")
int uretprobe_cu_launch_kernel(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&driver_entry_map, &tid);
	if (!entry)
		return 0;

	driver_emit_event(ctx, pid, tid, entry, (__s32)PT_REGS_RC(ctx));
	bpf_map_delete_elem(&driver_entry_map, &tid);
	return 0;
}

// ---- cuMemcpy ----
// CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
// arg0 = ByteCount (param 3)

SEC("uprobe/cuMemcpy")
int uprobe_cu_memcpy(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 byte_count = (__u64)PT_REGS_PARM3(ctx);

	driver_save_entry(tid, DRIVER_OP_MEMCPY, byte_count, 0);
	return 0;
}

SEC("uretprobe/cuMemcpy")
int uretprobe_cu_memcpy(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&driver_entry_map, &tid);
	if (!entry)
		return 0;

	driver_emit_event(ctx, pid, tid, entry, (__s32)PT_REGS_RC(ctx));
	bpf_map_delete_elem(&driver_entry_map, &tid);
	return 0;
}

// ---- cuMemcpyAsync ----
// CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
// arg0 = ByteCount (param 3)

SEC("uprobe/cuMemcpyAsync")
int uprobe_cu_memcpy_async(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 byte_count = (__u64)PT_REGS_PARM3(ctx);

	driver_save_entry(tid, DRIVER_OP_MEMCPY_ASYNC, byte_count, 0);
	return 0;
}

SEC("uretprobe/cuMemcpyAsync")
int uretprobe_cu_memcpy_async(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&driver_entry_map, &tid);
	if (!entry)
		return 0;

	driver_emit_event(ctx, pid, tid, entry, (__s32)PT_REGS_RC(ctx));
	bpf_map_delete_elem(&driver_entry_map, &tid);
	return 0;
}

// ---- cuCtxSynchronize ----
// CUresult cuCtxSynchronize(void)
// No args.

SEC("uprobe/cuCtxSynchronize")
int uprobe_cu_ctx_sync(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();

	driver_save_entry(tid, DRIVER_OP_CTX_SYNC, 0, 0);
	return 0;
}

SEC("uretprobe/cuCtxSynchronize")
int uretprobe_cu_ctx_sync(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&driver_entry_map, &tid);
	if (!entry)
		return 0;

	driver_emit_event(ctx, pid, tid, entry, (__s32)PT_REGS_RC(ctx));
	bpf_map_delete_elem(&driver_entry_map, &tid);
	return 0;
}

// ---- cuMemAlloc_v2 ----
// CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
// arg0 = bytesize (param 2)

SEC("uprobe/cuMemAlloc_v2")
int uprobe_cu_mem_alloc(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 bytesize = (__u64)PT_REGS_PARM2(ctx);

	driver_save_entry(tid, DRIVER_OP_MEM_ALLOC, bytesize, 0);
	return 0;
}

SEC("uretprobe/cuMemAlloc_v2")
int uretprobe_cu_mem_alloc(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&driver_entry_map, &tid);
	if (!entry)
		return 0;

	driver_emit_event(ctx, pid, tid, entry, (__s32)PT_REGS_RC(ctx));
	bpf_map_delete_elem(&driver_entry_map, &tid);
	return 0;
}

// ---- cuMemAllocManaged ----
// CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags)
// Unified Memory allocation in driver API. arg0 = bytesize (param 2).
// Symbol may appear as cuMemAllocManaged or cuMemAllocManaged_v2.

SEC("uprobe/cuMemAllocManaged")
int uprobe_cu_mem_alloc_managed(struct pt_regs *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	__u64 bytesize = (__u64)PT_REGS_PARM2(ctx);

	driver_save_entry(tid, DRIVER_OP_MEM_ALLOC_MANAGED, bytesize, 0);
	return 0;
}

SEC("uretprobe/cuMemAllocManaged")
int uretprobe_cu_mem_alloc_managed(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	struct entry_state *entry = bpf_map_lookup_elem(&driver_entry_map, &tid);
	if (!entry)
		return 0;

	driver_emit_event(ctx, pid, tid, entry, (__s32)PT_REGS_RC(ctx));
	bpf_map_delete_elem(&driver_entry_map, &tid);
	return 0;
}

// Force BTF emission for struct cuda_event (reused for driver events).
const struct cuda_event *_unused_driver_event_force_btf __attribute__((unused));
const struct cuda_event_stack *_unused_driver_event_stack_force_btf __attribute__((unused));
const struct ingero_config *_unused_driver_config_force_btf __attribute__((unused));

char LICENSE[] SEC("license") = "Dual BSD/GPL";
