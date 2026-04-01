// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
//
// host_trace.bpf.c — eBPF tracepoints for host kernel event tracing
//
// Attaches to kernel tracepoints to capture host-level events affecting
// GPU workload performance. Events correlated with CUDA events in userspace
// to identify root causes of latency spikes.

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "common.bpf.h"

// Separate ring buffer from CUDA — independent reader avoids head-of-line blocking.
// 1MB: host events are 48 bytes each (struct host_event, v0.7 with cgroup_id);
// at heavy contention (~100K mm_page_alloc/sec), ~21,800 events fit (218ms buffer).
// Sized smaller than CUDA/driver buffers since host events have lower per-PID throughput.
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1024 * 1024);
} host_events SEC(".maps");

// In-kernel PID filter. Only events involving these PIDs are emitted.
// Populated by Go agent; 256 entries covers K8s nodes with many GPU pods.
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 256);
	__type(key, __u32);   // PID (tgid)
	__type(value, __u8);  // presence = "trace this PID"
} target_pids SEC(".maps");

// In-kernel cgroup filter. Parallel to target_pids — allows filtering by
// cgroup v2 ID for K8s container scoping without fragile PID enumeration.
// 128 entries covers typical K8s nodes (one container per entry).
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 128);
	__type(key, __u64);   // cgroup_id (from bpf_get_current_cgroup_id)
	__type(value, __u8);  // presence flag
} target_cgroups SEC(".maps");

// Off-CPU timestamp tracking for sched_switch duration computation.
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 8192);
	__type(key, __u32);      // TID
	__type(value, __u64);    // off-cpu timestamp
} sched_off_map SEC(".maps");

// is_target checks if a process should be traced.
// Dual filter: match either by PID or by cgroup ID.
// When target_cgroups is empty (bare-metal mode), falls back to PID-only.
static __always_inline bool is_target(__u32 pid, __u64 cgroup_id)
{
	if (bpf_map_lookup_elem(&target_pids, &pid) != NULL)
		return true;
	if (bpf_map_lookup_elem(&target_cgroups, &cgroup_id) != NULL)
		return true;
	return false;
}

// is_target_pid is the PID-only check for sched events where we only have
// the PID from the tracepoint context (not the current task's cgroup).
static __always_inline bool is_target_pid(__u32 pid)
{
	return bpf_map_lookup_elem(&target_pids, &pid) != NULL;
}

static __always_inline void emit_host_event(__u32 pid, __u32 tid,
					    __u8 op, __u64 duration_ns,
					    __u32 cpu, __u32 target_pid)
{
	struct host_event *evt;

	evt = bpf_ringbuf_reserve(&host_events, sizeof(*evt), 0);
	if (!evt)
		return;

	evt->hdr.timestamp_ns = bpf_ktime_get_ns();
	evt->hdr.pid = pid;
	evt->hdr.tid = tid;
	evt->hdr.source = EVENT_SRC_HOST;
	evt->hdr.op = op;
	evt->hdr._pad = 0;
	evt->hdr._pad2 = 0;
	evt->hdr.cgroup_id = bpf_get_current_cgroup_id();
	evt->duration_ns = duration_ns;
	evt->cpu = cpu;
	evt->target_pid = target_pid;

	bpf_ringbuf_submit(evt, 0);
}

// ---- sched_switch ----
// Record off-CPU start for outgoing target; compute off-CPU duration for incoming.
//
// NOTE on cgroup_id: Uses is_target_pid() (PID-only filter), not is_target(),
// because prev_pid/next_pid come from the tracepoint context — they are NOT
// the "current task" from the kernel's perspective. bpf_get_current_cgroup_id()
// would return the scheduler's cgroup (wrong). The emitted cgroup_id in the
// event header still reflects the current task (the scheduler context), so
// Go-side code should NOT rely on it for container attribution of sched events.
SEC("tp/sched/sched_switch")
int handle_sched_switch(struct trace_event_raw_sched_switch *ctx)
{
	__u32 prev_pid = ctx->prev_pid;
	__u32 next_pid = ctx->next_pid;
	__u32 cpu = bpf_get_smp_processor_id();
	__u64 now = bpf_ktime_get_ns();

	if (is_target_pid(prev_pid))
		bpf_map_update_elem(&sched_off_map, &prev_pid, &now, BPF_ANY);

	// Emit event when a target process comes BACK on-CPU after being preempted.
	// next_pid is the incoming (resuming) task. If it's a target and has a
	// recorded off-CPU timestamp, compute the off-CPU duration.
	// target_pid = prev_pid: the process that was running immediately before
	// the target resumed — the most recent preemptor. Used by the straggler
	// detector to populate preempting_pids in StraggleState messages.
	if (is_target_pid(next_pid)) {
		__u64 *off_ts = bpf_map_lookup_elem(&sched_off_map, &next_pid);
		if (off_ts) {
			__u64 off_cpu_ns = now - *off_ts;
			emit_host_event(next_pid, next_pid,
					HOST_OP_SCHED_SWITCH, off_cpu_ns,
					cpu, prev_pid);
			bpf_map_delete_elem(&sched_off_map, &next_pid);
		}
	}

	return 0;
}

// ---- sched_wakeup ----
SEC("tp/sched/sched_wakeup")
int handle_sched_wakeup(struct trace_event_raw_sched_wakeup_template *ctx)
{
	__u32 wakee_pid = ctx->pid;

	if (!is_target_pid(wakee_pid))
		return 0;

	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 waker_tgid = (__u32)(pid_tgid >> 32);
	__u32 waker_tid = (__u32)pid_tgid;

	emit_host_event(waker_tgid, waker_tid,
			HOST_OP_SCHED_WAKEUP, 0,
			bpf_get_smp_processor_id(), wakee_pid);

	return 0;
}

// ---- mm_page_alloc ----
// Alloc size passed in duration_ns field; Go side interprets by op type.
// Uses is_target() (dual PID+cgroup filter) because the current task IS the
// allocating process — bpf_get_current_cgroup_id() correctly identifies the
// container. Same applies to process_exec and process_exit below.
SEC("tp/kmem/mm_page_alloc")
int handle_mm_page_alloc(struct trace_event_raw_mm_page_alloc *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;
	__u64 cgroup_id = bpf_get_current_cgroup_id();

	if (!is_target(tgid, cgroup_id))
		return 0;

	__u64 alloc_bytes = (__u64)4096 << ctx->order;

	emit_host_event(tgid, tid,
			HOST_OP_PAGE_ALLOC, alloc_bytes,
			bpf_get_smp_processor_id(), 0);

	return 0;
}

// ---- oom/mark_victim ----
// Always emitted (no PID filter) — OOM is rare and always relevant.
SEC("tp/oom/mark_victim")
int handle_oom_kill(struct trace_event_raw_mark_victim *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;

	emit_host_event(tgid, tid,
			HOST_OP_OOM_KILL, 0,
			bpf_get_smp_processor_id(), ctx->pid);

	return 0;
}

// ---- sched_process_exec ----
SEC("tp/sched/sched_process_exec")
int handle_process_exec(struct trace_event_raw_sched_process_exec *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;
	__u64 cgroup_id = bpf_get_current_cgroup_id();

	if (!is_target(tgid, cgroup_id))
		return 0;

	emit_host_event(tgid, tid,
			HOST_OP_PROCESS_EXEC, 0,
			bpf_get_smp_processor_id(), 0);

	return 0;
}

// ---- sched_process_exit ----
SEC("tp/sched/sched_process_exit")
int handle_process_exit(struct trace_event_raw_sched_process_template *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;
	__u64 cgroup_id = bpf_get_current_cgroup_id();

	if (!is_target(tgid, cgroup_id))
		return 0;

	emit_host_event(tgid, tid,
			HOST_OP_PROCESS_EXIT, 0,
			bpf_get_smp_processor_id(), 0);

	return 0;
}

// ---- sched_process_fork ----
// Emits child PID in target_pid; Go agent dynamically adds it to the filter map.
SEC("tp/sched/sched_process_fork")
int handle_process_fork(struct trace_event_raw_sched_process_fork *ctx)
{
	__u32 parent_pid = ctx->parent_pid;

	if (!is_target_pid(parent_pid))
		return 0;

	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = (__u32)(pid_tgid >> 32);
	__u32 tid = (__u32)pid_tgid;
	__u32 child_pid = ctx->child_pid;

	emit_host_event(tgid, tid,
			HOST_OP_PROCESS_FORK, 0,
			bpf_get_smp_processor_id(), child_pid);

	return 0;
}

// Force BTF type emission for bpf2go code generation.
const struct host_event *_unused_host_event_force_btf __attribute__((unused));

char LICENSE[] SEC("license") = "Dual BSD/GPL";
