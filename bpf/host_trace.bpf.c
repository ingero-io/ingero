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
// 1MB: host events are 64 bytes each (struct host_event, v0.10 with hdr.comm; was 48 in v0.9);
// at heavy contention (~100K mm_page_alloc/sec), ~16,400 events fit (164ms buffer).
// Sized smaller than CUDA/driver buffers since host events have lower per-PID throughput.
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1024 * 1024);
} host_events SEC(".maps");

/*
 * critical_events: dedicated ring buffer for low-frequency, high-value
 * events that must never be dropped — OOM kills, process exec/exit/fork.
 * These events are NEVER sampled or aggregated. 256KB is sufficient
 * headroom (combined event rate well under 100/sec on a busy host).
 */
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} critical_events SEC(".maps");

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

/*
 * mm_alloc_agg: aggregates mm_page_alloc events per PID to reduce
 * ring buffer pressure. Entries drained periodically from userspace
 * via the drainAggregationMaps goroutine.
 *
 * PERCPU_HASH: each CPU maintains its own copy of the value, eliminating
 * contention on __sync_fetch_and_add. Userspace sums across CPUs when
 * draining. Only used for NON-target (background) PIDs — tracked CUDA
 * PIDs still emit raw events for OOM/allocation correlation.
 */
struct mm_alloc_stats {
	__u64 count;
	__u64 total_bytes;
	__u64 last_ts;
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_HASH);
	__uint(max_entries, 4096);
	__type(key, __u32);  /* pid */
	__type(value, struct mm_alloc_stats);
} mm_alloc_agg SEC(".maps");

/*
 * sched_switch_agg: aggregates non-CUDA sched_switch events.
 * Key combines prev_pid and next_pid (upper 32 bits prev, lower next)
 * so each prev->next transition is counted under a single key.
 */
struct sched_switch_stats {
	__u64 count;
	__u64 total_off_cpu_ns;
	__u64 last_ts;
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_HASH);
	__uint(max_entries, 4096);
	__type(key, __u64);  /* (prev_pid << 32) | next_pid */
	__type(value, struct sched_switch_stats);
} sched_switch_agg SEC(".maps");

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

// emit_host_event populates and submits a host event.
//
// comm_src: optional pointer to a kernel-resident TASK_COMM_LEN buffer. If
// non-NULL, hdr.comm is read from this address (used by sched_switch where
// the event's PID is next_pid, not current — so bpf_get_current_comm() would
// capture the scheduler/preemptor, not the resuming target). If NULL, falls
// back to bpf_get_current_comm() which is correct when the event PID matches
// the current task (mm_page_alloc, process_exec, process_exit, fork, wakeup).
static __always_inline void emit_host_event(__u32 pid, __u32 tid,
					    __u8 op, __u64 duration_ns,
					    __u32 cpu, __u32 target_pid,
					    const char *comm_src)
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
	// Defensive zero-init: bpf_ringbuf_reserve does NOT zero its memory.
	// If bpf_probe_read_kernel below fails (very unlikely for tracepoint ctx
	// fields, but defensive against future verifier-permitted variants), the
	// comm field would otherwise leak whatever bytes were in the ringbuf slot.
	__builtin_memset(&evt->hdr.comm, 0, sizeof(evt->hdr.comm));
	if (comm_src)
		bpf_probe_read_kernel(&evt->hdr.comm, sizeof(evt->hdr.comm), comm_src);
	else
		bpf_get_current_comm(&evt->hdr.comm, sizeof(evt->hdr.comm));
	evt->duration_ns = duration_ns;
	evt->cpu = cpu;
	evt->target_pid = target_pid;

	bpf_ringbuf_submit(evt, 0);
}

// emit_critical_event mirrors emit_host_event but targets the dedicated
// critical_events ring buffer. Used for OOM kills and process lifecycle
// (exec/exit/fork) — events that must never be dropped due to high-frequency
// traffic on the main host_events buffer. See critical_events map definition
// for rationale.
static __always_inline void emit_critical_event(__u32 pid, __u32 tid,
						__u8 op, __u64 duration_ns,
						__u32 cpu, __u32 target_pid,
						const char *comm_src)
{
	struct host_event *evt;

	evt = bpf_ringbuf_reserve(&critical_events, sizeof(*evt), 0);
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
	// Defensive zero-init — bpf_ringbuf_reserve does not zero memory.
	__builtin_memset(&evt->hdr.comm, 0, sizeof(evt->hdr.comm));
	if (comm_src)
		bpf_probe_read_kernel(&evt->hdr.comm, sizeof(evt->hdr.comm), comm_src);
	else
		bpf_get_current_comm(&evt->hdr.comm, sizeof(evt->hdr.comm));
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
	bool prev_is_target = is_target_pid(prev_pid);
	bool next_is_target = is_target_pid(next_pid);

	if (prev_is_target)
		bpf_map_update_elem(&sched_off_map, &prev_pid, &now, BPF_ANY);

	// Emit event when a target process comes BACK on-CPU after being preempted.
	// next_pid is the incoming (resuming) task. If it's a target and has a
	// recorded off-CPU timestamp, compute the off-CPU duration.
	// target_pid = prev_pid: the process that was running immediately before
	// the target resumed — the most recent preemptor. Used by the straggler
	// detector to populate preempting_pids in StraggleState messages.
	if (next_is_target) {
		__u64 *off_ts = bpf_map_lookup_elem(&sched_off_map, &next_pid);
		if (off_ts) {
			__u64 off_cpu_ns = now - *off_ts;
			// next_comm is the resuming target's comm — captured by the
			// scheduler in tracepoint context. bpf_get_current_comm() would
			// return the outgoing task's comm (wrong attribution).
			emit_host_event(next_pid, next_pid,
					HOST_OP_SCHED_SWITCH, off_cpu_ns,
					cpu, prev_pid, ctx->next_comm);
			bpf_map_delete_elem(&sched_off_map, &next_pid);
			return 0;
		}
	}

	// If neither side is a tracked target, aggregate in-kernel to avoid
	// flooding the ring buffer with high-frequency background switches.
	// We include the off-CPU duration heuristically when we can compute it
	// (prev had a recorded off-CPU timestamp), otherwise we only bump count.
	if (!prev_is_target && !next_is_target) {
		__u64 key = ((__u64)prev_pid << 32) | (__u64)next_pid;
		__u64 off_cpu_ns = 0;
		__u64 *off_ts = bpf_map_lookup_elem(&sched_off_map, &next_pid);
		if (off_ts) {
			off_cpu_ns = now - *off_ts;
			bpf_map_delete_elem(&sched_off_map, &next_pid);
		}

		struct sched_switch_stats *stats =
			bpf_map_lookup_elem(&sched_switch_agg, &key);
		if (stats) {
			__sync_fetch_and_add(&stats->count, 1);
			if (off_cpu_ns)
				__sync_fetch_and_add(&stats->total_off_cpu_ns, off_cpu_ns);
			stats->last_ts = now;
		} else {
			struct sched_switch_stats new_stats = {
				.count = 1,
				.total_off_cpu_ns = off_cpu_ns,
				.last_ts = now,
			};
			bpf_map_update_elem(&sched_switch_agg, &key, &new_stats, BPF_ANY);
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

	// Event PID = waker (current task), so bpf_get_current_comm() is correct.
	emit_host_event(waker_tgid, waker_tid,
			HOST_OP_SCHED_WAKEUP, 0,
			bpf_get_smp_processor_id(), wakee_pid, NULL);

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
	__u64 alloc_bytes = (__u64)4096 << ctx->order;

	// Tracked target (CUDA PID or matched cgroup): emit raw event for full
	// fidelity — required for OOM/allocation correlation windows.
	if (is_target(tgid, cgroup_id)) {
		emit_host_event(tgid, tid,
				HOST_OP_PAGE_ALLOC, alloc_bytes,
				bpf_get_smp_processor_id(), 0, NULL);
		return 0;
	}

	// Non-target PIDs: aggregate in-kernel to reduce ring buffer pressure.
	// mm_page_alloc fires >100K/sec under load — per-event emission would
	// drop most events. Userspace drains this map on a 1s tick and emits
	// one summary event per PID per window.
	__u64 now = bpf_ktime_get_ns();
	__u32 pid_key = tgid;
	struct mm_alloc_stats *stats = bpf_map_lookup_elem(&mm_alloc_agg, &pid_key);
	if (stats) {
		__sync_fetch_and_add(&stats->count, 1);
		__sync_fetch_and_add(&stats->total_bytes, alloc_bytes);
		stats->last_ts = now;
	} else {
		struct mm_alloc_stats new_stats = {
			.count = 1,
			.total_bytes = alloc_bytes,
			.last_ts = now,
		};
		// BPF_ANY: insert-or-update. If map is full (4096 unique PIDs), the
		// update fails silently — acceptable loss for non-target workloads.
		bpf_map_update_elem(&mm_alloc_agg, &pid_key, &new_stats, BPF_ANY);
	}

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

	// OOM killer fires from the allocator's task context, which may or may
	// not be the victim. Use current task comm — same fidelity caveat as PID.
	// Routed to critical_events: guaranteed delivery for this rare, high-value event.
	emit_critical_event(tgid, tid,
			    HOST_OP_OOM_KILL, 0,
			    bpf_get_smp_processor_id(), ctx->pid, NULL);

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

	// Routed to critical_events: process lifecycle must not be dropped.
	emit_critical_event(tgid, tid,
			    HOST_OP_PROCESS_EXEC, 0,
			    bpf_get_smp_processor_id(), 0, NULL);

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

	// Routed to critical_events: process lifecycle must not be dropped.
	emit_critical_event(tgid, tid,
			    HOST_OP_PROCESS_EXIT, 0,
			    bpf_get_smp_processor_id(), 0, NULL);

	return 0;
}

// ---- sched_process_fork ----
// Emits child PID in target_pid; Go agent dynamically adds it to the filter map.
SEC("tp/sched/sched_process_fork")
int handle_process_fork(struct trace_event_raw_sched_process_fork *ctx)
{
	// ctx->parent_pid is the kernel PID of the TASK that called fork,
	// which in a multi-threaded process is a worker thread (TID), NOT
	// the tgid that target_pids is keyed on. Filter on the CURRENT
	// task's tgid instead — that's the user-space PID of the process
	// that owns the thread. Fixes dropped fork events for torch and
	// any other multithreaded Python / DDP / Ray workloads (Bug 11).
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = (__u32)(pid_tgid >> 32);

	if (!is_target_pid(tgid))
		return 0;

	__u32 tid = (__u32)pid_tgid;
	__u32 child_pid = ctx->child_pid;

	// Event PID = parent tgid (user-space PID), so userspace sees the
	// process identity that matches target_pids.
	// Routed to critical_events: process lifecycle must not be dropped.
	emit_critical_event(tgid, tid,
			    HOST_OP_PROCESS_FORK, 0,
			    bpf_get_smp_processor_id(), child_pid, NULL);

	return 0;
}

// Force BTF type emission for bpf2go code generation.
// NOTE: After modifying this file, run `make generate` to regenerate the
// Go bpf2go bindings (hosttrace_bpfel.go). Requires clang + the bpf2go
// toolchain; Windows dev machines should build on Linux/WSL.
const struct host_event *_unused_host_event_force_btf __attribute__((unused));
const struct mm_alloc_stats *_unused_mm_alloc_stats_force_btf __attribute__((unused));
const struct sched_switch_stats *_unused_sched_switch_stats_force_btf __attribute__((unused));

char LICENSE[] SEC("license") = "Dual BSD/GPL";
