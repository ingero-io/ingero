// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
#ifndef __INGERO_COMMON_BPF_H
#define __INGERO_COMMON_BPF_H

/* Shared event types for all eBPF programs */

/* Event source identifiers */
#define EVENT_SRC_CUDA    1
#define EVENT_SRC_NVIDIA  2
#define EVENT_SRC_HOST    3
#define EVENT_SRC_DRIVER  4
#define EVENT_SRC_IO      5
#define EVENT_SRC_TCP     6
#define EVENT_SRC_NET     7

/* Host kernel operation types */
#define HOST_OP_SCHED_SWITCH   1
#define HOST_OP_SCHED_WAKEUP   2
#define HOST_OP_PAGE_ALLOC     3
#define HOST_OP_OOM_KILL       4
#define HOST_OP_PROCESS_EXEC   5
#define HOST_OP_PROCESS_EXIT   6
#define HOST_OP_PROCESS_FORK   7

/* CUDA runtime operation types */
#define CUDA_OP_MALLOC           1
#define CUDA_OP_FREE             2
#define CUDA_OP_LAUNCH_KERNEL    3
#define CUDA_OP_MEMCPY           4
#define CUDA_OP_STREAM_SYNC      5
#define CUDA_OP_DEVICE_SYNC      6
#define CUDA_OP_MEMCPY_ASYNC     7
#define CUDA_OP_MALLOC_MANAGED   8

/* CUDA driver operation types (libcuda.so) */
#define DRIVER_OP_LAUNCH_KERNEL    1
#define DRIVER_OP_MEMCPY           2
#define DRIVER_OP_MEMCPY_ASYNC     3
#define DRIVER_OP_CTX_SYNC         4
#define DRIVER_OP_MEM_ALLOC        5
#define DRIVER_OP_MEM_ALLOC_MANAGED 6

/* Block I/O operation types */
#define IO_OP_READ       1
#define IO_OP_WRITE      2
#define IO_OP_DISCARD    3

/* TCP operation types */
#define TCP_OP_RETRANSMIT  1

/* Network socket operation types */
#define NET_OP_SEND   1
#define NET_OP_RECV   2

/* Per-thread entry state: timestamp + args at CUDA/driver function entry.
 * Shared by cuda_trace.bpf.c and driver_trace.bpf.c.
 * Keyed by TID — each thread can only be in one CUDA call at a time.
 */
struct entry_state {
	__u64 timestamp_ns;
	__u8  op;
	__u8  _pad[7];
	__u64 arg0;
	__u64 arg1;
};

/* Stack trace capture — max userspace frames per event.
 * bpf_get_stack() supports up to 127 frames; 64 covers deep Python +
 * native CUDA call chains without bloating ring buffer events to >1KB.
 */
#define MAX_STACK_DEPTH 64

/* Base event header — all events start with this.
 *
 * Layout with explicit padding (32 bytes total):
 *   offset 0:  timestamp_ns  (u64)
 *   offset 8:  pid           (u32)
 *   offset 12: tid           (u32)
 *   offset 16: source        (u8)
 *   offset 17: op            (u8)
 *   offset 18: _pad          (u16)
 *   offset 20: _pad2         (u32) — explicit; replaces implicit compiler padding
 *   offset 24: cgroup_id     (u64) — bpf_get_current_cgroup_id() for K8s container scoping
 *
 * v0.6 header was 24 bytes (20 explicit + 4 implicit padding).
 * v0.7 adds cgroup_id at offset 24, making the header 32 bytes (+8 net).
 */
struct ingero_event_hdr {
	__u64 timestamp_ns;
	__u32 pid;
	__u32 tid;
	__u8  source;       /* EVENT_SRC_* */
	__u8  op;           /* operation type */
	__u16 _pad;
	__u32 _pad2;        /* explicit alignment padding (was implicit in v0.6) */
	__u64 cgroup_id;    /* cgroup v2 inode ID; 0 or 1 = no meaningful cgroup */
};

/* CUDA runtime event (64 bytes, was 56 in v0.6) */
struct cuda_event {
	struct ingero_event_hdr hdr;
	__u64 duration_ns;
	__u64 arg0;          /* operation-specific: size, device_id, etc. */
	__u64 arg1;
	__s32 return_code;
	__u32 gpu_id;
};

/* nvidia.ko driver event */
struct nvidia_event {
	struct ingero_event_hdr hdr;
	__u64 duration_ns;
	__u64 dma_size;
	__u32 gpu_id;
	__u32 context_id;
};

/* Host kernel event (48 bytes, was 40 in v0.6) */
struct host_event {
	struct ingero_event_hdr hdr;
	__u64 duration_ns;
	__u32 cpu;
	__u32 target_pid;    /* for sched events: who was affected */
};

/* Block I/O event (64 bytes) — block_rq_issue / block_rq_complete.
 * Prefixed with ingero_ to avoid colliding with kernel's struct io_event in vmlinux.h.
 */
struct ingero_io_event {
	struct ingero_event_hdr hdr;
	__u64 duration_ns;      /* time from issue to complete */
	__u32 dev;              /* device major:minor (MKDEV) */
	__u32 nr_sector;        /* request size in sectors */
	__u64 sector;           /* starting sector number */
	__u8  rwbs;             /* R=read, W=write, D=discard */
	__u8  _pad_io[7];
};

/* TCP event (48 bytes) — tcp_retransmit_skb */
struct ingero_tcp_event {
	struct ingero_event_hdr hdr;
	__u32 saddr;            /* source IPv4 address */
	__u32 daddr;            /* destination IPv4 address */
	__u16 sport;            /* source port */
	__u16 dport;            /* destination port */
	__u8  state;            /* TCP state at time of retransmit */
	__u8  _pad_tcp[3];
};

/* Network socket event (56 bytes) — sendto/recvfrom syscalls */
struct ingero_net_event {
	struct ingero_event_hdr hdr;
	__u64 duration_ns;      /* syscall duration (entry → exit) */
	__u32 fd;               /* socket file descriptor */
	__u32 bytes;            /* bytes sent or received */
	__u8  direction;        /* NET_OP_SEND or NET_OP_RECV */
	__u8  _pad_net[7];
};

/*
 * Runtime configuration — written by Go userspace, read by eBPF programs.
 * Stored in a BPF_MAP_TYPE_ARRAY with a single entry (key 0).
 */
struct ingero_config {
	__u8 capture_stack;  /* 1 = capture userspace stack traces, 0 = skip */
	__u8 _pad[7];
};

/*
 * Extended CUDA/driver event with userspace stack trace.
 *
 * When config.capture_stack == 1, the uretprobe emits this instead of the
 * base cuda_event. The Go parser distinguishes by record length:
 *   64 bytes  → cuda_event (no stack)
 *   584 bytes → cuda_event_stack (with stack)
 *
 * stack_ips[] is filled by bpf_get_stack(BPF_F_USER_STACK). The helper
 * writes raw instruction pointers (IPs) from the userspace call chain.
 * Symbol resolution happens in Go, not in eBPF.
 *
 * This struct is allocated via bpf_ringbuf_reserve(), NOT on the eBPF
 * stack (which is limited to 512 bytes). 584 bytes in ring buffer is fine.
 */
struct cuda_event_stack {
	struct ingero_event_hdr hdr;       /* 32 bytes (was 20+4pad in v0.6) */
	__u64 duration_ns;                 /* 8 */
	__u64 arg0;                        /* 8 */
	__u64 arg1;                        /* 8 */
	__s32 return_code;                 /* 4 */
	__u32 gpu_id;                      /* 4 */
	/* --- stack section (aligned to 8 bytes by preceding fields) --- */
	__u16 stack_depth;                 /* 2: number of valid IPs */
	__u16 _stack_pad[3];               /* 6: align stack_ips to 8-byte boundary */
	__u64 stack_ips[MAX_STACK_DEPTH];  /* 512: raw instruction pointers */
};
/* Total: 32 + 8+8+8+4+4 + 2+6+512 = 584 bytes (was 576 in v0.6) */

/*
 * target_cgroups — in-kernel cgroup filter map (defined in host_trace.bpf.c).
 *
 * When populated by Go userspace, allows filtering host events by cgroup v2 ID
 * — essential for K8s container scoping without fragile PID enumeration.
 * Currently only used by host tracepoints; CUDA/driver uprobes filter by PID.
 *
 * 128 entries covers typical K8s nodes (one entry per container on node).
 * Note: SetTargetCGroup() is wired but not yet called from CLI — reserved for
 * v0.8 noisy-neighbor detection (per-cgroup scheduler latency).
 */

// Watchdog heartbeat map -- orchestrator writes timestamp every 10ms
// Probes check staleness to bypass remediation if orchestrator is dead
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u64);  // nanosecond timestamp from bpf_ktime_get_ns()
} ingero_watchdog SEC(".maps");

// ---- Watchdog: orchestrator liveness check (Story 3.4) ----
// If the orchestrator heartbeat is older than this threshold, cudaMalloc and
// cudaFree probes skip event processing. 50ms = 50,000,000 nanoseconds.
// Matches config.toml [watchdog] stale_threshold_ms default.
#define WATCHDOG_STALE_NS 50000000ULL

/*
 * watchdog_is_stale -- returns 1 if orchestrator heartbeat is missing or expired.
 * Reads ingero_watchdog[0] (defined above by Story 3.3).
 * Both this and the orchestrator use CLOCK_BOOTTIME (bpf_ktime_get_ns).
 */
static __always_inline int watchdog_is_stale(void)
{
	__u32 key = 0;
	__u64 *last_hb = bpf_map_lookup_elem(&ingero_watchdog, &key);
	if (!last_hb || *last_hb == 0)
		return 1;  /* no heartbeat ever written -> bypass */

	__u64 now = bpf_ktime_get_ns();
	__u64 delta = now - *last_hb;
	return delta > WATCHDOG_STALE_NS;
}

#endif /* __INGERO_COMMON_BPF_H */
