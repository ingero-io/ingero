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
#define EVENT_SRC_CUDA_GRAPH 8

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

/* CUDA Graph lifecycle operation types */
#define GRAPH_OP_BEGIN_CAPTURE  1
#define GRAPH_OP_END_CAPTURE    2
#define GRAPH_OP_INSTANTIATE    3
#define GRAPH_OP_LAUNCH         4
/* Reserved for P1: */
#define GRAPH_OP_DESTROY        5
#define GRAPH_OP_EXEC_UPDATE    6
#define GRAPH_OP_EXEC_DESTROY   7

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
 * Layout with explicit padding (48 bytes total):
 *   offset 0:  timestamp_ns  (u64)
 *   offset 8:  pid           (u32)
 *   offset 12: tid           (u32)
 *   offset 16: source        (u8)
 *   offset 17: op            (u8)
 *   offset 18: _pad          (u16) — repurposed as per-event flag bits;
 *                                    see INGERO_EVENT_FLAG_* below. Older
 *                                    probes that never set flag bits
 *                                    leave this at 0 (unchanged ABI).
 *   offset 20: _pad2         (u32) — explicit; replaces implicit compiler padding
 *   offset 24: cgroup_id     (u64) — bpf_get_current_cgroup_id() for K8s container scoping
 *   offset 32: comm          (char[16]) — bpf_get_current_comm() for PID-reuse-resilient process identity
 *
 * v0.6 header was 24 bytes (20 explicit + 4 implicit padding).
 * v0.7 added cgroup_id at offset 24, making the header 32 bytes (+8 net).
 * v0.10 adds comm[16] at offset 32, making the header 48 bytes (+16 net).
 * comm is 16 bytes naturally (matches TASK_COMM_LEN), keeps 8-byte alignment.
 */
struct ingero_event_hdr {
	__u64 timestamp_ns;
	__u32 pid;
	__u32 tid;
	__u8  source;       /* EVENT_SRC_* */
	__u8  op;           /* operation type */
	__u16 _pad;         /* flag bits — see INGERO_EVENT_FLAG_* (most probes leave 0) */
	__u32 _pad2;        /* explicit alignment padding (was implicit in v0.6) */
	__u64 cgroup_id;    /* cgroup v2 inode ID; 0 or 1 = no meaningful cgroup */
	char  comm[16];     /* process name from bpf_get_current_comm() — TASK_COMM_LEN */
};

/*
 * Per-event flag bits stored in ingero_event_hdr._pad.
 * ABI-backwards-compatible: older probes that never set this field leave
 * it 0, which means "no flags" — callers can safely AND-test any bit.
 */
#define INGERO_EVENT_FLAG_PY_FRAMES 0x0001u  /* event payload has trailing py_walk_result */

/* CUDA runtime event (80 bytes, was 64 in v0.9, 56 in v0.6) */
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

/* Host kernel event (64 bytes, was 48 in v0.9, 40 in v0.6) */
struct host_event {
	struct ingero_event_hdr hdr;
	__u64 duration_ns;
	__u32 cpu;
	__u32 target_pid;    /* for sched events: who was affected */
};

/* Block I/O event (80 bytes) — block_rq_issue / block_rq_complete.
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

/* TCP event (64 bytes) — tcp_retransmit_skb */
struct ingero_tcp_event {
	struct ingero_event_hdr hdr;
	__u32 saddr;            /* source IPv4 address */
	__u32 daddr;            /* destination IPv4 address */
	__u16 sport;            /* source port */
	__u16 dport;            /* destination port */
	__u8  state;            /* TCP state at time of retransmit */
	__u8  _pad_tcp[3];
};

/* Network socket event (72 bytes) — sendto/recvfrom syscalls */
struct ingero_net_event {
	struct ingero_event_hdr hdr;
	__u64 duration_ns;      /* syscall duration (entry → exit) */
	__u32 fd;               /* socket file descriptor */
	__u32 bytes;            /* bytes sent or received */
	__u8  direction;        /* NET_OP_SEND or NET_OP_RECV */
	__u8  _pad_net[7];
};

/* Per-thread entry state for CUDA Graph probes.
 * Separate from entry_state to avoid collisions — a thread may be in a
 * graph API call while another map tracks CUDA runtime calls.
 * Keyed by TID.
 */
struct graph_entry_state {
	__u64 timestamp_ns;
	__u8  op;
	__u8  _pad[7];
	__u64 stream_handle;
	__u64 graph_handle;
	__u64 exec_handle;
	__u32 capture_mode;
	__u32 _pad2;
};

/* CUDA Graph lifecycle event (88 bytes).
 *
 * Layout:
 *   offset  0: hdr              (48 bytes — ingero_event_hdr)
 *   offset 48: duration_ns      (8)
 *   offset 56: stream_handle    (8) — stream for BeginCapture/EndCapture/Launch
 *   offset 64: graph_handle     (8) — graph for EndCapture/Instantiate
 *   offset 72: exec_handle      (8) — executable for Instantiate/Launch
 *   offset 80: capture_mode     (4) — for BeginCapture (0=global, 1=thread_local, 2=relaxed)
 *   offset 84: return_code      (4) — cudaError_t
 */
struct cuda_graph_event {
	struct ingero_event_hdr hdr;
	__u64 duration_ns;
	__u64 stream_handle;
	__u64 graph_handle;
	__u64 exec_handle;
	__u32 capture_mode;
	__s32 return_code;
};

/*
 * ingero_config: runtime configuration read by all BPF probes.
 * Stored in a single-entry BPF_MAP_TYPE_ARRAY map (config_map).
 *
 * ABI stability: fields may only be APPENDED. Existing fields and
 * padding must not change (they are read by compiled BPF programs
 * that may predate newer Go-side code).
 */
struct ingero_config {
	__u8  capture_stack;   /* 1 = capture userspace stack traces */
	__u8  _pad1[3];
	__u32 sampling_rate;   /* 0 or 1 = emit all events; N > 1 = emit 1 per N */
	__u32 _pad2;           /* 12 bytes total (u32 alignment, no trailing pad) */
};

/*
 * Extended CUDA/driver event with userspace stack trace.
 *
 * When config.capture_stack == 1, the uretprobe emits this instead of the
 * base cuda_event. The Go parser distinguishes by record length:
 *   80 bytes  → cuda_event (no stack)
 *   600 bytes → cuda_event_stack (with stack)
 *
 * stack_ips[] is filled by bpf_get_stack(BPF_F_USER_STACK). The helper
 * writes raw instruction pointers (IPs) from the userspace call chain.
 * Symbol resolution happens in Go, not in eBPF.
 *
 * This struct is allocated via bpf_ringbuf_reserve(), NOT on the eBPF
 * stack (which is limited to 512 bytes). 600 bytes in ring buffer is fine.
 */
struct cuda_event_stack {
	struct ingero_event_hdr hdr;       /* 48 bytes (was 32 in v0.9, 20+4pad in v0.6) */
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
/* Total: 48 + 8+8+8+4+4 + 2+6+512 = 600 bytes (was 584 in v0.9, 576 in v0.6) */

/*
 * py_runtime_state: per-PID Python runtime configuration pushed from
 * userspace into py_runtime_map. The eBPF Python frame walker reads
 * this to know where _PyRuntime lives and what the offset table is.
 *
 * All offsets are byte offsets within their respective CPython structs.
 * Supports CPython 3.10, 3.11, and 3.12 via python_minor dispatch in
 * the walker.
 *
 * Layout: 8 (runtime_addr) + 12*2 (legacy uint16 offsets) + 1 (python_minor)
 *       + 1 (pad) + 2 (off_cframe_current_frame) = 36 bytes total.
 * ABI-stable append-only: the v1 layout was 32 bytes (3.12-only). v2
 * appends python_minor + pad + off_cframe_current_frame (4 extra bytes).
 * Older 32-byte writes (legacy 3.12-only clients) leave the appended
 * fields zero, which the walker interprets as python_minor==0 => assume
 * 3.12 to preserve backward compatibility.
 */
struct py_runtime_state {
	__u64 runtime_addr;                  /* address of _PyRuntime in target process */
	__u16 off_runtime_interpreters_head; /* _PyRuntime.interpreters.head offset */
	__u16 off_tstate_head;               /* PyInterpreterState.threads.head offset */
	__u16 off_tstate_next;               /* PyThreadState.next offset */
	__u16 off_tstate_native_tid;         /* PyThreadState.native_thread_id offset */
	__u16 off_tstate_frame;              /* PyThreadState.current_frame (3.12) or .frame (3.10) or .cframe (3.11) offset */
	__u16 off_frame_back;                /* _PyInterpreterFrame.previous (3.11/12) or PyFrameObject.f_back (3.10) */
	__u16 off_frame_code;                /* _PyInterpreterFrame.executable (3.11/12) or PyFrameObject.f_code (3.10) */
	__u16 off_code_filename;             /* PyCodeObject.co_filename offset */
	__u16 off_code_name;                 /* PyCodeObject.co_name offset */
	__u16 off_code_firstlineno;          /* PyCodeObject.co_firstlineno offset */
	__u16 off_unicode_state;             /* PyASCIIObject.state offset */
	__u16 off_unicode_data;              /* PyASCIIObject.data inline offset */
	/* v2 fields — appended for multi-version support. Older 32-byte
	 * writes (3.12-only clients) leave these zero, which the walker
	 * interprets as "assume 3.12" to preserve backward compatibility. */
	__u8  python_minor;                  /* 10, 11, or 12; 0 = legacy caller, treat as 12 */
	__u8  _pad3;
	__u16 off_cframe_current_frame;      /* _PyCFrame.current_frame offset (3.11 only; 0 for others) */
	/* v3 field — appended for PEP 684 subinterpreter support.
	 * PyInterpreterState.next is the first field of struct _is across
	 * 3.9..3.14, so the value is 0 on every supported version. The
	 * walker reads at this offset inside an outer loop bounded by
	 * PY_MAX_INTERPRETERS; single-interpreter processes see next==NULL
	 * on the second iteration and exit the loop with identical behavior
	 * to the legacy single-interpreter path. */
	__u16 off_interp_next;               /* PyInterpreterState.next offset */
};

/* Single Python frame extracted by the BPF walker. */
struct py_frame {
	char filename[128];
	char funcname[128];
	__s32 firstlineno;
	__u32 _pad;  /* keep 8-byte aligned, total 264 bytes */
};

#define PY_MAX_FRAMES 16

/*
 * py_walk_result: result returned by walk_python_frames(). Embedded in
 * extended cuda event when py walker is active.
 */
struct py_walk_result {
	__u8 depth;            /* number of valid frames in frames[] */
	__u8 truncated;        /* 1 if walk hit PY_MAX_FRAMES limit */
	__u8 _pad[6];          /* align to 8 */
	struct py_frame frames[PY_MAX_FRAMES];
};

/*
 * Extended CUDA event with userspace stack AND Python frames.
 *
 * When the eBPF Python walker is active AND a frame walk returned
 * non-empty results, emit_event() in cuda_trace.bpf.c reserves this
 * variant instead of cuda_event_stack. The ringbuf record length and
 * the INGERO_EVENT_FLAG_PY_FRAMES bit in hdr._pad let the Go parser
 * dispatch:
 *    80   bytes → cuda_event            (no stack, no python)
 *   600   bytes → cuda_event_stack      (stack only)
 *   4832  bytes → cuda_event_stack_py   (stack + python; flag bit set)
 *
 * py_walk_result is currently 4232 bytes (2 + 6 pad + 16 * 264), so
 * the total wire size is 4832 bytes. Well within BPF ringbuf limits
 * and well over the 512-byte BPF stack limit (allocated via ringbuf
 * reserve, never on the stack).
 */
struct cuda_event_stack_py {
	struct cuda_event_stack base;    /* 600 bytes */
	struct py_walk_result   py;      /* 4232 bytes */
};

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

// Watchdog heartbeat map -- external remediation service writes timestamp
// Probes check staleness to bypass remediation if the service is dead
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u64);  // nanosecond timestamp from bpf_ktime_get_ns()
} ingero_watchdog SEC(".maps");

// ---- Watchdog: remediation service liveness check ----
// If the orchestrator heartbeat is older than this threshold, cudaMalloc and
// cudaFree probes skip event processing. 50ms = 50,000,000 nanoseconds.
// Matches config.toml [watchdog] stale_threshold_ms default.
#define WATCHDOG_STALE_NS 50000000ULL

/*
 * watchdog_is_stale -- returns 1 if orchestrator heartbeat is missing or expired.
 * Reads ingero_watchdog[0] (defined above).
 * Both this and the remediation service use CLOCK_BOOTTIME (bpf_ktime_get_ns).
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
