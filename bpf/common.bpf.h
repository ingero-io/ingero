// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
#ifndef __INGERO_COMMON_BPF_H
#define __INGERO_COMMON_BPF_H

/* Shared event types for all eBPF programs */

/* Event source identifiers */
#define EVENT_SRC_CUDA    1
#define EVENT_SRC_NVIDIA  2
#define EVENT_SRC_HOST    3
#define EVENT_SRC_DRIVER  4

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

/* CUDA driver operation types (libcuda.so) */
#define DRIVER_OP_LAUNCH_KERNEL    1
#define DRIVER_OP_MEMCPY           2
#define DRIVER_OP_MEMCPY_ASYNC     3
#define DRIVER_OP_CTX_SYNC         4
#define DRIVER_OP_MEM_ALLOC        5

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

/* Base event header — all events start with this */
struct ingero_event_hdr {
	__u64 timestamp_ns;
	__u32 pid;
	__u32 tid;
	__u8  source;       /* EVENT_SRC_* */
	__u8  op;           /* operation type */
	__u16 _pad;
};

/* CUDA runtime event */
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

/* Host kernel event */
struct host_event {
	struct ingero_event_hdr hdr;
	__u64 duration_ns;
	__u32 cpu;
	__u32 target_pid;    /* for sched events: who was affected */
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
 *   56 bytes  → cuda_event (no stack)
 *   576 bytes → cuda_event_stack (with stack)
 *
 * stack_ips[] is filled by bpf_get_stack(BPF_F_USER_STACK). The helper
 * writes raw instruction pointers (IPs) from the userspace call chain.
 * Symbol resolution happens in Go, not in eBPF.
 *
 * This struct is allocated via bpf_ringbuf_reserve(), NOT on the eBPF
 * stack (which is limited to 512 bytes). 576 bytes in ring buffer is fine.
 */
struct cuda_event_stack {
	struct ingero_event_hdr hdr;       /* 20 bytes */
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
/* Total: 20 + 4(implicit pad) + 8+8+8+4+4 + 2+6+512 = 576 bytes */

#endif /* __INGERO_COMMON_BPF_H */
