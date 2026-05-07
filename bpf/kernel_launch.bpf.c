// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
//
// v0.15 item M: kernel grid/block dimensions uprobe.
//
// Hooks libcuda.so cuLaunchKernel and emits one ringbuf event per
// launch with the grid (X,Y,Z) and block (X,Y) dimensions. block_z
// requires reading PARM7 from the stack on amd64 / register x6 on
// arm64; libbpf's PT_REGS_PARMx macros only cover PARM1-PARM5
// cross-arch, so block_z is defaulted to 0 here and recovered in a
// v0.15.x follow-up that adds per-arch stack-arg reads.
//
// CUDA Driver API signature (CUresult cuLaunchKernel):
//   PARM1: CUfunction f
//   PARM2: unsigned int gridDimX
//   PARM3: unsigned int gridDimY
//   PARM4: unsigned int gridDimZ
//   PARM5: unsigned int blockDimX
//   PARM6: unsigned int blockDimY  (cross-arch via raw register read)
//
// PARM6 access: libbpf doesn't define PT_REGS_PARM6 cross-arch, so
// we read the platform register directly. On amd64 SysV ABI it is
// r9; on arm64 AAPCS64 it is x5.

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#include "common.bpf.h"

char LICENSE[] SEC("license") = "GPL";

// PARM6 cross-arch raw-register fallback. libbpf does not expose
// PT_REGS_PARM6 because it is not portable across the architectures
// libbpf supports; we constrain ourselves to amd64 + arm64.
#if defined(__TARGET_ARCH_x86)
#define INGERO_PARM6(ctx) ((__u64)(ctx)->r9)
#elif defined(__TARGET_ARCH_arm64)
#define INGERO_PARM6(ctx) ((__u64)((struct user_pt_regs *)(ctx))->regs[5])
#else
#define INGERO_PARM6(ctx) ((__u64)0)
#endif

struct kernel_launch_event {
	__u64 timestamp_ns;
	__u64 cgroup_id;
	__u64 func_handle;
	__u32 pid;
	__u32 tgid;
	__u32 grid_x;
	__u32 grid_y;
	__u32 grid_z;
	__u32 block_x;
	__u32 block_y;
	__u32 _pad0;
};

const struct kernel_launch_event *_unused_kernel_launch_event_force_btf __attribute__((unused));

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1024 * 1024); // 1 MiB; kernel launches are higher-rate than ioctls
} kernel_launch_events SEC(".maps");

SEC("uprobe/cuLaunchKernel")
int uprobe_cu_launch_kernel(struct pt_regs *ctx)
{
	struct kernel_launch_event *ev;
	__u64 pid_tgid;

	ev = bpf_ringbuf_reserve(&kernel_launch_events, sizeof(*ev), 0);
	if (!ev)
		return 0;

	pid_tgid = bpf_get_current_pid_tgid();
	ev->timestamp_ns = bpf_ktime_get_ns();
	ev->cgroup_id    = bpf_get_current_cgroup_id();
	ev->pid          = (__u32)pid_tgid;
	ev->tgid         = (__u32)(pid_tgid >> 32);
	ev->func_handle  = (__u64)PT_REGS_PARM1(ctx);
	ev->grid_x       = (__u32)PT_REGS_PARM2(ctx);
	ev->grid_y       = (__u32)PT_REGS_PARM3(ctx);
	ev->grid_z       = (__u32)PT_REGS_PARM4(ctx);
	ev->block_x      = (__u32)PT_REGS_PARM5(ctx);
	ev->block_y      = (__u32)INGERO_PARM6(ctx);
	ev->_pad0        = 0;

	bpf_ringbuf_submit(ev, 0);
	return 0;
}
