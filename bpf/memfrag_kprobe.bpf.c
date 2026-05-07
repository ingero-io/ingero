// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
//
// v0.15 item K (W1 memfrag IOCTL kprobe).
//
// Hooks `nvidia_unlocked_ioctl(struct file *, unsigned int cmd,
// unsigned long arg)` on the closed NVIDIA driver and emits one
// ringbuf event per invocation. The Go side filters by `cmd` to
// identify memory-related operations.
//
// Honest scope for v0.15:
//   - The BPF program records the IOCTL `cmd` field per event.
//   - Argument-buffer decode (NVOS32_PARAMETERS shape, alloc size,
//     virtual address) is NOT done in v0.15. The user pointer
//     dereferencing requires per-driver-version validation that
//     the v0.15 ship cycle does not have.
//   - The Go side maps `cmd` to a coarse classification: counts
//     by command number per (pid, cgroup_id, cmd).
//
// Validation gate: the agent only loads this program when
// --enable-experimental-kprobes is set AND the running NVIDIA
// driver + Linux kernel pair is on internal/kprobe.DefaultAllowlist.
// Outside the allowlist: warn at startup, do not load.

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#include "common.bpf.h"

char LICENSE[] SEC("license") = "GPL";

// Event shape: fixed-width fields so the Go side reads via
// binary.LittleEndian without a CO-RE bridge. Mirrored at
// internal/ebpf/memfrag/memfrag.go:Event.
struct memfrag_ioctl_event {
	__u64 timestamp_ns;
	__u64 cgroup_id;
	__u32 pid;
	__u32 tgid;
	__u32 cmd;       // raw IOCTL cmd from nvidia_unlocked_ioctl
	__u32 _pad0;
};

// Force BTF emission for the event type so bpf2go's `-type` flag
// can collect it.
const struct memfrag_ioctl_event *_unused_memfrag_ioctl_event_force_btf __attribute__((unused));

// Ringbuf for events; size 256 KiB matches the lower-traffic
// probes (driver, throttle).
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} memfrag_events SEC(".maps");

SEC("kprobe/nvidia_unlocked_ioctl")
int BPF_KPROBE(nvidia_unlocked_ioctl_enter, struct file *filp, unsigned int cmd, unsigned long arg)
{
	struct memfrag_ioctl_event *ev;
	__u64 pid_tgid;

	ev = bpf_ringbuf_reserve(&memfrag_events, sizeof(*ev), 0);
	if (!ev)
		return 0;

	pid_tgid = bpf_get_current_pid_tgid();
	ev->timestamp_ns = bpf_ktime_get_ns();
	ev->cgroup_id    = bpf_get_current_cgroup_id();
	ev->pid          = (__u32)pid_tgid;
	ev->tgid         = (__u32)(pid_tgid >> 32);
	ev->cmd          = cmd;
	ev->_pad0        = 0;

	bpf_ringbuf_submit(ev, 0);
	return 0;
}
