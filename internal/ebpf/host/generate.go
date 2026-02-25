// Package host manages eBPF tracepoints for host kernel event tracing.
//
// This package traces scheduler and memory subsystem events:
//   - sched_switch: CPU preemption (off-CPU duration for target PIDs)
//   - sched_wakeup: thread wakeups for target PIDs
//   - mm_page_alloc: page allocations by target PIDs
//   - oom/mark_victim: OOM killer events (always, no PID filter)
//
// These host events are correlated with CUDA events by the correlation
// engine to explain WHY GPU operations were slow.
package host

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -D__TARGET_ARCH_$BPF_TARGET_ARCH -I../../../bpf/headers -I../../../bpf" -target bpfel -type host_event hostTrace ../../../bpf/host_trace.bpf.c
