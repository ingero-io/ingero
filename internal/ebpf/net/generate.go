// Package net manages eBPF tracepoints for network socket I/O tracing.
//
// Traces socket send/recv syscalls via:
//   - tp/syscalls/sys_enter_sendto + sys_exit_sendto
//   - tp/syscalls/sys_enter_recvfrom + sys_exit_recvfrom
//
// Correlates with CUDA events to identify "GPU idle during network I/O"
// patterns in inference serving and tool-calling agents.
package net

// BPF_TARGET_ARCH is set by the Makefile (x86 or arm64) via: make generate
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -D__TARGET_ARCH_$BPF_TARGET_ARCH -I../../../bpf/headers -I../../../bpf" -target bpfel -type ingero_net_event netTrace ../../../bpf/net_trace.bpf.c
