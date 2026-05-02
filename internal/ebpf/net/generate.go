// Package net manages eBPF tracepoints for network socket I/O tracing.
//
// Traces socket send/recv syscalls via:
//   - tp/syscalls/sys_enter_sendto + sys_exit_sendto
//   - tp/syscalls/sys_enter_recvfrom + sys_exit_recvfrom
//
// Correlates with CUDA events to identify "GPU idle during network I/O"
// patterns in inference serving and tool-calling agents.
package net

// bpf2go compiles once per target arch (amd64, arm64) and injects the
// matching -D__TARGET_ARCH_<arch> automatically.
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -I../../../bpf/headers -I../../../bpf" -target amd64,arm64 -type ingero_net_event netTrace ../../../bpf/net_trace.bpf.c
