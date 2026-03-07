// Package tcp manages eBPF tracepoints for TCP retransmission tracing.
//
// Traces TCP segment retransmissions via:
//   - tp_btf/tcp_retransmit_skb: TCP segment retransmitted
//
// Correlates with CUDA events to diagnose NCCL hangs and
// network-induced GPU idle periods.
package tcp

// BPF_TARGET_ARCH is set by the Makefile (x86 or arm64) via: make generate
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -D__TARGET_ARCH_$BPF_TARGET_ARCH -I../../../bpf/headers -I../../../bpf" -target bpfel -type ingero_tcp_event tcpTrace ../../../bpf/tcp_trace.bpf.c
