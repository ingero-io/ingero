// Package tcp manages eBPF tracepoints for TCP retransmission tracing.
//
// Traces TCP segment retransmissions via:
//   - tp_btf/tcp_retransmit_skb: TCP segment retransmitted
//
// Correlates with CUDA events to diagnose NCCL hangs and
// network-induced GPU idle periods.
package tcp

// bpf2go compiles once per target arch (amd64, arm64) and injects the
// matching -D__TARGET_ARCH_<arch> automatically.
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -I../../../bpf/headers -I../../../bpf" -target amd64,arm64 -type ingero_tcp_event tcpTrace ../../../bpf/tcp_trace.bpf.c
