// Package blockio manages eBPF tracepoints for block I/O request tracing.
//
// Traces block device I/O (reads, writes, discards) via:
//   - tp_btf/block_rq_issue: request dispatched to device driver
//   - tp_btf/block_rq_complete: request completed
//
// Measures I/O latency and correlates with CUDA events to identify
// disk bottlenecks (DataLoader, checkpoint writes, model loads).
package blockio

// BPF_TARGET_ARCH is set by the Makefile (x86 or arm64) via: make generate
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -D__TARGET_ARCH_$BPF_TARGET_ARCH -I../../../bpf/headers -I../../../bpf" -target bpfel -type ingero_io_event ioTrace ../../../bpf/io_trace.bpf.c
