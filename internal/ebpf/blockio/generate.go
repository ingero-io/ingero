// Package blockio manages eBPF tracepoints for block I/O request tracing.
//
// Traces block device I/O (reads, writes, discards) via:
//   - tp/block/block_rq_issue: request dispatched to device driver
//   - tp/block/block_rq_complete: request completed
//
// Measures I/O latency and correlates with CUDA events to identify
// disk bottlenecks (DataLoader, checkpoint writes, model loads).
package blockio

// bpf2go compiles once per target arch (amd64, arm64) and injects the
// matching -D__TARGET_ARCH_<arch> automatically.
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -I../../../bpf/headers -I../../../bpf" -target amd64,arm64 -type ingero_io_event ioTrace ../../../bpf/io_trace.bpf.c
