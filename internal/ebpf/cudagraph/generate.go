// Package cudagraph manages eBPF uprobes for CUDA Graph lifecycle tracing.
package cudagraph

// BPF_TARGET_ARCH is set by the Makefile (x86 or arm64) via: make generate
// Do not run 'go generate' directly — use 'make generate' which auto-detects the architecture.
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -D__TARGET_ARCH_$BPF_TARGET_ARCH -I../../../bpf/headers -I../../../bpf" -target bpfel -type cuda_graph_event cudaGraphTrace ../../../bpf/cuda_graph_trace.bpf.c
