// Package cudagraph manages eBPF uprobes for CUDA Graph lifecycle tracing.
package cudagraph

// bpf2go compiles once per target arch (amd64, arm64) and injects the
// matching -D__TARGET_ARCH_<arch> automatically.
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -I../../../bpf/headers -I../../../bpf" -target amd64,arm64 -type cuda_graph_event cudaGraphTrace ../../../bpf/cuda_graph_trace.bpf.c
