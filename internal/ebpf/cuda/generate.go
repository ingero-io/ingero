// Package cuda manages eBPF uprobes for CUDA Runtime API tracing.
package cuda

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -D__TARGET_ARCH_x86 -I../../../bpf/headers -I../../../bpf" -target bpfel -type cuda_event cudaTrace ../../../bpf/cuda_trace.bpf.c
