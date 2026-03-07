// Package driver manages eBPF uprobes for CUDA Driver API tracing (libcuda.so).
//
// The Driver API (cuLaunchKernel, cuMemcpy*, cuCtxSynchronize) is called directly
// by cuBLAS, cuDNN, and other NVIDIA libraries — bypassing the CUDA Runtime API.
// Without these probes, kernel launches from optimized math libraries are invisible.

package driver

// BPF_TARGET_ARCH is set by the Makefile (x86 or arm64) via: make generate
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -D__TARGET_ARCH_$BPF_TARGET_ARCH -I../../../bpf/headers -I../../../bpf" -target bpfel -type cuda_event driverTrace ../../../bpf/driver_trace.bpf.c
