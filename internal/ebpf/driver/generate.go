// Package driver manages eBPF uprobes for CUDA Driver API tracing (libcuda.so).
//
// The Driver API (cuLaunchKernel, cuMemcpy*, cuCtxSynchronize) is called directly
// by cuBLAS, cuDNN, and other NVIDIA libraries — bypassing the CUDA Runtime API.
// Without these probes, kernel launches from optimized math libraries are invisible.

package driver

// bpf2go compiles once per target arch (amd64, arm64) and injects the
// matching -D__TARGET_ARCH_<arch> automatically.
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -I../../../bpf/headers -I../../../bpf" -target amd64,arm64 -type cuda_event driverTrace ../../../bpf/driver_trace.bpf.c
