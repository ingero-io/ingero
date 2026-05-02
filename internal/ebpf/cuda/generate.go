// Package cuda manages eBPF uprobes for CUDA Runtime API tracing.
package cuda

// bpf2go compiles once per target arch (amd64, arm64) and injects the
// matching -D__TARGET_ARCH_<arch> automatically; the per-arch .o files
// are selected at Go build time via build constraints in the generated
// _<arch>_bpfel.go shims.
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -I../../../bpf/headers -I../../../bpf" -target amd64,arm64 -type cuda_event cudaTrace ../../../bpf/cuda_trace.bpf.c
