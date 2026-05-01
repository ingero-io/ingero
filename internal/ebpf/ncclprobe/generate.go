// Package ncclprobe attaches eBPF uprobes to NCCL collective entry
// points (ncclCommInitRank, ncclCommDestroy, ncclAllReduce,
// ncclAllGather, ncclReduceScatter, ncclBcast) in libnccl.so or
// statically-linked-NCCL hosts (libtorch_cuda.so etc).
package ncclprobe

// bpf2go compiles once per target arch (amd64, arm64) and injects the
// matching -D__TARGET_ARCH_<arch> automatically; the per-arch .o files
// are selected at Go build time via build constraints in the generated
// _<arch>_bpfel.go shims.
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -I../../../bpf/headers -I../../../bpf" -target amd64,arm64 -type nccl_event ncclTrace ../../../bpf/nccl_trace.bpf.c
