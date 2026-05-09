// Package kernellaunch is the v0.15 item M loader. Hooks
// libcuda.so cuLaunchKernel and captures grid/block dims.
// Loaded only when --enable-experimental-kprobes is set AND the
// running driver + kernel pair is on internal/kprobe.DefaultAllowlist.
package kernellaunch

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -I../../../bpf/headers -I../../../bpf" -target amd64,arm64 -type kernel_launch_event kernelLaunch ../../../bpf/kernel_launch.bpf.c
