// Package memfrag is the v0.15 W1 IOCTL-kprobe loader. Hooks
// nvidia_unlocked_ioctl on the closed NVIDIA driver and counts
// invocations per (cmd, pid, cgroup). Loaded only when the agent
// is run with --enable-experimental-kprobes AND the running
// driver + kernel pair is on internal/kprobe.DefaultAllowlist.
package memfrag

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang-14 -cflags "-O2 -g -Wall -Werror -I../../../bpf/headers -I../../../bpf" -target amd64,arm64 -type memfrag_ioctl_event memfragKprobe ../../../bpf/memfrag_kprobe.bpf.c
