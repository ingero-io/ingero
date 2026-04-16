// Package pytrace provides Go-side helpers for the in-kernel CPython
// frame walker.
//
// No `go generate` directive lives in this file by design — the walker
// code (bpf/python_walker.bpf.h) is compiled into a peer tracer's BPF
// object (currently internal/ebpf/cuda), which owns the py_runtime_map
// this package operates on. If the walker is later split out into its
// own BPF object, reintroduce a bpf2go directive here.
package pytrace
