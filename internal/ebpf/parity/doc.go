// Package parity hosts cross-architecture parity assertions for the
// per-arch BPF artifacts produced by bpf2go's `-target amd64,arm64` mode.
//
// The package contains no production code, only tests. It exists as a
// dedicated location so the assertions can read .bpf.o files from sibling
// internal/ebpf/<feature>/ packages without inheriting their build
// constraints.
package parity
