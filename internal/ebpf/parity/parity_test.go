package parity

import (
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/btf"
)

// bpfObjects enumerates the per-arch BPF objects shipped in the agent.
// One entry per internal/ebpf/<pkg>/ directory that has a //go:generate
// directive.
var bpfObjects = []struct {
	pkg  string
	stem string // bpf2go output stem (e.g. cudaTrace -> "cudatrace")
}{
	{"blockio", "iotrace"},
	{"cuda", "cudatrace"},
	{"cudagraph", "cudagraphtrace"},
	{"driver", "drivertrace"},
	{"host", "hosttrace"},
	{"ncclprobe", "nccltrace"},
	{"net", "nettrace"},
	{"tcp", "tcptrace"},
}

// archPair pins the per-arch ELF stem suffix used by bpf2go -target amd64,arm64
// (the linux-arch name suffix bpf2go writes is "x86" for amd64, "arm64" for
// arm64; both with bpfel endianness suffix).
var archPair = []struct {
	arch       string
	suffix     string // file suffix without extension
	expectReg  string // CO-RE relocation target struct that uprobe programs SHOULD use on this arch
	forbidReg  string // CO-RE relocation target struct that MUST NOT appear on this arch (saiyam #35 pattern)
}{
	{arch: "x86", suffix: "x86_bpfel", expectReg: "pt_regs", forbidReg: "user_pt_regs"},
	{arch: "arm64", suffix: "arm64_bpfel", expectReg: "user_pt_regs", forbidReg: "pt_regs"},
}

// TestPerArchObjectsParseCleanly asserts every per-arch .bpf.o is a
// well-formed BPF ELF that cilium/ebpf can parse.
func TestPerArchObjectsParseCleanly(t *testing.T) {
	for _, obj := range bpfObjects {
		for _, a := range archPair {
			path := filepath.Join("..", obj.pkg, obj.stem+"_"+a.suffix+".o")
			t.Run(obj.pkg+"/"+a.arch, func(t *testing.T) {
				if _, err := ebpf.LoadCollectionSpec(path); err != nil {
					t.Fatalf("load %s: %v", path, err)
				}
			})
		}
	}
}

// TestPerArchObjectsHaveStructuralParity asserts that every per-arch pair
// (x86 vs arm64) exposes the same set of programs and maps with the same
// types and sizes. The two compilations differ only in CO-RE relocation
// targets (validated separately in TestPerArchCORERelocationsMatchArch),
// not in user-visible structure.
func TestPerArchObjectsHaveStructuralParity(t *testing.T) {
	for _, obj := range bpfObjects {
		t.Run(obj.pkg, func(t *testing.T) {
			x86 := loadSpec(t, obj.pkg, obj.stem, "x86_bpfel")
			arm := loadSpec(t, obj.pkg, obj.stem, "arm64_bpfel")

			// Program-name parity
			x86Progs := sortedKeys(x86.Programs)
			armProgs := sortedKeys(arm.Programs)
			if !equalStrings(x86Progs, armProgs) {
				t.Errorf("program-name set divergence:\n  x86  : %v\n  arm64: %v", x86Progs, armProgs)
			}

			// Per-program: same Type and SectionName
			for name, xp := range x86.Programs {
				ap, ok := arm.Programs[name]
				if !ok {
					continue
				}
				if xp.Type != ap.Type {
					t.Errorf("program %q: type x86=%v arm64=%v", name, xp.Type, ap.Type)
				}
				if xp.SectionName != ap.SectionName {
					t.Errorf("program %q: section x86=%q arm64=%q", name, xp.SectionName, ap.SectionName)
				}
			}

			// Map-name parity
			x86Maps := sortedKeys(x86.Maps)
			armMaps := sortedKeys(arm.Maps)
			if !equalStrings(x86Maps, armMaps) {
				t.Errorf("map-name set divergence:\n  x86  : %v\n  arm64: %v", x86Maps, armMaps)
			}

			// Per-map: type, key_size, value_size, max_entries
			for name, xm := range x86.Maps {
				am, ok := arm.Maps[name]
				if !ok {
					continue
				}
				if xm.Type != am.Type {
					t.Errorf("map %q: type x86=%v arm64=%v", name, xm.Type, am.Type)
				}
				if xm.KeySize != am.KeySize {
					t.Errorf("map %q: key_size x86=%d arm64=%d", name, xm.KeySize, am.KeySize)
				}
				if xm.ValueSize != am.ValueSize {
					t.Errorf("map %q: value_size x86=%d arm64=%d", name, xm.ValueSize, am.ValueSize)
				}
				if xm.MaxEntries != am.MaxEntries {
					t.Errorf("map %q: max_entries x86=%d arm64=%d", name, xm.MaxEntries, am.MaxEntries)
				}
			}
		})
	}
}

// TestPerArchCORERelocationsMatchArch is the literal regression test for
// issue #35 (DGX Spark CO-RE relocation failure). On an x86 .bpf.o, every
// CO-RE relocation that touches pt_regs/user_pt_regs MUST target the x86
// `pt_regs` struct (with by-name field accessors like `di`, `si`). On an
// arm64 .bpf.o, every such relocation MUST target the arm64
// `user_pt_regs` struct (with `regs[N]` array indices).
//
// The bug at v0.10.0 was that release binaries for both archs embedded a
// single .bpf.o (compiled with -D__TARGET_ARCH_x86 because the host that
// ran `make generate` was x86_64), so the linux_arm64 release tarball
// shipped x86 CO-RE relocations and failed verifier load on every aarch64
// kernel.
//
// Build constraints in the bpf2go-generated _<arch>_bpfel.go shims now
// select the right .o per Go target arch; this test guards the produced
// .o files themselves so any future regression fails CI.
func TestPerArchCORERelocationsMatchArch(t *testing.T) {
	for _, obj := range bpfObjects {
		for _, a := range archPair {
			t.Run(obj.pkg+"/"+a.arch, func(t *testing.T) {
				spec := loadSpec(t, obj.pkg, obj.stem, a.suffix)

				expectMarker := `Struct:"` + a.expectReg + `"`
				forbidMarker := `Struct:"` + a.forbidReg + `"`
				sawExpected := false

				for progName, ps := range spec.Programs {
					for i := range ps.Instructions {
						r := btf.CORERelocationMetadata(&ps.Instructions[i])
						if r == nil {
							continue
						}
						s := r.String()
						if strings.Contains(s, forbidMarker) {
							t.Errorf("[%s/%s] program %q: CO-RE relocation targets %q, which is the wrong arch's struct (issue #35 regression):\n  %s",
								obj.pkg, a.arch, progName, a.forbidReg, s)
						}
						if strings.Contains(s, expectMarker) {
							sawExpected = true
						}
					}
				}

				// Sanity: tracepoint-only objects (io, net, tcp, host) may
				// have zero pt_regs CO-RE relocations. Don't fail on that;
				// just log so the test output makes the picture clear.
				if !sawExpected {
					t.Logf("[%s/%s] no CO-RE relocations against %q (expected for tracepoint-only programs; uprobe programs should hit at least one)",
						obj.pkg, a.arch, a.expectReg)
				}
			})
		}
	}
}

func loadSpec(t *testing.T, pkg, stem, archSuffix string) *ebpf.CollectionSpec {
	t.Helper()
	path := filepath.Join("..", pkg, stem+"_"+archSuffix+".o")
	spec, err := ebpf.LoadCollectionSpec(path)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	return spec
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
