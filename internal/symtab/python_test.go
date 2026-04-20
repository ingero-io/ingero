package symtab

import (
	"testing"
)

func TestDetectPythonFromRegions(t *testing.T) {
	tests := []struct {
		name    string
		regions []MapRegion
		wantVer string
		wantNil bool
	}{
		{
			name: "libpython3.10",
			regions: []MapRegion{
				{Start: 0x7f0000, End: 0x7f1000, Perms: "r-xp", Path: "/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0"},
			},
			wantVer: "3.10",
		},
		{
			name: "libpython3.12 conda",
			regions: []MapRegion{
				{Start: 0x7f0000, End: 0x7f1000, Perms: "r-xp", Path: "/home/user/miniconda3/lib/libpython3.12.so.1.0"},
			},
			wantVer: "3.12",
		},
		{
			name: "python3.11 binary",
			regions: []MapRegion{
				{Start: 0x550000, End: 0x551000, Perms: "r-xp", Path: "/usr/bin/python3.11"},
			},
			wantVer: "3.11",
		},
		{
			name: "no python",
			regions: []MapRegion{
				{Start: 0x7f0000, End: 0x7f1000, Perms: "r-xp", Path: "/usr/lib/libcudart.so.12"},
				{Start: 0x7f1000, End: 0x7f2000, Perms: "r-xp", Path: "/usr/lib/libc.so.6"},
			},
			wantNil: true,
		},
		{
			name:    "empty regions",
			regions: nil,
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := detectPythonFromRegions(tt.regions)
			if tt.wantNil {
				if info != nil {
					t.Errorf("expected nil, got %+v", info)
				}
				return
			}
			if info == nil {
				t.Fatal("expected non-nil PythonInfo")
			}
			if info.Version != tt.wantVer {
				t.Errorf("Version = %q, want %q", info.Version, tt.wantVer)
			}
		})
	}
}

func TestDetectPythonFromRegions_PrefersLibpython(t *testing.T) {
	// When both the python binary and libpython appear in maps,
	// libpython should be preferred (it has the full symbol table).
	regions := []MapRegion{
		{Start: 0x550000, End: 0x551000, Perms: "r-xp", Path: "/usr/bin/python3.10"},
		{Start: 0x7f0000, End: 0x7f1000, Perms: "r-xp", Path: "/home/user/miniconda3/lib/libpython3.10.so.1.0"},
	}

	info := detectPythonFromRegions(regions)
	if info == nil {
		t.Fatal("expected non-nil PythonInfo")
	}
	if info.LibPath != "/home/user/miniconda3/lib/libpython3.10.so.1.0" {
		t.Errorf("expected libpython path, got %q", info.LibPath)
	}
}

func TestDetectPythonFromRegions_FallbackToBinary(t *testing.T) {
	// When only the python binary appears (no libpython — statically linked),
	// the binary path should be returned as fallback.
	regions := []MapRegion{
		{Start: 0x550000, End: 0x551000, Perms: "r-xp", Path: "/usr/bin/python3.10"},
		{Start: 0x7f0000, End: 0x7f1000, Perms: "r-xp", Path: "/usr/lib/libcudart.so.12"},
	}

	info := detectPythonFromRegions(regions)
	if info == nil {
		t.Fatal("expected non-nil PythonInfo")
	}
	if info.LibPath != "/usr/bin/python3.10" {
		t.Errorf("expected binary path as fallback, got %q", info.LibPath)
	}
}

func TestDetectPythonFromRegions_BinaryBeforeLibpython(t *testing.T) {
	// Binary appears first in maps, but libpython appears later.
	// Should still prefer libpython.
	regions := []MapRegion{
		{Start: 0x400000, End: 0x401000, Perms: "r-xp", Path: "/usr/bin/python3.12"},
		{Start: 0x500000, End: 0x501000, Perms: "r-xp", Path: "/usr/lib/libc.so.6"},
		{Start: 0x7f0000, End: 0x7f1000, Perms: "r-xp", Path: "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0"},
	}

	info := detectPythonFromRegions(regions)
	if info == nil {
		t.Fatal("expected non-nil PythonInfo")
	}
	if info.LibPath != "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0" {
		t.Errorf("expected libpython path, got %q", info.LibPath)
	}
}

func TestDetectPythonFromExeTarget(t *testing.T) {
	tests := []struct {
		name     string
		target   string
		wantVer  string
		wantNil  bool
	}{
		{
			name:    "ubuntu system python",
			target:  "/usr/bin/python3.12",
			wantVer: "3.12",
		},
		{
			name:    "uv distribution",
			target:  "/home/ubuntu/.local/share/uv/python/cpython-3.13.13-linux-x86_64-gnu/bin/python3.13",
			wantVer: "3.13",
		},
		{
			name:    "opt/pytorch symlink target",
			target:  "/opt/pytorch/bin/python3.10",
			wantVer: "3.10",
		},
		{
			name:    "new minor version works via regex",
			target:  "/usr/bin/python3.14",
			wantVer: "3.14",
		},
		{
			name:    "no version suffix - rejected",
			target:  "/usr/bin/python3",
			wantNil: true,
		},
		{
			name:    "non-python exe - rejected",
			target:  "/usr/bin/bash",
			wantNil: true,
		},
		{
			name:    "empty target - rejected",
			target:  "",
			wantNil: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := detectPythonFromExeTarget(tt.target)
			if tt.wantNil {
				if got != nil {
					t.Errorf("want nil, got %+v", got)
				}
				return
			}
			if got == nil {
				t.Fatalf("want non-nil PythonInfo for %q", tt.target)
			}
			if got.Version != tt.wantVer {
				t.Errorf("Version = %q, want %q", got.Version, tt.wantVer)
			}
			if got.LibPath != tt.target {
				t.Errorf("LibPath = %q, want %q (should be the exe target)", got.LibPath, tt.target)
			}
		})
	}
}

func TestFindPyRegion(t *testing.T) {
	uvPath := "/home/ubuntu/.local/share/uv/python/cpython-3.14.4-linux-x86_64-gnu/bin/python3.14"
	otherPath := "/tmp/venv-3.14/lib/python3.14/site-packages/_internal/python3.14-shim"
	libcudaPath := "/usr/lib/x86_64-linux-gnu/libcudart.so.12"

	tests := []struct {
		name        string
		regions     []MapRegion
		info        *PythonInfo
		wantPath    string
		wantExact   bool
		wantMatched bool
	}{
		{
			name: "exact path match",
			regions: []MapRegion{
				{Path: libcudaPath},
				{Path: uvPath, Start: 0x400000},
			},
			info:        &PythonInfo{Minor: 14, LibPath: uvPath},
			wantPath:    uvPath,
			wantExact:   true,
			wantMatched: true,
		},
		{
			name: "fallback to equivalent 3.14 region when exact path differs",
			regions: []MapRegion{
				{Path: libcudaPath},
				{Path: otherPath, Start: 0x400000},
			},
			info:        &PythonInfo{Minor: 14, LibPath: uvPath},
			wantPath:    otherPath,
			wantExact:   false,
			wantMatched: true,
		},
		{
			name: "minor mismatch rejects fallback",
			regions: []MapRegion{
				{Path: "/usr/bin/python3.12"},
			},
			info:        &PythonInfo{Minor: 14, LibPath: uvPath},
			wantMatched: false,
		},
		{
			name: "no python region at all",
			regions: []MapRegion{
				{Path: libcudaPath},
				{Path: "/usr/lib/libc.so.6"},
			},
			info:        &PythonInfo{Minor: 14, LibPath: uvPath},
			wantMatched: false,
		},
		{
			name: "exact wins over equivalent when both present",
			regions: []MapRegion{
				{Path: otherPath, Start: 0x300000},
				{Path: uvPath, Start: 0x400000},
			},
			info:        &PythonInfo{Minor: 14, LibPath: uvPath},
			wantPath:    uvPath,
			wantExact:   true,
			wantMatched: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, exact, ok := findPyRegion(tt.regions, tt.info)
			if ok != tt.wantMatched {
				t.Fatalf("matched = %v, want %v", ok, tt.wantMatched)
			}
			if !ok {
				return
			}
			if r.Path != tt.wantPath {
				t.Errorf("Path = %q, want %q", r.Path, tt.wantPath)
			}
			if exact != tt.wantExact {
				t.Errorf("exact = %v, want %v", exact, tt.wantExact)
			}
		})
	}
}

func TestPythonInfo_IsSupportedVersion(t *testing.T) {
	tests := []struct {
		minor int
		want  bool
	}{
		{9, false},
		{10, true},
		{11, true},
		{12, true},
		{13, false},
	}

	for _, tt := range tests {
		info := &PythonInfo{Minor: tt.minor}
		if got := info.IsSupportedVersion(); got != tt.want {
			t.Errorf("Python 3.%d: IsSupportedVersion() = %v, want %v", tt.minor, got, tt.want)
		}
	}
}

func TestGetPyOffsets(t *testing.T) {
	// 3.9 and 3.13/3.14 got scaffolding (dispatcher + offset tables) for
	// the eBPF walker even though the userspace walker (IsSupportedVersion)
	// still gates at 10-12 only. Assert each scaffolded version returns a
	// non-nil table with the critical string-extraction offsets populated.
	for _, minor := range []int{9, 10, 11, 12, 13, 14} {
		offsets := GetPyOffsets(minor)
		if offsets == nil {
			t.Errorf("GetPyOffsets(%d) returned nil", minor)
			continue
		}
		if offsets.RuntimeInterpretersHead == 0 {
			t.Errorf("Python 3.%d: RuntimeInterpretersHead is 0", minor)
		}
		if offsets.CodeFilename == 0 {
			t.Errorf("Python 3.%d: CodeFilename is 0", minor)
		}
		if offsets.CodeName == 0 {
			t.Errorf("Python 3.%d: CodeName is 0", minor)
		}
	}

	// Bracketing sentinels: anything below 3.9 or above 3.14 hits the
	// switch default and MUST return nil so callers skip the ebpf walker.
	if offsets := GetPyOffsets(8); offsets != nil {
		t.Error("expected nil for unsupported version 3.8")
	}
	if offsets := GetPyOffsets(15); offsets != nil {
		t.Error("expected nil for unsupported version 3.15")
	}
}
