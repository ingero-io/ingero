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
	for _, minor := range []int{10, 11, 12} {
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

	if offsets := GetPyOffsets(9); offsets != nil {
		t.Error("expected nil for unsupported version 3.9")
	}
}
