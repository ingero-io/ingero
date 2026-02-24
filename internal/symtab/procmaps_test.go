package symtab

import (
	"testing"
)

func TestParseMapsLine(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		wantOK   bool
		wantPath string
		wantExec bool
		wantStart uint64
		wantEnd   uint64
		wantOff   uint64
	}{
		{
			name:      "executable region with path",
			line:      "7f1234000000-7f1234100000 r-xp 00001000 08:01 12345 /usr/lib/x86_64-linux-gnu/libcudart.so.12.1.55",
			wantOK:    true,
			wantPath:  "/usr/lib/x86_64-linux-gnu/libcudart.so.12.1.55",
			wantExec:  true,
			wantStart: 0x7f1234000000,
			wantEnd:   0x7f1234100000,
			wantOff:   0x1000,
		},
		{
			name:      "read-write region",
			line:      "7f1234100000-7f1234200000 rw-p 00100000 08:01 12345 /usr/lib/x86_64-linux-gnu/libcudart.so.12.1.55",
			wantOK:    true,
			wantPath:  "/usr/lib/x86_64-linux-gnu/libcudart.so.12.1.55",
			wantExec:  false,
			wantStart: 0x7f1234100000,
			wantEnd:   0x7f1234200000,
			wantOff:   0x100000,
		},
		{
			name:   "anonymous region (stack)",
			line:   "7ffc12340000-7ffc12360000 rw-p 00000000 00:00 0 [stack]",
			wantOK: true,
			wantPath: "", // [stack] is filtered out
		},
		{
			name:   "anonymous region (heap)",
			line:   "55a123400000-55a123500000 rw-p 00000000 00:00 0 [heap]",
			wantOK: true,
			wantPath: "",
		},
		{
			name:   "vdso",
			line:   "7ffc12380000-7ffc12382000 r-xp 00000000 00:00 0 [vdso]",
			wantOK: true,
			wantPath: "",
		},
		{
			name:     "libpython",
			line:     "7f5678000000-7f5678200000 r-xp 00000000 08:01 54321 /usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0",
			wantOK:   true,
			wantPath: "/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0",
			wantExec: true,
		},
		{
			name:   "too few fields",
			line:   "7f1234000000-7f1234100000 r-xp",
			wantOK: false,
		},
		{
			name:   "empty line",
			line:   "",
			wantOK: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, ok := parseMapsLine(tt.line)
			if ok != tt.wantOK {
				t.Fatalf("parseMapsLine() ok = %v, want %v", ok, tt.wantOK)
			}
			if !ok {
				return
			}
			if r.Path != tt.wantPath {
				t.Errorf("Path = %q, want %q", r.Path, tt.wantPath)
			}
			if tt.wantExec && !r.IsExecutable() {
				t.Errorf("expected executable region")
			}
			if tt.wantStart != 0 && r.Start != tt.wantStart {
				t.Errorf("Start = 0x%x, want 0x%x", r.Start, tt.wantStart)
			}
			if tt.wantEnd != 0 && r.End != tt.wantEnd {
				t.Errorf("End = 0x%x, want 0x%x", r.End, tt.wantEnd)
			}
			if tt.wantOff != 0 && r.Offset != tt.wantOff {
				t.Errorf("Offset = 0x%x, want 0x%x", r.Offset, tt.wantOff)
			}
		})
	}
}

func TestFindRegion(t *testing.T) {
	regions := []MapRegion{
		{Start: 0x1000, End: 0x2000, Path: "/lib/a.so"},
		{Start: 0x3000, End: 0x4000, Path: "/lib/b.so"},
		{Start: 0x5000, End: 0x6000, Path: "/lib/c.so"},
	}

	tests := []struct {
		addr     uint64
		wantPath string
	}{
		{0x1500, "/lib/a.so"},
		{0x1000, "/lib/a.so"}, // start inclusive
		{0x1FFF, "/lib/a.so"}, // end exclusive
		{0x2000, ""},           // boundary: 0x2000 is NOT in region [0x1000, 0x2000)
		{0x3000, "/lib/b.so"},
		{0x5FFF, "/lib/c.so"},
		{0x0FFF, ""},           // before first region
		{0x7000, ""},           // after last region
	}

	for _, tt := range tests {
		r := FindRegion(regions, tt.addr)
		gotPath := ""
		if r != nil {
			gotPath = r.Path
		}
		if gotPath != tt.wantPath {
			t.Errorf("FindRegion(0x%x) = %q, want %q", tt.addr, gotPath, tt.wantPath)
		}
	}
}

func TestMapRegionContains(t *testing.T) {
	r := MapRegion{Start: 0x7f0000, End: 0x7f1000}

	if !r.Contains(0x7f0000) {
		t.Error("should contain start address")
	}
	if !r.Contains(0x7f0500) {
		t.Error("should contain middle address")
	}
	if r.Contains(0x7f1000) {
		t.Error("should not contain end address (exclusive)")
	}
	if r.Contains(0x7effff) {
		t.Error("should not contain address before region")
	}
}
