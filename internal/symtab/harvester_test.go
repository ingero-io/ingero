package symtab

import (
	"os"
	"runtime"
	"strings"
	"testing"
)

func TestFilterLDEnv(t *testing.T) {
	in := []string{
		"PATH=/usr/bin",
		"LD_LIBRARY_PATH=/opt/lib",
		"LD_PRELOAD=/x.so",
		"LD_AUDIT=/y.so",
		"HOME=/root",
		"LDFLAGS=-g", // NOT stripped — no underscore after LD
	}
	got := filterLDEnv(in)
	for _, kv := range got {
		if strings.HasPrefix(kv, "LD_") {
			t.Errorf("filterLDEnv kept LD_-prefixed var: %q", kv)
		}
	}
	// Non-LD_ entries must survive.
	wantKeep := []string{"PATH=/usr/bin", "HOME=/root", "LDFLAGS=-g"}
	for _, w := range wantKeep {
		found := false
		for _, kv := range got {
			if kv == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("filterLDEnv dropped non-LD_ var: %q", w)
		}
	}
}

func TestShouldChrootForPID(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("chroot-for-pid is Linux-only")
	}
	tests := []struct {
		name string
		pid  int
		want bool
	}{
		{name: "pid zero", pid: 0, want: false},
		{name: "negative pid", pid: -1, want: false},
		{
			name: "self pid: same mount namespace",
			pid:  os.Getpid(),
			want: false, // self can't differ from self
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shouldChrootForPID(tt.pid)
			if got != tt.want {
				t.Errorf("shouldChrootForPID(%d) = %v, want %v", tt.pid, got, tt.want)
			}
		})
	}
}

// TestHarvestOffsets_PlainExec is a smoke test that the signature-changed
// HarvestOffsets still runs the no-chroot code path. We invoke it with
// pid=0 (forces plain exec regardless of euid) and a dummy binary. We
// expect a non-nil error, but critically we're exercising the call path,
// not testing actual harvesting.
func TestHarvestOffsets_PlainExec(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("harvester subprocess is Linux-only")
	}
	// /no/such/binary doesn't exist; expect an error, not a panic.
	_, err := HarvestOffsets("/no/such/binary", 0)
	if err == nil {
		t.Error("expected error from harvester with nonexistent binary, got nil")
	}
}

// TestParseHarvesterOutput_SanityCaps verifies that the parser drops
// numeric offsets whose value exceeds a per-field sanity cap. Caps
// defend against the harvester's pointer-chasing heuristics matching
// plausibly-valid but wrong offsets on certain CPython builds (e.g.,
// uv-distributed 3.11.15 / 3.12.13).
func TestParseHarvesterOutput_SanityCaps(t *testing.T) {
	u := func(n uint64) *uint64 { return &n }

	tests := []struct {
		name  string
		input string
		want  HarvestedOffsets
	}{
		{
			name:  "TstateFrame within cap kept",
			input: "TstateFrame=56\n",
			want:  HarvestedOffsets{TstateFrame: u(56)},
		},
		{
			name:  "TstateFrame above cap dropped",
			input: "TstateFrame=512\n",
			want:  HarvestedOffsets{},
		},
		{
			name:  "InterpTstateHead 16 (observed good) kept",
			input: "InterpTstateHead=16\n",
			want:  HarvestedOffsets{InterpTstateHead: u(16)},
		},
		{
			name:  "InterpTstateHead 1048 (observed bad) dropped",
			input: "InterpTstateHead=1048\n",
			want:  HarvestedOffsets{},
		},
		{
			name:  "TstateNativeThreadID above cap dropped",
			input: "TstateNativeThreadID=1000\n",
			want:  HarvestedOffsets{},
		},
		{
			name:  "FrameCode above cap dropped",
			input: "FrameCode=1024\n",
			want:  HarvestedOffsets{},
		},
		{
			name:  "CodeFirstLineNo above cap dropped",
			input: "CodeFirstLineNo=512\n",
			want:  HarvestedOffsets{},
		},
		{
			name: "All capped fields within cap pass through together",
			input: "TstateFrame=56\n" +
				"InterpTstateHead=16\n" +
				"FrameCode=24\n" +
				"CodeFirstLineNo=80\n" +
				"TstateNativeThreadID=32\n",
			want: HarvestedOffsets{
				TstateFrame:          u(56),
				InterpTstateHead:     u(16),
				FrameCode:            u(24),
				CodeFirstLineNo:      u(80),
				TstateNativeThreadID: u(32),
			},
		},
		{
			name:  "Uncapped fields (FrameBack) unaffected by caps",
			input: "FrameBack=99999\n",
			want:  HarvestedOffsets{FrameBack: u(99999)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseHarvesterOutput([]byte(tt.input))
			checkUPtr(t, "TstateFrame", got.TstateFrame, tt.want.TstateFrame)
			checkUPtr(t, "InterpTstateHead", got.InterpTstateHead, tt.want.InterpTstateHead)
			checkUPtr(t, "TstateNativeThreadID", got.TstateNativeThreadID, tt.want.TstateNativeThreadID)
			checkUPtr(t, "FrameCode", got.FrameCode, tt.want.FrameCode)
			checkUPtr(t, "CodeFirstLineNo", got.CodeFirstLineNo, tt.want.CodeFirstLineNo)
			checkUPtr(t, "FrameBack", got.FrameBack, tt.want.FrameBack)
		})
	}
}

func checkUPtr(t *testing.T, name string, got, want *uint64) {
	t.Helper()
	switch {
	case got == nil && want == nil:
		return
	case got == nil && want != nil:
		t.Errorf("%s: got nil, want %d", name, *want)
	case got != nil && want == nil:
		t.Errorf("%s: got %d, want nil (dropped by cap)", name, *got)
	case *got != *want:
		t.Errorf("%s: got %d, want %d", name, *got, *want)
	}
}
