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
// pid=0 (forces plain exec regardless of euid) and a dummy binary — we
// expect a non-nil error, but critically we're exercising the call path,
// not testing actual harvesting.
func TestHarvestOffsets_PlainExec(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("harvester subprocess is Linux-only")
	}
	// /no/such/binary doesn't exist — expect an error, not a panic.
	_, err := HarvestOffsets("/no/such/binary", 0)
	if err == nil {
		t.Error("expected error from harvester with nonexistent binary, got nil")
	}
}
