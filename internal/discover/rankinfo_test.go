package discover

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestRankDetection(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("rank detection requires /proc (Linux only)")
	}

	tests := []struct {
		name      string
		environ   string // null-separated key=value pairs
		wantRank  *int
		wantLocal *int
		wantWorld *int
	}{
		{
			name:      "all present",
			environ:   "RANK=3\x00LOCAL_RANK=0\x00WORLD_SIZE=4\x00PATH=/usr/bin\x00",
			wantRank:  intPtr(3),
			wantLocal: intPtr(0),
			wantWorld: intPtr(4),
		},
		{
			name:      "partial — only RANK",
			environ:   "RANK=1\x00PATH=/usr/bin\x00",
			wantRank:  intPtr(1),
			wantLocal: nil,
			wantWorld: nil,
		},
		{
			name:      "absent — no rank vars",
			environ:   "PATH=/usr/bin\x00HOME=/home/user\x00",
			wantRank:  nil,
			wantLocal: nil,
			wantWorld: nil,
		},
		{
			name:      "non-integer RANK ignored",
			environ:   "RANK=abc\x00LOCAL_RANK=0\x00WORLD_SIZE=4\x00",
			wantRank:  nil,
			wantLocal: intPtr(0),
			wantWorld: intPtr(4),
		},
		{
			name:      "empty environ",
			environ:   "",
			wantRank:  nil,
			wantLocal: nil,
			wantWorld: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a fake /proc/<pid>/environ file.
			dir := t.TempDir()
			pid := uint32(99999)
			procDir := filepath.Join(dir, fmt.Sprintf("%d", pid))
			os.MkdirAll(procDir, 0o755)
			os.WriteFile(filepath.Join(procDir, "environ"), []byte(tt.environ), 0o644)

			rc := &RankCache{
				cache:    make(map[uint32]*RankInfo),
				procPath: dir,
			}
			ri := rc.Lookup(pid)

			if !intPtrEqual(ri.Rank, tt.wantRank) {
				t.Errorf("Rank = %v, want %v", fmtIntPtr(ri.Rank), fmtIntPtr(tt.wantRank))
			}
			if !intPtrEqual(ri.LocalRank, tt.wantLocal) {
				t.Errorf("LocalRank = %v, want %v", fmtIntPtr(ri.LocalRank), fmtIntPtr(tt.wantLocal))
			}
			if !intPtrEqual(ri.WorldSize, tt.wantWorld) {
				t.Errorf("WorldSize = %v, want %v", fmtIntPtr(ri.WorldSize), fmtIntPtr(tt.wantWorld))
			}
		})
	}
}

func TestRankCaching(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("rank detection requires /proc (Linux only)")
	}

	dir := t.TempDir()
	pid := uint32(88888)
	procDir := filepath.Join(dir, fmt.Sprintf("%d", pid))
	os.MkdirAll(procDir, 0o755)
	os.WriteFile(filepath.Join(procDir, "environ"), []byte("RANK=5\x00"), 0o644)

	rc := &RankCache{
		cache:    make(map[uint32]*RankInfo),
		procPath: dir,
	}

	// First lookup reads from filesystem.
	ri1 := rc.Lookup(pid)
	if ri1.Rank == nil || *ri1.Rank != 5 {
		t.Fatalf("first lookup: Rank = %v, want 5", fmtIntPtr(ri1.Rank))
	}

	// Modify the file — cached result should be returned.
	os.WriteFile(filepath.Join(procDir, "environ"), []byte("RANK=99\x00"), 0o644)
	ri2 := rc.Lookup(pid)
	if ri2.Rank == nil || *ri2.Rank != 5 {
		t.Errorf("second lookup should be cached: Rank = %v, want 5", fmtIntPtr(ri2.Rank))
	}
}

func TestRankDeadProcess(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("rank detection requires /proc (Linux only)")
	}

	dir := t.TempDir()
	// PID directory doesn't exist — simulates dead process.
	rc := &RankCache{
		cache:    make(map[uint32]*RankInfo),
		procPath: dir,
	}
	ri := rc.Lookup(77777)
	if ri.Rank != nil || ri.LocalRank != nil || ri.WorldSize != nil {
		t.Errorf("dead process should return all nil ranks")
	}
}

func intPtr(v int) *int { return &v }

func intPtrEqual(a, b *int) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return *a == *b
}

func fmtIntPtr(p *int) string {
	if p == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%d", *p)
}
