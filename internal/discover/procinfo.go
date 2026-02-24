// Package discover provides CUDA process detection and system capability checks.
//
// procinfo.go adds process metadata enrichment: reads /proc/<pid>/cmdline
// to display "python3 train.py (PID 4821)" instead of "PID 4821".
//
// Call chain: ProcCache.LookupCommand(pid) called from watch.go, explain.go, mcp/server.go
// when displaying events. Lazily populated on first event per PID.
package discover

import (
	"fmt"
	"os"
	"strings"
	"sync"
)

// ProcCache is a thread-safe PID→command mapping, lazily populated from /proc.
type ProcCache struct {
	mu    sync.RWMutex
	cache map[uint32]string

	// For testing: override /proc base path.
	procPath string
}

// NewProcCache creates a new process metadata cache.
func NewProcCache() *ProcCache {
	return &ProcCache{
		cache:    make(map[uint32]string),
		procPath: "/proc",
	}
}

// LookupCommand returns the command line for the given PID.
// Returns a cached result if available, otherwise reads /proc/<pid>/cmdline.
// Returns "" if the process info cannot be read (process exited, etc.).
func (pc *ProcCache) LookupCommand(pid uint32) string {
	pc.mu.RLock()
	if cmd, ok := pc.cache[pid]; ok {
		pc.mu.RUnlock()
		return cmd
	}
	pc.mu.RUnlock()

	cmd := pc.readCmdline(pid)

	pc.mu.Lock()
	pc.cache[pid] = cmd
	pc.mu.Unlock()

	return cmd
}

// FormatPID returns a human-readable process identifier.
// If command info is available: "python3 train.py (PID 4821)"
// Otherwise: "PID 4821"
func (pc *ProcCache) FormatPID(pid uint32) string {
	cmd := pc.LookupCommand(pid)
	if cmd == "" {
		return fmt.Sprintf("PID %d", pid)
	}
	return fmt.Sprintf("%s (PID %d)", cmd, pid)
}

func (pc *ProcCache) readCmdline(pid uint32) string {
	path := fmt.Sprintf("%s/%d/cmdline", pc.procPath, pid)
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	if len(data) == 0 {
		return ""
	}

	// /proc/pid/cmdline uses NUL as separator between args.
	// Replace NULs with spaces, trim trailing.
	args := strings.Split(strings.TrimRight(string(data), "\x00"), "\x00")

	// Return the first 2 args for concise display (e.g., "python3 train.py").
	switch {
	case len(args) == 0:
		return ""
	case len(args) == 1:
		return basename(args[0])
	default:
		return basename(args[0]) + " " + args[1]
	}
}

// basename returns the last component of a path.
func basename(path string) string {
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[i+1:]
	}
	return path
}
