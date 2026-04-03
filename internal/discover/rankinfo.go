package discover

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// RankInfo holds distributed training rank metadata for a traced process.
// Fields are pointers to distinguish "not set" (nil) from "set to 0".
type RankInfo struct {
	Rank      *int
	LocalRank *int
	WorldSize *int
}

// RankCache caches per-PID rank info. Safe for concurrent use.
type RankCache struct {
	mu       sync.RWMutex
	cache    map[uint32]*RankInfo
	procPath string // override for testing (default "/proc")
}

// NewRankCache creates a new rank detection cache.
func NewRankCache() *RankCache {
	return &RankCache{
		cache:    make(map[uint32]*RankInfo),
		procPath: "/proc",
	}
}

// Lookup returns rank info for the given PID.
// Returns a cached result if available, otherwise reads /proc/[pid]/environ.
// On non-Linux or if the process is gone, returns a zero RankInfo (all nil).
// A nil receiver is safe and returns a zero RankInfo.
func (rc *RankCache) Lookup(pid uint32) *RankInfo {
	if rc == nil {
		return &RankInfo{}
	}
	rc.mu.RLock()
	if ri, ok := rc.cache[pid]; ok {
		rc.mu.RUnlock()
		return ri
	}
	rc.mu.RUnlock()

	ri := rc.detect(pid)

	rc.mu.Lock()
	rc.cache[pid] = ri
	rc.mu.Unlock()

	return ri
}

func (rc *RankCache) detect(pid uint32) *RankInfo {
	if runtime.GOOS != "linux" {
		return &RankInfo{}
	}

	envs, err := rc.readEnviron(pid)
	if err != nil {
		return &RankInfo{}
	}

	ri := &RankInfo{}
	for _, kv := range envs {
		k, v, ok := strings.Cut(kv, "=")
		if !ok {
			continue
		}
		switch k {
		case "RANK":
			if n, err := strconv.Atoi(v); err == nil {
				ri.Rank = &n
			}
		case "LOCAL_RANK":
			if n, err := strconv.Atoi(v); err == nil {
				ri.LocalRank = &n
			}
		case "WORLD_SIZE":
			if n, err := strconv.Atoi(v); err == nil {
				ri.WorldSize = &n
			}
		}
	}
	return ri
}

// readEnviron reads /proc/[pid]/environ and splits by null bytes.
func (rc *RankCache) readEnviron(pid uint32) ([]string, error) {
	path := fmt.Sprintf("%s/%d/environ", rc.procPath, pid)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}
	return strings.Split(strings.TrimRight(string(data), "\x00"), "\x00"), nil
}
