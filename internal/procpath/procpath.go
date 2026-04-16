// Package procpath provides helpers for accessing files in another process's
// mount namespace via /proc/<pid>/root/. Used when ingero runs inside a
// container and needs to read binaries or libraries belonging to a target
// process whose paths (parsed from /proc/<pid>/maps) refer to the target's
// namespace rather than ingero's.
package procpath

import (
	"fmt"
	"os"
)

// ResolveContainerPath returns a path that ingero can open from its own
// namespace. It first tries path directly (a no-op on bare metal or with
// shared mounts). If that fails, it tries /proc/<pid>/root/<path>, which
// traverses the target's mount namespace via procfs. If both fail, the
// original path is returned so the caller can surface a meaningful error.
//
// Safe to call with pid == 0 or pid == os.Getpid(); in both cases the
// fallback reduces to a no-op or self-reference. No syscalls beyond os.Stat.
//
// TODO(walker-container): when ingero's /proc/self/ns/mnt differs from
// /proc/<pid>/ns/mnt, prefer the /proc/<pid>/root/ path even if the direct
// stat succeeds — otherwise a bind-mounted path that exists in both
// namespaces but points to different files (e.g., two python3.12 binaries)
// would silently pick the wrong one.
func ResolveContainerPath(pid int, path string) string {
	if _, err := os.Stat(path); err == nil {
		return path
	}
	if pid > 0 {
		alt := fmt.Sprintf("/proc/%d/root%s", pid, path)
		if _, err := os.Stat(alt); err == nil {
			return alt
		}
	}
	return path
}
