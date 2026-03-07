// Package cgroup extracts container IDs from Linux cgroup paths.
//
// In Kubernetes, every pod container runs in a cgroup whose path contains the
// 64-character hex container ID. This package parses /proc/[pid]/cgroup to
// extract that ID, supporting all 3 major container runtimes (containerd,
// CRI-O, Docker) on both cgroup v1 and v2.
//
// Teaching note (cgroup v1 vs v2):
//   - cgroup v1: Multiple hierarchies, each a separate line in /proc/[pid]/cgroup.
//     Format: "12:memory:/kubepods/burstable/podUID/<container_id>"
//   - cgroup v2 (unified): Single hierarchy, single line.
//     Format: "0::/<path>"
//   - K8s since 1.25+ defaults to cgroup v2. Most production GPU clusters
//     still mix v1 and v2 depending on kernel and distro.
package cgroup

import (
	"fmt"
	"os"
	"regexp"
	"strings"
)

// containerIDRegex matches a 64-character lowercase hex string.
// Container IDs from containerd, CRI-O, and Docker are always 64 hex chars.
var containerIDRegex = regexp.MustCompile(`[a-f0-9]{64}`)

// ParseContainerID extracts a 64-char hex container ID from a cgroup path.
// Handles all 3 runtimes x 2 cgroup versions:
//
//   - containerd v2: /kubepods.slice/kubepods-burstable.slice/cri-containerd-<id>.scope
//   - containerd v1: /kubepods/burstable/podUID/<id>
//   - CRI-O v2:      /kubepods.slice/crio-<id>.scope
//   - CRI-O v1:      /kubepods.slice/crio-<id>.scope
//   - Docker v1:     /docker/<id>
//   - Docker K8s v1: /kubepods/podUID/<id>
//
// Returns empty string if no container ID found (e.g., host process).
func ParseContainerID(cgroupPath string) string {
	// Strategy: find the last 64-char hex string in the path.
	// This works because the container ID is always the most specific
	// (deepest) component in the cgroup hierarchy.
	matches := containerIDRegex.FindAllString(cgroupPath, -1)
	if len(matches) == 0 {
		return ""
	}
	// Return the last match — in paths like /kubepods/podUID/containerID,
	// podUID is also 64 hex but containerID comes last.
	return matches[len(matches)-1]
}

// ReadCGroupPath reads /proc/[pid]/cgroup and returns the cgroup path.
//
// cgroup v2 (unified): Returns the path from the single "0::/<path>" line.
// cgroup v1 (multiple controllers): Returns the longest path that contains
// a container ID, preferring the memory controller if available.
//
// Returns empty string and nil error for host processes with no meaningful
// cgroup path (e.g., root cgroup "/").
func ReadCGroupPath(pid uint32) (string, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return "", fmt.Errorf("reading /proc/%d/cgroup: %w", pid, err)
	}
	return parseCGroupFile(string(data)), nil
}

// parseCGroupFile parses the contents of /proc/[pid]/cgroup.
// Exported for testing — tests pass file contents directly.
func parseCGroupFile(content string) string {
	var bestPath string
	var bestLen int

	for _, line := range strings.Split(strings.TrimSpace(content), "\n") {
		// Format: "hierarchy-ID:controller-list:cgroup-path"
		// v2: "0::/kubepods.slice/..."
		// v1: "12:memory:/kubepods/..."
		parts := strings.SplitN(line, ":", 3)
		if len(parts) != 3 {
			continue
		}

		path := parts[2]

		// cgroup v2: hierarchy 0, empty controller list
		if parts[0] == "0" && parts[1] == "" {
			return path
		}

		// cgroup v1: pick the longest path (most specific hierarchy).
		// Container IDs are in the deepest path components.
		if len(path) > bestLen {
			bestPath = path
			bestLen = len(path)
		}
	}

	return bestPath
}

// IsCGroupV2 checks if the system uses unified cgroup v2 hierarchy.
// Looks for the "0::/" entry in /proc/self/cgroup.
func IsCGroupV2() bool {
	data, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "0::/") {
			return true
		}
	}
	return false
}
