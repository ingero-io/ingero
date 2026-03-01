package k8s

import (
	"os"
	"strconv"

	"github.com/ingero-io/ingero/internal/cgroup"
)

// FindGPUPodPIDs returns PIDs of processes running in GPU-requesting pods.
// It scans /proc/*/cgroup to find container IDs, then matches them against
// the PodCache's GPU pods.
//
// This is the K8s equivalent of discover.FindCUDAProcesses() — instead of
// scanning /proc/*/maps for libcudart.so, we find PIDs by container membership.
// The two approaches are complementary: FindGPUPodPIDs catches processes that
// haven't loaded CUDA yet (e.g., during initialization).
func FindGPUPodPIDs(cache *PodCache) ([]int, error) {
	// Build a set of container IDs belonging to GPU pods.
	gpuPods := cache.GPUPods()
	if len(gpuPods) == 0 {
		return nil, nil
	}

	gpuContainerIDs := make(map[string]bool)
	for _, pod := range gpuPods {
		for _, cid := range pod.ContainerIDs {
			gpuContainerIDs[cid] = true
		}
	}

	// Scan /proc for processes in GPU pod containers.
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, err
	}

	seen := make(map[int]bool)
	var pids []int
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(entry.Name())
		if err != nil || pid <= 0 {
			continue
		}

		// Read /proc/[pid]/cgroup and extract container ID.
		cgroupPath, err := cgroup.ReadCGroupPath(uint32(pid))
		if err != nil {
			continue
		}
		containerID := cgroup.ParseContainerID(cgroupPath)
		if containerID == "" {
			continue
		}

		// ParseContainerID returns the full 64-char lowercase hex
		// (regex [a-f0-9]{64}). K8s API container IDs are also lowercase.
		if gpuContainerIDs[containerID] && !seen[pid] {
			seen[pid] = true
			pids = append(pids, pid)
		}
	}
	return pids, nil
}
