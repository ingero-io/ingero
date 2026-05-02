// Package orchestrator detects which workload orchestrator the agent
// is running under (Slurm, Docker, ECS, K8s, or none) and surfaces a
// stable per-process identity for cost-attribution and per-job
// correlation.
//
// v0.12.3 (Roadmap §4.6 P7): multi-orchestrator metadata adapters.
// K8s is already covered by internal/k8s; this package adds the
// non-K8s adapters so Slurm / Docker / ECS deployments aren't
// second-class.
//
// Each detector is a cheap local check (env var or /proc cgroup read);
// no network calls. Detect() returns the FIRST match in priority order:
// K8s -> Slurm -> ECS -> Docker -> none. K8s wins because its agents
// typically run inside Docker too, and we want the K8s identity, not
// the underlying container ID.
package orchestrator

import (
	"os"
	"regexp"
	"strings"
)

// Orchestrator names. Stable values, intentionally short, used as
// resource-attribute string values downstream.
type Orchestrator string

const (
	OrchestratorNone   Orchestrator = ""
	OrchestratorK8s    Orchestrator = "k8s"
	OrchestratorSlurm  Orchestrator = "slurm"
	OrchestratorDocker Orchestrator = "docker"
	OrchestratorECS    Orchestrator = "ecs"
)

// Identity is the per-process orchestrator identity. Orchestrator is
// always set when any detector matched; the per-orchestrator fields are
// populated only for the matching orchestrator. JobID / TaskID /
// ContainerID semantics:
//   - Slurm: JobID = $SLURM_JOB_ID
//   - ECS:   TaskID = task ARN tail; ContainerID = container ID from
//            metadata endpoint when reachable, else from cgroup
//   - Docker: ContainerID = cgroup path tail
//   - K8s:   ContainerID = cgroup container ID; JobID/TaskID empty
type Identity struct {
	Orchestrator Orchestrator
	JobID        string
	TaskID       string
	ContainerID  string
}

// Detect returns the orchestrator identity for the current process.
// Cheap (env var reads + one /proc read); safe to call at startup.
func Detect() Identity {
	if id := detectK8s(); id.Orchestrator != "" {
		return id
	}
	if id := detectSlurm(); id.Orchestrator != "" {
		return id
	}
	if id := detectECS(); id.Orchestrator != "" {
		return id
	}
	if id := detectDocker(); id.Orchestrator != "" {
		return id
	}
	return Identity{}
}

// detectK8s checks for K8s service-account token mount + cgroup
// container ID. The token file is the canonical signal that we're
// inside a K8s pod. Non-K8s containers don't have it.
func detectK8s() Identity {
	const tokenPath = "/var/run/secrets/kubernetes.io/serviceaccount/token"
	if _, err := os.Stat(tokenPath); err != nil {
		return Identity{}
	}
	return Identity{
		Orchestrator: OrchestratorK8s,
		ContainerID:  containerIDFromCgroup(),
	}
}

// detectSlurm checks SLURM_JOB_ID. Slurm sets this in every step's
// environment; absence is conclusive.
func detectSlurm() Identity {
	jid := os.Getenv("SLURM_JOB_ID")
	if jid == "" {
		return Identity{}
	}
	return Identity{
		Orchestrator: OrchestratorSlurm,
		JobID:        jid,
	}
}

// detectECS checks for the ECS task-metadata endpoint env var (set on
// every ECS task since the AWS SDK v3 era). We don't dial the endpoint
// here; the env var alone is conclusive.
func detectECS() Identity {
	uri := os.Getenv("ECS_CONTAINER_METADATA_URI_V4")
	if uri == "" {
		// Fall back to V3 for very old ECS agent versions.
		uri = os.Getenv("ECS_CONTAINER_METADATA_URI")
	}
	if uri == "" {
		return Identity{}
	}
	// TaskID isn't directly exposed via env; the metadata endpoint
	// would yield it but that's a network call. Leave empty here;
	// a caller that wants the ARN can probe the endpoint.
	return Identity{
		Orchestrator: OrchestratorECS,
		ContainerID:  containerIDFromCgroup(),
	}
}

// detectDocker checks the cgroup path for /docker/ or /containerd/
// segments. Plain Docker (without orchestrator) has /docker/<id>.
// containerd-without-K8s has /containerd/<id>. Either qualifies.
func detectDocker() Identity {
	cid := containerIDFromCgroup()
	if cid == "" {
		return Identity{}
	}
	return Identity{
		Orchestrator: OrchestratorDocker,
		ContainerID:  cid,
	}
}

// containerIDFromCgroup extracts the container ID from /proc/self/cgroup
// when the agent is running inside a container. Returns "" on a host.
//
// Recognized layouts:
//   - cgroup v1 Docker: /docker/<64-hex>
//   - cgroup v2 Docker: /system.slice/docker-<64-hex>.scope
//   - containerd:       /system.slice/containerd-<64-hex>.scope
//   - K8s + Docker:     /kubepods.slice/kubepods-burstable.slice/<...>/docker-<id>.scope
//   - K8s + containerd: similar with cri-containerd-<id>.scope
//
// Returns the 64-hex container ID, lowercased, no prefix.
func containerIDFromCgroup() string {
	data, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return ""
	}
	return parseContainerID(string(data))
}

var (
	cgroupHexRE      = regexp.MustCompile(`[0-9a-fA-F]{64}`)
	cgroupShortHexRE = regexp.MustCompile(`[0-9a-fA-F]{12,}`)
)

func parseContainerID(cgroupContent string) string {
	for _, line := range strings.Split(cgroupContent, "\n") {
		if line == "" {
			continue
		}
		// Take the path component (after the second colon for v1, or
		// after the first colon for v2 unified hierarchy).
		parts := strings.SplitN(line, ":", 3)
		if len(parts) < 3 {
			continue
		}
		path := parts[2]
		if m := cgroupHexRE.FindString(path); m != "" {
			return strings.ToLower(m)
		}
	}
	// Fallback: try a shorter hex match (some older Docker installs
	// truncated the ID to 12 chars).
	for _, line := range strings.Split(cgroupContent, "\n") {
		parts := strings.SplitN(line, ":", 3)
		if len(parts) < 3 {
			continue
		}
		if m := cgroupShortHexRE.FindString(parts[2]); m != "" {
			return strings.ToLower(m)
		}
	}
	return ""
}
