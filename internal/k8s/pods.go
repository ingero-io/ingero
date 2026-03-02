package k8s

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	// pollInterval is how often we re-list pods from the K8s API.
	// 30s is sufficient for pod enrichment — pods don't churn that fast.
	// v0.8 can switch to the watch API for real-time updates if needed.
	pollInterval = 30 * time.Second

	// waitReadyTimeout is the default max wait for the first pod list.
	waitReadyTimeout = 5 * time.Second
)

// PodInfo holds metadata for a single pod, indexed by container ID.
type PodInfo struct {
	Name         string   // pod name (e.g., "training-job-0")
	Namespace    string   // namespace (e.g., "default")
	ContainerIDs []string // 64-char hex IDs, runtime prefix stripped
	GPURequested bool     // true if any container requests nvidia.com/gpu
	Phase        string   // current phase: Pending, Running, Succeeded, Failed, Unknown
	RestartCount int32    // sum of all container restart counts
}

// PodLifecycleEvent represents a detected pod status transition.
type PodLifecycleEvent struct {
	PodName     string
	Namespace   string
	EventType   string // "restart", "eviction", "oom_kill", "phase_change"
	Detail      string // human-readable detail
	DetectedAt  time.Time
}

// LogFunc is an optional debug logger. Set via PodCache.SetDebugLog().
type LogFunc func(format string, args ...interface{})

// PodCache maintains an in-memory index of pods on this node, keyed by
// container ID for O(1) lookup during event processing.
//
// Lifecycle:
//  1. NewPodCache(client) — creates the cache
//  2. go cache.Run(ctx) — starts polling in a goroutine
//  3. cache.WaitReady(ctx, timeout) — blocks until first list succeeds
//  4. cache.Lookup(containerID) — returns pod info or nil
//
// Run() respects ctx.Done() and stops polling on SIGINT/timeout.
type PodCache struct {
	client *Client
	debugf LogFunc // optional debug logger

	mu       sync.RWMutex
	byContID map[string]*PodInfo // container ID → pod info

	// Pod lifecycle tracking: previous state for transition detection.
	prevState map[string]podSnapshot // key: namespace/name

	// Lifecycle events detected since last drain.
	lifecycleEvents []PodLifecycleEvent

	ready chan struct{} // closed after first successful list
	once  sync.Once     // ensures ready is closed exactly once

	warnedNodeName bool // true after warning about empty MY_NODE_NAME
}

// podSnapshot captures pod state for diff-based transition detection.
type podSnapshot struct {
	Phase        string
	RestartCount int32
}

// NewPodCache creates a cache that will be populated by Run().
func NewPodCache(client *Client) *PodCache {
	return &PodCache{
		client:    client,
		byContID:  make(map[string]*PodInfo),
		prevState: make(map[string]podSnapshot),
		ready:     make(chan struct{}),
	}
}

// SetDebugLog sets an optional debug logger for cache operations.
func (pc *PodCache) SetDebugLog(f LogFunc) {
	pc.debugf = f
}

func (pc *PodCache) logf(format string, args ...interface{}) {
	if pc.debugf != nil {
		pc.debugf(format, args...)
	}
}

// Run polls the K8s API for pods on this node. Call as a goroutine.
// Stops when ctx is cancelled. Refreshes the client token periodically.
func (pc *PodCache) Run(ctx context.Context) {
	// Do an immediate list, then poll on interval.
	pc.refresh()

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	// Token refresh runs less frequently than pod polling.
	tokenTicker := time.NewTicker(tokenRefreshInterval)
	defer tokenTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pc.refresh()
		case <-tokenTicker.C:
			if err := pc.client.RefreshToken(); err != nil {
				pc.logf("K8s token refresh failed: %v", err)
			}
		}
	}
}

// WaitReady blocks until the first pod list succeeds or the timeout/context expires.
// This prevents a race where resolveTargets() runs before the cache is populated.
func (pc *PodCache) WaitReady(ctx context.Context, timeout time.Duration) error {
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case <-pc.ready:
		return nil
	case <-timer.C:
		return fmt.Errorf("pod cache not ready after %v", timeout)
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Lookup returns the PodInfo for a container ID, or nil if not found.
// Container IDs should be 64-char hex (no runtime prefix).
func (pc *PodCache) Lookup(containerID string) *PodInfo {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	return pc.byContID[containerID]
}

// DrainLifecycleEvents returns and clears all accumulated lifecycle events
// since the last drain. Thread-safe.
func (pc *PodCache) DrainLifecycleEvents() []PodLifecycleEvent {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	events := pc.lifecycleEvents
	pc.lifecycleEvents = nil
	return events
}

// GPUPods returns all pods on this node that request nvidia.com/gpu.
func (pc *PodCache) GPUPods() []*PodInfo {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	// Deduplicate: multiple container IDs may point to the same pod.
	// Key by namespace/name (semantic identity) not pointer address,
	// since pointer addresses change across refresh() calls.
	type podKey struct{ ns, name string }
	seen := make(map[podKey]bool)
	var result []*PodInfo
	for _, info := range pc.byContID {
		k := podKey{info.Namespace, info.Name}
		if info.GPURequested && !seen[k] {
			seen[k] = true
			result = append(result, info)
		}
	}
	return result
}

// refresh lists pods from the K8s API and rebuilds the container ID index.
// Also detects pod lifecycle transitions (restarts, evictions, phase changes).
func (pc *PodCache) refresh() {
	pods, err := pc.listPods()
	if err != nil {
		// Don't clear the cache on transient errors — stale data is better
		// than no data for pod enrichment.
		pc.logf("K8s pod list failed: %v (using cached data)", err)
		return
	}

	index := make(map[string]*PodInfo, len(pods))
	for i := range pods {
		for _, cid := range pods[i].ContainerIDs {
			if cid != "" {
				index[cid] = &pods[i]
			}
		}
	}

	pc.mu.Lock()
	pc.byContID = index

	// Detect lifecycle transitions by comparing against previous state.
	now := time.Now()
	newState := make(map[string]podSnapshot, len(pods))
	for _, pod := range pods {
		key := pod.Namespace + "/" + pod.Name
		snap := podSnapshot{Phase: pod.Phase, RestartCount: pod.RestartCount}
		newState[key] = snap

		prev, existed := pc.prevState[key]
		if !existed {
			continue // first time seeing this pod
		}

		// Detect restart: restartCount increased.
		if snap.RestartCount > prev.RestartCount {
			delta := snap.RestartCount - prev.RestartCount
			pc.lifecycleEvents = append(pc.lifecycleEvents, PodLifecycleEvent{
				PodName:    pod.Name,
				Namespace:  pod.Namespace,
				EventType:  "restart",
				Detail:     fmt.Sprintf("container restart detected (count %d → %d, delta %d)", prev.RestartCount, snap.RestartCount, delta),
				DetectedAt: now,
			})
			pc.logf("K8s lifecycle: pod %s/%s restarted (count %d → %d)", pod.Namespace, pod.Name, prev.RestartCount, snap.RestartCount)
		}

		// Detect phase change (e.g., Running → Failed).
		if snap.Phase != prev.Phase && prev.Phase != "" {
			evtType := "phase_change"
			if snap.Phase == "Failed" {
				evtType = "eviction" // Failed phase often indicates eviction
			}
			pc.lifecycleEvents = append(pc.lifecycleEvents, PodLifecycleEvent{
				PodName:    pod.Name,
				Namespace:  pod.Namespace,
				EventType:  evtType,
				Detail:     fmt.Sprintf("phase changed: %s → %s", prev.Phase, snap.Phase),
				DetectedAt: now,
			})
			pc.logf("K8s lifecycle: pod %s/%s phase %s → %s", pod.Namespace, pod.Name, prev.Phase, snap.Phase)
		}
	}
	pc.prevState = newState

	pc.mu.Unlock()

	pc.once.Do(func() { close(pc.ready) })
}

// listPods calls the K8s API to list pods on this node.
// Uses server-side field selector to only fetch pods on our node.
func (pc *PodCache) listPods() ([]PodInfo, error) {
	path := "/api/v1/pods"
	if node := pc.client.NodeName(); node != "" {
		path += "?fieldSelector=spec.nodeName=" + node
	} else if !pc.warnedNodeName {
		// MY_NODE_NAME not set — listing all pods in cluster. This is
		// expensive on large clusters but functionally correct.
		fmt.Fprintf(os.Stderr, "  K8s: warning: MY_NODE_NAME not set, listing all pods (set env.MY_NODE_NAME in DaemonSet)\n")
		pc.warnedNodeName = true
	}

	data, err := pc.client.Get(path)
	if err != nil {
		return nil, err
	}

	return parsePodList(data)
}

// parsePodList extracts PodInfo from a K8s API pod list JSON response.
// Minimal struct definitions — only the fields we need.
func parsePodList(data []byte) ([]PodInfo, error) {
	var resp podListResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("parsing pod list: %w", err)
	}

	var pods []PodInfo
	for _, item := range resp.Items {
		info := PodInfo{
			Name:      item.Metadata.Name,
			Namespace: item.Metadata.Namespace,
			Phase:     item.Status.Phase,
		}

		// Check if any container requests GPU.
		for _, c := range item.Spec.Containers {
			if _, ok := c.Resources.Limits["nvidia.com/gpu"]; ok {
				info.GPURequested = true
				break
			}
		}

		// Extract container IDs and restart count from status.
		var totalRestarts int32
		for _, cs := range item.Status.ContainerStatuses {
			if cid := stripRuntimePrefix(cs.ContainerID); cid != "" {
				info.ContainerIDs = append(info.ContainerIDs, cid)
			}
			totalRestarts += cs.RestartCount
		}
		info.RestartCount = totalRestarts

		pods = append(pods, info)
	}
	return pods, nil
}

// stripRuntimePrefix removes the "containerd://", "docker://", or "cri-o://"
// prefix from a K8s API containerID field. Returns the raw 64-char hex ID
// for comparison with cgroup-parsed container IDs.
//
// Examples:
//
//	"containerd://abc123..." → "abc123..."
//	"docker://abc123..."    → "abc123..."
//	"cri-o://abc123..."     → "abc123..."
//	""                      → ""
func stripRuntimePrefix(containerID string) string {
	if idx := strings.Index(containerID, "://"); idx != -1 {
		return containerID[idx+3:]
	}
	return containerID
}

// --- Minimal K8s API response structs (only fields we need) ---

type podListResponse struct {
	Items []podItem `json:"items"`
}

type podItem struct {
	Metadata podMetadata `json:"metadata"`
	Spec     podSpec     `json:"spec"`
	Status   podStatus   `json:"status"`
}

type podMetadata struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
}

type podSpec struct {
	Containers []container `json:"containers"`
}

type container struct {
	Resources resourceRequirements `json:"resources"`
}

type resourceRequirements struct {
	Limits map[string]interface{} `json:"limits"`
}

type podStatus struct {
	Phase             string            `json:"phase"` // Pending, Running, Succeeded, Failed, Unknown
	ContainerStatuses []containerStatus `json:"containerStatuses"`
}

type containerStatus struct {
	ContainerID  string `json:"containerID"`
	RestartCount int32  `json:"restartCount"`
}
