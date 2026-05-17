package remediate

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/ingero-io/ingero/internal/memtrack"
	"github.com/ingero-io/ingero/internal/nvml"
	"github.com/ingero-io/ingero/internal/straggler"
)

// Server streams MemoryState updates as NDJSON over a Unix domain socket.
// A single consumer connection is supported at a time.
type Server struct {
	socketPath string
	// socketGid is the numeric group id to chown the socket to after bind.
	// When >= 0, the socket is chowned to (-1, socketGid) and chmod'd to
	// 0o770 so a sidecar consumer running under that supplementary group
	// can connect. When < 0 (default), the socket stays owner-only (0o700).
	// Numeric GID is required because distroless images carry no NSS
	// name->gid mapping.
	socketGid int
	listener  net.Listener
	mu        sync.Mutex
	conn      net.Conn // current consumer connection, nil if none
	dropped   uint64   // messages dropped due to write timeout or no connection
	// Per-reason drop counters. Writers hold s.mu while bumping
	// so readers snapshotting via DroppedByReason don't race.
	droppedByReason map[DropReason]uint64
	// rank, worldSize stamp every UDS straggler message with the agent's
	// distributed-training identity when SetRankWorldSize has been called
	// (worldSize > 0). Default zero suppresses the fields via omitempty.
	rank      int
	worldSize int
}

// DropReason classifies why a Send* call did not deliver its payload.
// Callers previously could not tell why a message was dropped — every
// failure path incremented a single atomic and returned nil. The typed
// reason lets callers drive per-reason metrics and alerts.
type DropReason string

const (
	// DropReasonNoClient: no consumer is currently connected.
	DropReasonNoClient DropReason = "no_client"
	// DropReasonMarshalError: json.Marshal failed (should never happen in
	// steady state; indicates a programming bug, not a transient fault).
	DropReasonMarshalError DropReason = "marshal_error"
	// DropReasonWriteError: the socket Write returned an error or hit the
	// 50ms deadline. Covers both hard errors (EPIPE on a consumer that
	// went away) and soft errors (write timeout on a slow reader).
	DropReasonWriteError DropReason = "write_error"
)

// ErrDropped signals that a Send* call did not deliver the payload.
// Callers can use errors.Is(err, ErrDropped) to filter failures without
// parsing the reason string. The typed reason is available via
// ErrDroppedWithReason.
var ErrDropped = errors.New("remediate: message dropped")

// DroppedError carries the reason for a drop along with sentinel support.
// Unwrapping to ErrDropped lets callers tolerate "we have observability
// coverage elsewhere" without switching on the reason.
type DroppedError struct {
	Reason DropReason
}

func (e *DroppedError) Error() string {
	return fmt.Sprintf("remediate: message dropped (reason=%s)", e.Reason)
}
func (e *DroppedError) Unwrap() error { return ErrDropped }

func newDropped(reason DropReason) *DroppedError { return &DroppedError{Reason: reason} }

// NewServer creates a UDS remediation server.
// If socketPath is empty, defaults to /tmp/ingero-remediate.sock.
// The socket is bound with 0o700 (owner-only). Use SetSocketGid to enable
// group-based access for a sidecar consumer.
func NewServer(socketPath string) *Server {
	if socketPath == "" {
		socketPath = "/tmp/ingero-remediate.sock"
	}
	return &Server{
		socketPath:      socketPath,
		socketGid:       -1,
		droppedByReason: map[DropReason]uint64{},
	}
}

// SetSocketGid configures a numeric GID to chown the socket to at bind
// time. When gid >= 0, Start() chowns the socket to (-1, gid) and chmods
// to 0o770 instead of 0o700. Must be called before Start().
func (s *Server) SetSocketGid(gid int) {
	s.socketGid = gid
}

// SetRankWorldSize stamps every outgoing UDS straggler message with the
// agent's distributed-training identity. worldSize > 0 enables the
// stamping (rank must be in [0, worldSize)); worldSize == 0 (the default)
// leaves the rank/world_size JSON fields absent via omitempty. Should be
// called before Start, but is safe to call later as long as the caller
// is single-threaded with respect to Send*.
func (s *Server) SetRankWorldSize(rank, worldSize int) {
	s.rank = rank
	s.worldSize = worldSize
}

// Start binds the Unix domain socket and begins accepting connections.
// Removes any stale socket file before binding.
func (s *Server) Start() error {
	if _, err := os.Stat(s.socketPath); err == nil {
		if err := os.Remove(s.socketPath); err != nil {
			return fmt.Errorf("removing stale socket %s: %w", s.socketPath, err)
		}
	}

	ln, err := net.Listen("unix", s.socketPath)
	if err != nil {
		return fmt.Errorf("listening on UDS %s: %w", s.socketPath, err)
	}
	// Restrict the socket. Without an explicit gid, owner-only (0o700).
	// With a gid configured, chown to (-1, gid) + 0o770 so a sidecar
	// consumer running under that supplementary group can connect. On
	// chown failure (e.g. unprivileged uid, restricted filesystem), fall
	// back to owner-only so the agent still functions — a WARN is emitted
	// and the sidecar will just get EACCES, which is already the caller's
	// symptom without this feature.
	mode := os.FileMode(0o700)
	if s.socketGid >= 0 {
		if err := os.Chown(s.socketPath, -1, s.socketGid); err != nil {
			log.Printf("WARN: remediate: chown_failed path=%s gid=%d error=%v falling_back_to=0700",
				s.socketPath, s.socketGid, err)
		} else {
			mode = 0o770
		}
	}
	if err := os.Chmod(s.socketPath, mode); err != nil {
		ln.Close()
		return fmt.Errorf("chmod socket %s: %w", s.socketPath, err)
	}
	s.listener = ln

	go s.acceptLoop()

	log.Printf("INFO: remediate: server_started path=%s", s.socketPath)
	return nil
}

// acceptLoop runs in a goroutine, accepting one consumer connection at a time.
func (s *Server) acceptLoop() {
	for {
		conn, err := s.listener.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return
			}
			// Also check for the string form — some Go versions don't wrap ErrClosed.
			if isClosedError(err) {
				return
			}
			log.Printf("WARN: remediate: accept_error error=%v", err)
			continue
		}

		s.mu.Lock()
		if s.conn != nil {
			s.conn.Close()
		}
		s.conn = conn
		s.mu.Unlock()

		log.Printf("INFO: remediate: client_connected remote=%s", conn.RemoteAddr())
	}
}

// isClosedError detects "use of closed network connection" errors
// that may not be wrapped as net.ErrClosed.
func isClosedError(err error) bool {
	return err != nil && errors.Is(err, net.ErrClosed)
}

// typedMessage wraps MemoryState with a type discriminator for the UDS protocol.
// The "type" field enables the orchestrator to dispatch by message type.
//
// v0.10: comm carries the kernel-captured process name. Older orchestrator
// builds silently ignore unknown JSON fields (Rust serde default behavior),
// so adding comm is non-breaking on the wire.
type typedMessage struct {
	Type           string  `json:"type"`
	PID            uint32  `json:"pid"`
	Comm           string  `json:"comm,omitempty"`
	GPUID          uint32  `json:"gpu_id"`
	AllocatedBytes uint64  `json:"allocated_bytes"`
	TotalVRAM      uint64  `json:"total_vram"`
	UtilizationPct float64 `json:"utilization_pct"`
	LastAllocSize  uint64  `json:"last_alloc_size"`
	TimestampNs    int64   `json:"timestamp_ns"`
}

// Send serializes ms as NDJSON and writes it to the connected consumer.
// Non-blocking: silently drops the message (bumping per-reason counters)
// if no client is connected, marshal fails, or the 50ms write deadline is
// hit. Signature matches memtrack.Sink (func(MemoryState)). Callers who
// want per-reason drop observability can snapshot via DroppedByReason().
// The mutex is held for the full write to prevent acceptLoop from replacing
// the connection mid-write. The 50ms deadline bounds the lock duration.
func (s *Server) Send(ms memtrack.MemoryState) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		s.bumpDropLocked(DropReasonNoClient)
		return
	}

	msg := typedMessage{
		Type:           "memory",
		PID:            ms.PID,
		Comm:           ms.Comm,
		GPUID:          ms.GPUID,
		AllocatedBytes: ms.AllocatedBytes,
		TotalVRAM:      ms.TotalVRAM,
		UtilizationPct: ms.UtilizationPct,
		LastAllocSize:  ms.LastAllocSize,
		TimestampNs:    ms.TimestampNs,
	}
	_ = s.writeLocked(msg)
}

// straggleMessage wraps StraggleState with a type discriminator for the UDS protocol.
// v0.10: comm carries kernel-captured process name (forward-compatible — see typedMessage).
type straggleMessage struct {
	Type              string   `json:"type"`
	PID               uint32   `json:"pid"`
	Comm              string   `json:"comm,omitempty"`
	ThroughputDropPct float64  `json:"throughput_drop_pct"`
	SchedSwitchCount  uint32   `json:"sched_switch_count"`
	PreemptingPIDs    []uint32 `json:"preempting_pids"`
	TimestampNs       int64    `json:"timestamp_ns"`
	// Sustained distinguishes initial detection (false) from re-emission
	// while sched_switch pressure remains elevated (true). Consumers use
	// this to gate remediation that only applies once per episode.
	Sustained bool `json:"sustained"`
	// EventID is a UUIDv4 generated at SendStraggle time. The cross-layer
	// detector has no parallel OTLP push today, so the id is local to
	// the UDS message; consumers that bridge to OTLP downstream can copy
	// it as `ingero.event.id`.
	EventID string `json:"event_id,omitempty"`
	// Rank, WorldSize identify the agent's distributed-training position.
	// Populated when the Server has been configured via SetRankWorldSize
	// (worldSize > 0). Absent (omitempty) for non-distributed deployments.
	Rank      int `json:"rank,omitempty"`
	WorldSize int `json:"world_size,omitempty"`
	// GPUID is the device index most recently observed launching kernels
	// for this PID. Lets the orchestrator pin the workload to GPU-local
	// NUMA cores instead of hardcoding GPU 0. Encoded as 0 when the
	// detector has not yet seen a CUDA/Driver event for this PID — the
	// consumer treats that as "unknown GPU" and falls back to GPU 0.
	GPUID uint32 `json:"gpu_id"`
}

// SendStraggle serializes a StraggleState as NDJSON with type "straggle" and
// writes it to the connected consumer. Non-blocking: returns *DroppedError
// (unwraps to ErrDropped) on any failure path. Implements straggler.Sink —
// that interface returns error, so callers already tolerate it.
func (s *Server) SendStraggle(ss straggler.StraggleState) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		s.bumpDropLocked(DropReasonNoClient)
		return newDropped(DropReasonNoClient)
	}

	msg := straggleMessage{
		Type:              "straggle",
		PID:               ss.PID,
		Comm:              ss.Comm,
		ThroughputDropPct: ss.ThroughputDropPct,
		SchedSwitchCount:  ss.SchedSwitchCount,
		PreemptingPIDs:    ss.PreemptingPIDs,
		TimestampNs:       ss.TimestampNs,
		Sustained:         ss.Sustained,
		EventID:           uuid.NewString(),
		GPUID:             ss.GPUID,
	}
	if s.worldSize > 0 {
		msg.Rank = s.rank
		msg.WorldSize = s.worldSize
	}
	return s.writeLocked(msg)
}

// fleetStragglerStateMessage is the UDS envelope for agent-side Fleet
// classifications (Story 3.4). Distinct from `straggle` (the local
// cross-layer detector) — this one is peer-relative via Fleet threshold.
type fleetStragglerStateMessage struct {
	Type           string    `json:"type"`
	NodeID         string    `json:"node_id"`
	ClusterID      string    `json:"cluster_id"`
	Score          float64   `json:"score"`
	Threshold      float64   `json:"threshold"`
	DetectionMode  string    `json:"detection_mode"`
	DominantSignal string    `json:"dominant_signal"`
	Timestamp      time.Time `json:"timestamp"`
	// EventID is the agent-generated UUIDv4 for this detection event;
	// the same value appears on the matching OTLP push as the
	// `ingero.event.id` data-point attribute. Consumers correlate the
	// two channels by this id.
	EventID string `json:"event_id,omitempty"`
	// Rank, WorldSize: see straggleMessage.
	Rank      int `json:"rank,omitempty"`
	WorldSize int `json:"world_size,omitempty"`
}

// fleetStragglerResolvedMessage marks the straggler->healthy transition.
type fleetStragglerResolvedMessage struct {
	Type      string    `json:"type"`
	NodeID    string    `json:"node_id"`
	ClusterID string    `json:"cluster_id"`
	Timestamp time.Time `json:"timestamp"`
	// EventID, Rank, WorldSize: see fleetStragglerStateMessage.
	EventID   string `json:"event_id,omitempty"`
	Rank      int    `json:"rank,omitempty"`
	WorldSize int    `json:"world_size,omitempty"`
}

// inferenceOutlierMessage is the UDS envelope for per-workload
// step-duration outliers detected by internal/infer's classifier.
// Emitted only when `ingero trace --inference` is set. Consumers
// (the EE orchestrator, custom operator scripts) decode by the
// `type` field and react however they choose; the FOSS agent's
// only job is publication.
//
// The shape mirrors fleetStragglerStateMessage where it makes sense
// (timestamp, node/cluster ids, event_id for cross-channel
// correlation) and adds workload-key fields (cgroup_path_hash, pid,
// stream_handle) plus the classification verdict (step_duration_ns,
// baseline_p95_ns, bucket).
type inferenceOutlierMessage struct {
	Type           string    `json:"type"`
	NodeID         string    `json:"node_id"`
	ClusterID      string    `json:"cluster_id"`
	Timestamp      time.Time `json:"timestamp"`
	EventID        string    `json:"event_id,omitempty"`
	CGroupPathHash string    `json:"cgroup_path_hash,omitempty"`
	PID            uint32    `json:"pid"`
	StreamHandle   uint64    `json:"stream_handle,omitempty"`
	// Phase is the v0.16.1 phase classification: "prefill" |
	// "decode" | "mixed" | "unknown" | "" (classifier disabled).
	// Backward compatible — pre-v0.16.1 consumers ignore it.
	Phase          string    `json:"phase,omitempty"`
	StepDurationNs int64     `json:"step_duration_ns"`
	BaselineP95Ns  int64     `json:"baseline_p95_ns"`
	BaselineMeanNs int64     `json:"baseline_mean_ns,omitempty"`
	Bucket         string    `json:"bucket"` // "1.5x" | "2x" | "3x"
	Rank           int       `json:"rank,omitempty"`
	WorldSize      int       `json:"world_size,omitempty"`

	// v0.16.3 contextual fields. All optional and backward-compatible:
	// pre-v0.16.3 consumers ignore unknown JSON fields.
	//
	// MemfragEventsInStep is the count of NVIDIA closed-driver IOCTL
	// events (KV-cache eviction or fragmenting allocation) observed
	// between the previous and current syncs. A non-zero value on a
	// decode-shape outlier flags VRAM pressure as the proximate cause.
	//
	// ThrottleReasons is the OR-fold of NVML clock-throttle bitmaps
	// observed during the step. Bit semantics follow NVML
	// nvmlClocksThrottleReasons (see open-gpu-kernel-modules headers).
	// A non-zero value means a thermal/power slowdown coincided with
	// the slow step; combined with the agent's existing
	// gpu.throttle.{power,thermal,sw,hw}.event_total counters,
	// operators can tell apart "outlier caused by throttle" from
	// "throttle elsewhere".
	//
	// MinSMClockMHz is reserved for a future v0.16.x extension that
	// pulls SM clock from the existing throttle poller; populated as 0
	// today.
	MemfragEventsInStep uint32 `json:"memfrag_events_in_step,omitempty"`
	ThrottleReasons     uint64 `json:"throttle_reasons,omitempty"`
	MinSMClockMHz       uint32 `json:"min_sm_clock_mhz,omitempty"`

	// KVCacheTopAllocAgesMs is the per-decode-outlier alloc-age
	// context: top-N oldest live cudaMalloc ages in milliseconds,
	// sorted oldest-first. Empty when the engine ran without
	// --inference-kvcache-lineage, or for non-decode outliers, or
	// when no live allocations were tracked for the PID (typical for
	// the first few seconds of the workload before its first
	// cudaMalloc). Backward-compatible: older consumers ignore unknown
	// JSON fields.
	KVCacheTopAllocAgesMs []uint64 `json:"kv_cache_top_alloc_ages_ms,omitempty"`
}

// inferenceSamplerDegradedMessage is the UDS envelope for the engine's
// flip-to-degraded edge. v0.16.3 sibling to inferenceOutlierMessage.
// Fired once per cooldown window so a back-to-back trigger after
// cooldown produces a second message; consumers can de-dup by
// (cgroup_path_hash, pid, cooldown_end) if needed.
//
// Cause is the human-friendly summary
// ("3x:cgroup=<hash>,pid=<n>,phase=<p>") that also lands as the
// AttrInferSamplerCause attribute on the gauge / counter side. Bucket
// is the bucket the OUTLIER WAS IN (not the configured threshold);
// CooldownEnd is when the sampler will return to healthy admission
// absent another trigger.
type inferenceSamplerDegradedMessage struct {
	Type           string    `json:"type"`
	NodeID         string    `json:"node_id"`
	ClusterID      string    `json:"cluster_id"`
	Timestamp      time.Time `json:"timestamp"`
	CGroupPathHash string    `json:"cgroup_path_hash,omitempty"`
	PID            uint32    `json:"pid"`
	StreamHandle   uint64    `json:"stream_handle,omitempty"`
	Phase          string    `json:"phase,omitempty"`
	Bucket         string    `json:"bucket"`
	Cause          string    `json:"cause"`
	CooldownEnd    time.Time `json:"cooldown_end"`
	Rank           int       `json:"rank,omitempty"`
	WorldSize      int       `json:"world_size,omitempty"`
}

// SendFleetStragglerState writes a peer-relative straggler notification
// to the UDS consumer. Non-blocking: drops silently if no consumer is
// connected or the write exceeds 50ms. Wire format follows the existing
// typed-message convention ("type" field first for cheap discrimination
// on the consumer side). eventID, when non-empty, is written as the
// `event_id` JSON field and matches the `ingero.event.id` attribute on
// the parallel OTLP push for the same detection event.
func (s *Server) SendFleetStragglerState(ts time.Time, nodeID, clusterID, detectionMode, dominantSignal, eventID string, score, threshold float64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		s.bumpDropLocked(DropReasonNoClient)
		return newDropped(DropReasonNoClient)
	}

	msg := fleetStragglerStateMessage{
		Type:           "straggler_state",
		NodeID:         nodeID,
		ClusterID:      clusterID,
		Score:          score,
		Threshold:      threshold,
		DetectionMode:  detectionMode,
		DominantSignal: dominantSignal,
		Timestamp:      ts,
		EventID:        eventID,
	}
	if s.worldSize > 0 {
		msg.Rank = s.rank
		msg.WorldSize = s.worldSize
	}
	return s.writeLocked(msg)
}

// InferenceOutlier captures one classified step-duration outlier for
// publication on the UDS socket. Struct-shaped so v0.16.3 can add
// contextual fields (memfrag, throttle, SM clock) without growing a
// positional argument list. Bucket is one of "1.5x" | "2x" | "3x" -
// the largest baseline-p95 multiplier the step crossed. EventID is the
// UUID that appears on the matching OTLP histogram data-point so
// consumers can correlate channels.
type InferenceOutlier struct {
	Timestamp      time.Time
	NodeID         string
	ClusterID      string
	EventID        string
	CGroupPathHash string
	PID            uint32
	StreamHandle   uint64
	Phase          string
	StepDurationNs int64
	BaselineP95Ns  int64
	BaselineMeanNs int64
	Bucket         string

	MemfragEventsInStep uint32
	ThrottleReasons     uint64
	MinSMClockMHz       uint32

	// KVCacheTopAllocAgesMs is the alloc-age context. See
	// inferenceOutlierMessage.KVCacheTopAllocAgesMs for the wire
	// shape. Empty for non-decode outliers and when the engine ran
	// without --inference-kvcache-lineage.
	KVCacheTopAllocAgesMs []uint64
}

// SendInferenceOutlier writes a per-workload step-duration outlier
// notification to the UDS consumer. Non-blocking; same drop semantics
// as SendFleetStragglerState.
//
// Wire-protocol-only. The FOSS agent classifies and publishes; any
// follow-up action lives in the consumer (notably the ingero-ee
// orchestrator, but the protocol is open).
func (s *Server) SendInferenceOutlier(o InferenceOutlier) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		s.bumpDropLocked(DropReasonNoClient)
		return newDropped(DropReasonNoClient)
	}

	msg := inferenceOutlierMessage{
		Type:                  "inference_outlier",
		NodeID:                o.NodeID,
		ClusterID:             o.ClusterID,
		Timestamp:             o.Timestamp,
		EventID:               o.EventID,
		CGroupPathHash:        o.CGroupPathHash,
		PID:                   o.PID,
		StreamHandle:          o.StreamHandle,
		Phase:                 o.Phase,
		StepDurationNs:        o.StepDurationNs,
		BaselineP95Ns:         o.BaselineP95Ns,
		BaselineMeanNs:        o.BaselineMeanNs,
		Bucket:                o.Bucket,
		MemfragEventsInStep:   o.MemfragEventsInStep,
		ThrottleReasons:       o.ThrottleReasons,
		MinSMClockMHz:         o.MinSMClockMHz,
		KVCacheTopAllocAgesMs: o.KVCacheTopAllocAgesMs,
	}
	if s.worldSize > 0 {
		msg.Rank = s.rank
		msg.WorldSize = s.worldSize
	}
	return s.writeLocked(msg)
}

// InferenceSamplerDegraded captures one flip-to-degraded edge from
// the engine's sampler observability path. v0.16.3 sibling to
// InferenceOutlier; same wire-protocol-only contract.
type InferenceSamplerDegraded struct {
	Timestamp      time.Time
	NodeID         string
	ClusterID      string
	CGroupPathHash string
	PID            uint32
	StreamHandle   uint64
	Phase          string
	Bucket         string
	Cause          string
	CooldownEnd    time.Time
}

// SendInferenceSamplerDegraded writes a sampler-degraded edge
// notification. Non-blocking; same drop semantics as SendInferenceOutlier.
func (s *Server) SendInferenceSamplerDegraded(d InferenceSamplerDegraded) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		s.bumpDropLocked(DropReasonNoClient)
		return newDropped(DropReasonNoClient)
	}

	msg := inferenceSamplerDegradedMessage{
		Type:           "inference_sampler_degraded",
		NodeID:         d.NodeID,
		ClusterID:      d.ClusterID,
		Timestamp:      d.Timestamp,
		CGroupPathHash: d.CGroupPathHash,
		PID:            d.PID,
		StreamHandle:   d.StreamHandle,
		Phase:          d.Phase,
		Bucket:         d.Bucket,
		Cause:          d.Cause,
		CooldownEnd:    d.CooldownEnd,
	}
	if s.worldSize > 0 {
		msg.Rank = s.rank
		msg.WorldSize = s.worldSize
	}
	return s.writeLocked(msg)
}

// hardwareFaultMessage is the UDS envelope for an NVML/Xid hardware
// fault produced by the agent's nvml probes (e.g. ThermalSustainTracker
// for kind=thermal_throttle, the Xid event reader for kind=xid). The
// EE-side dispatch in `orchestrator/src/uds.rs` matches on `type` then
// routes severity=critical events into the node_cordon + pod_drain
// playbook (Phase 13); warnings are counter-only.
//
// Wire shape mirrors the EE-side `HardwareFaultState` deserializer.
// Adding new optional fields is non-breaking because the EE side uses
// serde defaults for unknown JSON fields. `throttle_reasons` is one
// such forward-compat field: the agent emits it for log context, the
// orchestrator currently ignores it and would consume it once a future
// arm wants per-bitmask routing.
type hardwareFaultMessage struct {
	Type            string    `json:"type"`
	NodeID          string    `json:"node_id"`
	ClusterID       string    `json:"cluster_id"`
	Timestamp       time.Time `json:"timestamp"`
	EventID         string    `json:"event_id,omitempty"`
	Kind            string    `json:"kind"`
	Severity        string    `json:"severity"`
	XidNumber       uint32    `json:"xid_number,omitempty"`
	GPUID           uint32    `json:"gpu_id"`
	PID             uint32    `json:"pid,omitempty"`
	ThrottleReasons uint64    `json:"throttle_reasons,omitempty"`
	Rank            int       `json:"rank,omitempty"`
	WorldSize       int       `json:"world_size,omitempty"`
}

// SendHardwareFault writes an NVML/Xid hardware-fault notification to
// the UDS consumer. Non-blocking; same drop semantics as
// SendInferenceOutlier. The producer-facing input is `nvml.HardwareFault`
// (e.g., the value emitted by ThermalSustainTracker.Observe); the agent's
// node/cluster identity is passed separately because nvml is a
// hardware-only package and does not carry that identity.
//
// If fault.Timestamp is zero, the current UTC time is stamped. If
// fault.EventID is empty, a fresh UUIDv4 is generated so the matching
// OTLP push (when one is added) can correlate channels by id.
//
// Wire-protocol-only contract: the FOSS agent classifies and publishes;
// any auto-action lives in the consumer (notably the ingero-ee
// orchestrator, which gates dispatch on severity == "critical").
func (s *Server) SendHardwareFault(fault nvml.HardwareFault, nodeID, clusterID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		s.bumpDropLocked(DropReasonNoClient)
		return newDropped(DropReasonNoClient)
	}

	ts := fault.Timestamp
	if ts.IsZero() {
		ts = time.Now().UTC()
	}
	eventID := fault.EventID
	if eventID == "" {
		eventID = uuid.NewString()
	}

	msg := hardwareFaultMessage{
		Type:            "hardware_fault",
		NodeID:          nodeID,
		ClusterID:       clusterID,
		Timestamp:       ts,
		EventID:         eventID,
		Kind:            string(fault.Kind),
		Severity:        string(fault.Severity),
		XidNumber:       fault.XidNumber,
		GPUID:           fault.GPUID,
		PID:             fault.PID,
		ThrottleReasons: fault.ThrottleReasons,
	}
	if s.worldSize > 0 {
		msg.Rank = s.rank
		msg.WorldSize = s.worldSize
	}
	return s.writeLocked(msg)
}

// tcpRetransmitStormMessage is the UDS envelope for a per-PID TCP
// retransmit storm detected by internal/tcpretransmit's sustain
// tracker. The tracker aggregates `tcp_retransmit_skb` BTF
// tracepoint events (shipped in internal/ebpf/tcp) into a per-PID
// rolling rate and emits one storm per episode.
//
// EE-side dispatch in `orchestrator/src/uds.rs` routes this through
// the TcpRetransmitStorm chain (drain_lb_endpoint -> pod_drain):
// shift traffic away first, then evict + reschedule if the
// retransmit rate stays elevated.
type tcpRetransmitStormMessage struct {
	Type        string    `json:"type"`
	NodeID      string    `json:"node_id"`
	ClusterID   string    `json:"cluster_id"`
	Timestamp   time.Time `json:"timestamp"`
	EventID     string    `json:"event_id,omitempty"`
	PID         uint32    `json:"pid"`
	RatePerSec  float64   `json:"rate_per_sec"`
	SustainedMs uint64    `json:"sustained_ms"`
	Rank        int       `json:"rank,omitempty"`
	WorldSize   int       `json:"world_size,omitempty"`
}

// TcpRetransmitStorm is the producer-facing input for a per-PID
// retransmit-storm publication. Mirrors the SustainTracker.Storm
// output in internal/tcpretransmit but is kept structurally distinct
// so that package can evolve its internal representation without
// breaking the wire-facing contract here.
type TcpRetransmitStorm struct {
	PID         uint32
	RatePerSec  float64
	SustainedMs uint64
}

// SendTcpRetransmitStorm writes a per-PID retransmit-storm
// notification to the UDS consumer. Non-blocking; same drop semantics
// as SendInferenceOutlier. The agent's node/cluster identity is
// passed separately because the tcpretransmit package is hardware /
// kernel only and does not carry that identity.
//
// Wire-protocol-only contract: the FOSS agent classifies and
// publishes; the orchestrator dispatches drain_lb_endpoint +
// pod_drain (in chain) when configured to act.
func (s *Server) SendTcpRetransmitStorm(storm TcpRetransmitStorm, nodeID, clusterID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		s.bumpDropLocked(DropReasonNoClient)
		return newDropped(DropReasonNoClient)
	}

	msg := tcpRetransmitStormMessage{
		Type:        "tcp_retransmit_storm",
		NodeID:      nodeID,
		ClusterID:   clusterID,
		Timestamp:   time.Now().UTC(),
		EventID:     uuid.NewString(),
		PID:         storm.PID,
		RatePerSec:  storm.RatePerSec,
		SustainedMs: storm.SustainedMs,
	}
	if s.worldSize > 0 {
		msg.Rank = s.rank
		msg.WorldSize = s.worldSize
	}
	return s.writeLocked(msg)
}

// SendFleetStragglerResolved writes a straggler->healthy edge
// notification. Same non-blocking semantics as SendFleetStragglerState.
// eventID is per-recovery-edge; consumers may correlate it with the OTLP
// data-point attribute on the matching push.
func (s *Server) SendFleetStragglerResolved(ts time.Time, nodeID, clusterID, eventID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		s.bumpDropLocked(DropReasonNoClient)
		return newDropped(DropReasonNoClient)
	}

	msg := fleetStragglerResolvedMessage{
		Type:      "straggler_resolved",
		NodeID:    nodeID,
		ClusterID: clusterID,
		Timestamp: ts,
		EventID:   eventID,
	}
	if s.worldSize > 0 {
		msg.Rank = s.rank
		msg.WorldSize = s.worldSize
	}
	return s.writeLocked(msg)
}

// writeLocked marshals msg and writes it to the current connection.
// Caller must hold s.mu. Drops the connection on write failure to match
// the existing Send/SendStraggle pattern. Returns a typed *DroppedError
// on any failure so callers can distinguish delivered vs dropped.
func (s *Server) writeLocked(msg any) error {
	data, err := json.Marshal(msg)
	if err != nil {
		log.Printf("WARN: remediate: marshal_failed error=%v", err)
		s.bumpDropLocked(DropReasonMarshalError)
		return newDropped(DropReasonMarshalError)
	}
	data = append(data, '\n')

	s.conn.SetWriteDeadline(time.Now().Add(50 * time.Millisecond))
	_, err = s.conn.Write(data)
	if err != nil {
		s.bumpDropLocked(DropReasonWriteError)
		log.Printf("WARN: remediate: write_failed error=%v dropped=%d", err, atomic.LoadUint64(&s.dropped))
		s.conn.Close()
		s.conn = nil
		return newDropped(DropReasonWriteError)
	}
	return nil
}

// bumpDropLocked increments both the legacy aggregate atomic counter and
// the per-reason map. Caller must hold s.mu.
func (s *Server) bumpDropLocked(reason DropReason) {
	atomic.AddUint64(&s.dropped, 1)
	s.droppedByReason[reason]++
}

// DroppedByReason returns a snapshot of per-reason drop counters. Safe to
// call from any goroutine. Intended for a /metrics endpoint or a periodic
// log-emitter.
func (s *Server) DroppedByReason() map[DropReason]uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[DropReason]uint64, len(s.droppedByReason))
	for k, v := range s.droppedByReason {
		out[k] = v
	}
	return out
}

// Close stops the server, closes any active connection, and removes the socket file.
// Safe to call even if Start() was never called or failed.
func (s *Server) Close() error {
	if s.listener != nil {
		s.listener.Close()
	}

	s.mu.Lock()
	if s.conn != nil {
		s.conn.Close()
		s.conn = nil
	}
	s.mu.Unlock()

	os.Remove(s.socketPath)
	log.Printf("INFO: remediate: server_stopped path=%s", s.socketPath)
	return nil
}

// Dropped returns the number of messages dropped due to no client or write errors.
// Safe to call from any goroutine.
func (s *Server) Dropped() uint64 {
	return atomic.LoadUint64(&s.dropped)
}
