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

	"github.com/ingero-io/ingero/internal/memtrack"
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
// builds silently ignore unknown JSON fields (Rust serde default behavior,
// verified — no #[serde(deny_unknown_fields)] in ingero-ee/orchestrator/src/),
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
}

// fleetStragglerResolvedMessage marks the straggler->healthy transition.
type fleetStragglerResolvedMessage struct {
	Type      string    `json:"type"`
	NodeID    string    `json:"node_id"`
	ClusterID string    `json:"cluster_id"`
	Timestamp time.Time `json:"timestamp"`
}

// SendFleetStragglerState writes a peer-relative straggler notification
// to the UDS consumer. Non-blocking: drops silently if no consumer is
// connected or the write exceeds 50ms. Wire format follows the existing
// typed-message convention ("type" field first for cheap discrimination
// on the consumer side).
func (s *Server) SendFleetStragglerState(ts time.Time, nodeID, clusterID, detectionMode, dominantSignal string, score, threshold float64) error {
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
	}
	return s.writeLocked(msg)
}

// SendFleetStragglerResolved writes a straggler->healthy edge
// notification. Same non-blocking semantics as SendFleetStragglerState.
func (s *Server) SendFleetStragglerResolved(ts time.Time, nodeID, clusterID string) error {
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
