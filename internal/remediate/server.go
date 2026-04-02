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
	listener   net.Listener
	mu         sync.Mutex
	conn       net.Conn // current consumer connection, nil if none
	dropped    uint64   // messages dropped due to write timeout or no connection
}

// NewServer creates a UDS remediation server.
// If socketPath is empty, defaults to /tmp/ingero-remediate.sock.
func NewServer(socketPath string) *Server {
	if socketPath == "" {
		socketPath = "/tmp/ingero-remediate.sock"
	}
	return &Server{socketPath: socketPath}
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
type typedMessage struct {
	Type           string  `json:"type"`
	PID            uint32  `json:"pid"`
	GPUID          uint32  `json:"gpu_id"`
	AllocatedBytes uint64  `json:"allocated_bytes"`
	TotalVRAM      uint64  `json:"total_vram"`
	UtilizationPct float64 `json:"utilization_pct"`
	LastAllocSize  uint64  `json:"last_alloc_size"`
	TimestampNs    int64   `json:"timestamp_ns"`
}

// Send serializes ms as NDJSON and writes it to the connected consumer.
// Non-blocking: silently drops the message if no client is connected or the
// write exceeds 50ms. Never returns an error.
// The mutex is held for the full write to prevent acceptLoop from replacing
// the connection mid-write. The 50ms deadline bounds the lock duration.
func (s *Server) Send(ms memtrack.MemoryState) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		atomic.AddUint64(&s.dropped, 1)
		return
	}

	msg := typedMessage{
		Type:           "memory",
		PID:            ms.PID,
		GPUID:          ms.GPUID,
		AllocatedBytes: ms.AllocatedBytes,
		TotalVRAM:      ms.TotalVRAM,
		UtilizationPct: ms.UtilizationPct,
		LastAllocSize:  ms.LastAllocSize,
		TimestampNs:    ms.TimestampNs,
	}

	data, err := json.Marshal(msg)
	if err != nil {
		log.Printf("WARN: remediate: marshal_failed error=%v", err)
		return
	}
	data = append(data, '\n')

	s.conn.SetWriteDeadline(time.Now().Add(50 * time.Millisecond))
	_, err = s.conn.Write(data)
	if err != nil {
		atomic.AddUint64(&s.dropped, 1)
		log.Printf("WARN: remediate: write_failed error=%v dropped=%d", err, atomic.LoadUint64(&s.dropped))
		s.conn.Close()
		s.conn = nil
	}
}

// straggleMessage wraps StraggleState with a type discriminator for the UDS protocol.
type straggleMessage struct {
	Type              string   `json:"type"`
	PID               uint32   `json:"pid"`
	ThroughputDropPct float64  `json:"throughput_drop_pct"`
	SchedSwitchCount  uint32   `json:"sched_switch_count"`
	PreemptingPIDs    []uint32 `json:"preempting_pids"`
	TimestampNs       int64    `json:"timestamp_ns"`
}

// SendStraggle serializes a StraggleState as NDJSON with type "straggle" and
// writes it to the connected consumer. Non-blocking: silently drops the message
// if no client is connected or the write exceeds 50ms. Implements straggler.Sink.
func (s *Server) SendStraggle(ss straggler.StraggleState) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		atomic.AddUint64(&s.dropped, 1)
		return nil
	}

	msg := straggleMessage{
		Type:              "straggle",
		PID:               ss.PID,
		ThroughputDropPct: ss.ThroughputDropPct,
		SchedSwitchCount:  ss.SchedSwitchCount,
		PreemptingPIDs:    ss.PreemptingPIDs,
		TimestampNs:       ss.TimestampNs,
	}

	data, err := json.Marshal(msg)
	if err != nil {
		log.Printf("WARN: remediate: straggle_marshal_failed error=%v", err)
		return nil
	}
	data = append(data, '\n')

	s.conn.SetWriteDeadline(time.Now().Add(50 * time.Millisecond))
	_, err = s.conn.Write(data)
	if err != nil {
		atomic.AddUint64(&s.dropped, 1)
		log.Printf("WARN: remediate: straggle_write_failed error=%v dropped=%d", err, atomic.LoadUint64(&s.dropped))
		s.conn.Close()
		s.conn = nil
	}
	return nil
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
