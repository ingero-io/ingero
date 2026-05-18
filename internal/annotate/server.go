// Package annotate implements the inbound annotation ingest socket.
//
// The `trace` process listens on a local Unix-domain socket and accepts
// newline-delimited JSON annotation lines. Each accepted line is
// validated, resolved to a process incarnation, stamped with the
// writer's SO_PEERCRED provenance, and handed to a Sink (the agent's
// SQLite store).
//
// This is an INBOUND trust boundary. It is deliberately NOT modelled on
// internal/remediate's outbound server:
//
//   - The remediation socket is read-only egress; this one is
//     write-ingress into the trace store, so every input is validated.
//   - The remediation server accepts one consumer connection at a time;
//     this server accepts many, one goroutine per connection, because
//     the motivating callers (Lightning multi-worker, Ray) write
//     concurrently.
//   - The socket path is fixed inside an agent-owned directory, not an
//     operator flag, so the agent never unlinks an arbitrary path at
//     bind.
//
// The socket is 0o700 owner-only by default; group access is a
// documented opt-in via SetSocketGid, mirroring the remediation
// server's SetSocketGid.
package annotate

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sys/unix"

	"github.com/ingero-io/ingero/pkg/annotate"
	"github.com/ingero-io/ingero/pkg/contract"
)

// Sink receives validated, provenance-stamped annotations. The agent's
// SQLite store satisfies it via (*store.Store).RecordAnnotation.
type Sink interface {
	RecordAnnotation(a annotate.Annotation) error
}

// incarnationResolver turns a bare PID into a process incarnation by
// reading the process start time. Pulled behind an interface so tests
// can supply a deterministic resolver without spawning processes.
type incarnationResolver func(pid uint32) annotate.ProcessIncarnation

// Server is the inbound annotation ingest UDS server.
type Server struct {
	socketDir  string
	socketPath string

	// socketGid, when >= 0, is chowned onto the socket after bind and
	// the mode is widened to 0o770 so a writer in that supplementary
	// group can connect. < 0 (default) keeps the socket 0o700
	// owner-only. Numeric GID: distroless images carry no NSS mapping.
	socketGid int

	sink     Sink
	resolver incarnationResolver

	// pidUID resolves a PID to its real uid. Pulled behind a field,
	// like resolver, so tests can supply a deterministic lookup
	// without depending on live /proc state. NewServer defaults it to
	// readPIDRealUID.
	pidUID func(uint32) (uint32, bool)

	listener net.Listener

	mu     sync.Mutex
	conns  map[net.Conn]struct{}
	closed bool

	// Counters, all atomic so a /metrics reader or a periodic log
	// emitter can snapshot them from any goroutine.
	accepted  atomic.Uint64 // connections accepted
	ingested  atomic.Uint64 // annotation rows accepted into the sink
	rejected  atomic.Uint64 // lines rejected (malformed, oversized, invalid)
	sinkError atomic.Uint64 // lines that validated but the sink rejected
}

// NewServer creates an ingest server writing accepted annotations to
// sink. The socket lives in an agent-owned directory: contract.
// AnnotationSocketDir (/run/ingero) when /run is writable, otherwise a
// 0o700 directory the agent owns under the user's home. Either way the
// directory is off the world-writable /tmp so a local attacker cannot
// pre-create it and hijack the socket.
func NewServer(sink Sink) *Server {
	dir := resolveSocketDir()
	return &Server{
		socketDir:  dir,
		socketPath: filepath.Join(dir, contract.AnnotationSocketName),
		socketGid:  -1,
		sink:       sink,
		resolver:   annotate.ResolveIncarnation,
		pidUID:     readPIDRealUID,
		conns:      make(map[net.Conn]struct{}),
	}
}

// SocketPath returns the agent-owned ingest socket path the server
// binds and a client should dial. It resolves the same directory the
// server uses, so the `ingero annotate` client never needs an
// operator-supplied socket flag.
func SocketPath() string {
	return filepath.Join(resolveSocketDir(), contract.AnnotationSocketName)
}

// resolveSocketDir picks the agent-owned directory for the ingest
// socket. It prefers contract.AnnotationSocketDir (under /run, not
// /tmp); when /run is not writable - an unprivileged trace, a
// distroless image with a read-only /run - it falls back to a 0o700
// directory under the user's home. Both roots are off the
// world-writable /tmp.
func resolveSocketDir() string {
	parent := filepath.Dir(contract.AnnotationSocketDir) // "/run"
	if fi, err := os.Stat(parent); err == nil && fi.IsDir() {
		if unix.Access(parent, unix.W_OK) == nil {
			return contract.AnnotationSocketDir
		}
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".ingero", "annotate")
	}
	// Last resort: a uid-qualified path under /run owned by the agent.
	return fmt.Sprintf("%s-%d", contract.AnnotationSocketDir, os.Geteuid())
}

// verifyAgentOwnedDir asserts that dir is a directory the agent can
// safely own the ingest socket inside: it must be a real directory
// (not a symlink), owned by the agent's effective uid, and carry no
// group- or other-write bits. os.MkdirAll no-ops on an existing path,
// so this is the only thing standing between the agent and a
// pre-created world-writable directory used to hijack the socket.
func verifyAgentOwnedDir(dir string) error {
	fi, err := os.Lstat(dir)
	if err != nil {
		return fmt.Errorf("lstat: %w", err)
	}
	if fi.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("path is a symlink")
	}
	if !fi.IsDir() {
		return fmt.Errorf("path is not a directory")
	}
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("cannot read directory ownership")
	}
	if int(st.Uid) != os.Geteuid() {
		return fmt.Errorf("directory is owned by uid %d, not the agent uid %d", st.Uid, os.Geteuid())
	}
	if fi.Mode().Perm()&0o022 != 0 {
		return fmt.Errorf("directory is group- or other-writable (mode %#o)", fi.Mode().Perm())
	}
	return nil
}

// SetSocketGid configures a numeric GID to chown the socket to at bind
// time. When gid >= 0, Start chowns the socket to (-1, gid) and widens
// the mode to 0o770 so a writer running under that supplementary group
// can connect. Must be called before Start. Mirrors the remediation
// server's SetSocketGid.
func (s *Server) SetSocketGid(gid int) { s.socketGid = gid }

// Start creates the agent-owned socket directory, binds the listener,
// applies the socket permissions, and begins accepting connections.
func (s *Server) Start() error {
	// Create the agent-owned dir 0o700. MkdirAll is a no-op when the
	// directory already exists, so creation alone does not prove the
	// directory is safe: a local attacker could have pre-created it
	// world-writable to hijack the socket. verifyAgentOwnedDir below
	// asserts the post-condition (real directory, agent-owned, no
	// group/other write) and refuses to start otherwise.
	if err := os.MkdirAll(s.socketDir, 0o700); err != nil {
		return fmt.Errorf("creating annotation socket dir %s: %w", s.socketDir, err)
	}
	if err := verifyAgentOwnedDir(s.socketDir); err != nil {
		return fmt.Errorf("annotation socket dir %s is unsafe: %w", s.socketDir, err)
	}

	// Remove a stale socket from a previous run. Lstat (not Stat) so a
	// symlink planted at the socket path is detected instead of
	// followed: removing a followed symlink deletes the symlink target,
	// an arbitrary-path delete. Only an actual socket is removed.
	if fi, err := os.Lstat(s.socketPath); err == nil {
		switch {
		case fi.Mode()&os.ModeSymlink != 0:
			return fmt.Errorf("annotation socket path %s is a symlink; refusing to start", s.socketPath)
		case fi.Mode()&os.ModeSocket != 0:
			if err := os.Remove(s.socketPath); err != nil {
				return fmt.Errorf("removing stale annotation socket %s: %w", s.socketPath, err)
			}
		default:
			return fmt.Errorf("annotation socket path %s exists and is not a socket; refusing to start", s.socketPath)
		}
	}

	ln, err := net.Listen("unix", s.socketPath)
	if err != nil {
		return fmt.Errorf("listening on annotation UDS %s: %w", s.socketPath, err)
	}

	// Restrict the socket. Owner-only (0o700) by default; with a gid
	// configured, chown to (-1, gid) + 0o770. On chown failure fall
	// back to owner-only so the agent still functions - a WARN is
	// emitted and the writer just gets EACCES.
	mode := os.FileMode(0o700)
	if s.socketGid >= 0 {
		if err := os.Chown(s.socketPath, -1, s.socketGid); err != nil {
			log.Printf("WARN: annotate: chown_failed path=%s gid=%d error=%v falling_back_to=0700",
				s.socketPath, s.socketGid, err)
		} else {
			mode = 0o770
		}
	}
	if err := os.Chmod(s.socketPath, mode); err != nil {
		ln.Close()
		return fmt.Errorf("chmod annotation socket %s: %w", s.socketPath, err)
	}

	s.listener = ln
	go s.acceptLoop()
	log.Printf("INFO: annotate: ingest_socket_started path=%s mode=%#o", s.socketPath, mode)
	return nil
}

// acceptLoop accepts connections and spawns one goroutine per
// connection. Unlike the remediation server's single-consumer model,
// this is multi-connection: the motivating callers write concurrently.
func (s *Server) acceptLoop() {
	for {
		conn, err := s.listener.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return
			}
			log.Printf("WARN: annotate: accept_error error=%v", err)
			continue
		}
		s.mu.Lock()
		if s.closed {
			s.mu.Unlock()
			conn.Close()
			return
		}
		s.conns[conn] = struct{}{}
		s.mu.Unlock()
		s.accepted.Add(1)
		go s.handleConn(conn)
	}
}

// handleConn reads NDJSON lines from one connection until EOF or error.
// A malformed, oversized, or invalid line is rejected without dropping
// the connection or the listener - the next line is still read.
func (s *Server) handleConn(conn net.Conn) {
	defer func() {
		conn.Close()
		s.mu.Lock()
		delete(s.conns, conn)
		s.mu.Unlock()
	}()

	// Capture SO_PEERCRED once per connection. The kernel credentials
	// are fixed for the life of the connection, so a single read is
	// enough; every annotation on this connection gets the same
	// provenance.
	//
	// A peercred failure is fatal to the connection: a zero Provenance
	// is indistinguishable from a legitimate "uid 0 wrote this" row, so
	// ingesting it would silently launder an unattributed annotation.
	// Drop the connection instead - the writer can retry and the
	// listener is unaffected.
	prov, err := peerCred(conn)
	if err != nil {
		s.rejected.Add(1)
		log.Printf("WARN: annotate: peercred_failed error=%v conn dropped", err)
		return
	}

	rl := newRateLimiter(contract.AnnotationConnRateLimit,
		time.Duration(contract.AnnotationConnRateWindowMs)*time.Millisecond)

	scanner := bufio.NewScanner(conn)
	// Cap the scanner buffer at the contract line limit so an oversized
	// line is rejected by the framing layer instead of allocating
	// unboundedly. bufio.Scanner returns bufio.ErrTooLong when a token
	// exceeds the buffer; that ends the scan for this connection.
	scanner.Buffer(make([]byte, 0, 4096), contract.AnnotationMaxLineBytes)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		if len(line) > contract.AnnotationMaxLineBytes {
			s.rejected.Add(1)
			continue
		}
		if !rl.allow() {
			// Per-connection rate cap exceeded; reject the line but keep
			// reading - the window will roll.
			s.rejected.Add(1)
			continue
		}
		s.processLine(line, prov)
	}
	if err := scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			// An oversized line tripped the framing cap. The connection
			// cannot be re-synced to a line boundary, so it is dropped;
			// the listener and other connections are unaffected.
			s.rejected.Add(1)
			log.Printf("WARN: annotate: oversized_line conn dropped (cap=%d bytes)",
				contract.AnnotationMaxLineBytes)
		}
		// Other errors (EPIPE on mid-line disconnect, reset) just end
		// the connection; nothing to log at WARN for a normal hangup.
	}
}

// wireAnnotation is the on-wire NDJSON object. Field names are pinned
// in pkg/contract. Numeric fields are decoded as json.Number-free
// concrete types so a missing field is the zero value.
type wireAnnotation struct {
	Timestamp int64             `json:"ts"`
	Labels    map[string]string `json:"labels"`
	PID       uint32            `json:"pid"`
	StartTime uint64            `json:"start_time"`
	SpanStart int64             `json:"span_start"`
	SpanEnd   int64             `json:"span_end"`
	Version   int               `json:"v"`
}

// processLine decodes, validates, resolves, and sinks one NDJSON line.
// A failure at any step bumps a counter and returns; it never panics
// and never drops the connection.
func (s *Server) processLine(line []byte, prov annotate.Provenance) {
	var w wireAnnotation
	if err := json.Unmarshal(line, &w); err != nil {
		s.rejected.Add(1)
		return
	}

	a := annotate.Annotation{
		TimestampNs: w.Timestamp,
		Labels:      w.Labels,
		SpanStartNs: w.SpanStart,
		SpanEndNs:   w.SpanEnd,
		Provenance:  prov,
	}
	// Stamp receive time when the writer omitted the timestamp.
	if a.TimestampNs == 0 {
		a.TimestampNs = time.Now().UnixNano()
	}

	// Resolve the process incarnation. The annotated PID and any
	// caller-supplied start_time are NOT trusted: a caller can name any
	// PID, including one it does not own. So:
	//   - the incarnation start_time is always re-resolved from /proc,
	//     never taken from the wire; a caller value that disagrees with
	//     /proc means the caller is lying or racing PID reuse - reject.
	//   - the annotated PID's real uid must match the connecting peer's
	//     uid, so a caller can only annotate its own processes.
	// start_time without a PID is ignored (unscoped).
	if w.PID != 0 {
		resolved := s.resolver(w.PID)
		if w.StartTime != 0 && resolved.StartTime != 0 && w.StartTime != resolved.StartTime {
			s.rejected.Add(1)
			log.Printf("WARN: annotate: start_time_mismatch pid=%d caller=%d resolved=%d",
				w.PID, w.StartTime, resolved.StartTime)
			return
		}
		// Verify the annotated PID is owned by the connecting peer.
		// PeerPID 0 only occurs when peercred could not be read, and
		// that path already drops the connection before processLine; a
		// captured peer is therefore always present here.
		//
		// Fail closed: if the PID's real uid cannot be read (the
		// process exited, or never existed), the scope cannot be
		// verified, so the PID-scoped annotation is rejected. Accepting
		// it would let a caller bypass the ownership check by naming a
		// dead or about-to-exit PID.
		owner, ok := s.pidUID(w.PID)
		if !ok {
			s.rejected.Add(1)
			log.Printf("WARN: annotate: pid_scope_unverifiable pid=%d peer_uid=%d",
				w.PID, prov.PeerUID)
			return
		}
		if owner != prov.PeerUID {
			s.rejected.Add(1)
			log.Printf("WARN: annotate: pid_scope_rejected pid=%d pid_uid=%d peer_uid=%d",
				w.PID, owner, prov.PeerUID)
			return
		}
		a.Process = resolved
	}

	if err := a.Validate(); err != nil {
		s.rejected.Add(1)
		return
	}

	if err := s.sink.RecordAnnotation(a); err != nil {
		s.sinkError.Add(1)
		log.Printf("WARN: annotate: sink_error error=%v", err)
		return
	}
	s.ingested.Add(1)
}

// Stats is a point-in-time snapshot of the server's counters.
type Stats struct {
	Accepted  uint64 // connections accepted
	Ingested  uint64 // annotations written to the sink
	Rejected  uint64 // lines rejected by framing or validation
	SinkError uint64 // validated lines the sink rejected
}

// ReadStats returns a snapshot of the server counters. Safe from any
// goroutine.
func (s *Server) ReadStats() Stats {
	return Stats{
		Accepted:  s.accepted.Load(),
		Ingested:  s.ingested.Load(),
		Rejected:  s.rejected.Load(),
		SinkError: s.sinkError.Load(),
	}
}

// SocketPath returns the bound socket path, for logging and for tests.
func (s *Server) SocketPath() string { return s.socketPath }

// Close stops the listener, closes every live connection, and removes
// the socket file. Safe to call even if Start was never called.
func (s *Server) Close() error {
	s.mu.Lock()
	s.closed = true
	if s.listener != nil {
		s.listener.Close()
	}
	for c := range s.conns {
		c.Close()
	}
	s.conns = make(map[net.Conn]struct{})
	s.mu.Unlock()

	if s.socketPath != "" {
		os.Remove(s.socketPath)
	}
	log.Printf("INFO: annotate: ingest_socket_stopped path=%s", s.socketPath)
	return nil
}
