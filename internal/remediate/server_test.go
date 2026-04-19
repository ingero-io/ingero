package remediate_test

import (
	"bufio"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/memtrack"
	"github.com/ingero-io/ingero/internal/remediate"
	"github.com/ingero-io/ingero/internal/straggler"
)

func tempSockPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "test.sock")
}

func startServer(t *testing.T) (*remediate.Server, string) {
	t.Helper()
	path := tempSockPath(t)
	srv := remediate.NewServer(path)
	if err := srv.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { srv.Close() })
	return srv, path
}

func dialUDS(t *testing.T, path string) net.Conn {
	t.Helper()
	// Retry briefly to allow accept loop to be ready.
	var conn net.Conn
	var err error
	for range 20 {
		conn, err = net.Dial("unix", path)
		if err == nil {
			return conn
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("Dial: %v", err)
	return nil
}

func sampleState() memtrack.MemoryState {
	return memtrack.MemoryState{
		PID:            12345,
		AllocatedBytes: 10737418240,
		TotalVRAM:      17179869184,
		UtilizationPct: 62.5,
		LastAllocSize:  268435456,
		TimestampNs:    1711180800000000000,
	}
}

func TestServer(t *testing.T) {
	t.Run("start_and_accept", func(t *testing.T) {
		_, path := startServer(t)
		conn := dialUDS(t, path)
		defer conn.Close()
		// Connection succeeded — that's the assertion.
	})

	t.Run("send_writes_valid_ndjson", func(t *testing.T) {
		srv, path := startServer(t)
		conn := dialUDS(t, path)
		defer conn.Close()

		// Give accept loop time to register the connection.
		time.Sleep(20 * time.Millisecond)

		ms := sampleState()
		srv.Send(ms)

		conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		scanner := bufio.NewScanner(conn)
		if !scanner.Scan() {
			t.Fatalf("expected a line, got error: %v", scanner.Err())
		}
		line := scanner.Bytes()

		// Parse and verify all 6 fields.
		var got memtrack.MemoryState
		if err := json.Unmarshal(line, &got); err != nil {
			t.Fatalf("JSON unmarshal: %v", err)
		}

		if got.PID != ms.PID {
			t.Errorf("PID: got %d, want %d", got.PID, ms.PID)
		}
		if got.AllocatedBytes != ms.AllocatedBytes {
			t.Errorf("AllocatedBytes: got %d, want %d", got.AllocatedBytes, ms.AllocatedBytes)
		}
		if got.TotalVRAM != ms.TotalVRAM {
			t.Errorf("TotalVRAM: got %d, want %d", got.TotalVRAM, ms.TotalVRAM)
		}
		if got.UtilizationPct != ms.UtilizationPct {
			t.Errorf("UtilizationPct: got %f, want %f", got.UtilizationPct, ms.UtilizationPct)
		}
		if got.LastAllocSize != ms.LastAllocSize {
			t.Errorf("LastAllocSize: got %d, want %d", got.LastAllocSize, ms.LastAllocSize)
		}
		if got.TimestampNs != ms.TimestampNs {
			t.Errorf("TimestampNs: got %d, want %d", got.TimestampNs, ms.TimestampNs)
		}
	})

	t.Run("send_without_client_drops_silently", func(t *testing.T) {
		srv, _ := startServer(t)
		srv.Send(sampleState())

		if d := srv.Dropped(); d != 1 {
			t.Errorf("Dropped: got %d, want 1", d)
		}
		// The per-reason counter names the reason as no_client.
		got := srv.DroppedByReason()
		if got[remediate.DropReasonNoClient] != 1 {
			t.Errorf("DroppedByReason[no_client]=%d, want 1; all=%v", got[remediate.DropReasonNoClient], got)
		}
	})

	// SendStraggle / SendFleetStraggler* return a typed *DroppedError that
	// unwraps to ErrDropped when no client is connected. This lets callers
	// distinguish dropped vs delivered without parsing strings.
	t.Run("send_straggle_without_client_returns_typed_error", func(t *testing.T) {
		srv, _ := startServer(t)
		err := srv.SendStraggle(straggler.StraggleState{PID: 1})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !errors.Is(err, remediate.ErrDropped) {
			t.Errorf("errors.Is(err, ErrDropped)=false; err=%v", err)
		}
		var de *remediate.DroppedError
		if !errors.As(err, &de) {
			t.Fatalf("errors.As(*DroppedError) failed; err=%v", err)
		}
		if de.Reason != remediate.DropReasonNoClient {
			t.Errorf("Reason=%q, want %q", de.Reason, remediate.DropReasonNoClient)
		}
		if got := srv.DroppedByReason()[remediate.DropReasonNoClient]; got != 1 {
			t.Errorf("DroppedByReason[no_client]=%d, want 1", got)
		}
	})

	t.Run("send_fleet_straggler_state_without_client_returns_typed_error", func(t *testing.T) {
		srv, _ := startServer(t)
		err := srv.SendFleetStragglerState(time.Now(), "n1", "c1", "mad", "throughput", 0.5, 0.8)
		if !errors.Is(err, remediate.ErrDropped) {
			t.Errorf("errors.Is(err, ErrDropped)=false; err=%v", err)
		}
		var de *remediate.DroppedError
		if errors.As(err, &de) && de.Reason != remediate.DropReasonNoClient {
			t.Errorf("Reason=%q, want %q", de.Reason, remediate.DropReasonNoClient)
		}
	})

	t.Run("stale_socket_cleanup", func(t *testing.T) {
		path := tempSockPath(t)

		// Create a regular file to simulate a stale socket.
		if err := os.WriteFile(path, []byte("stale"), 0600); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		srv := remediate.NewServer(path)
		if err := srv.Start(); err != nil {
			t.Fatalf("Start with stale socket: %v", err)
		}
		defer srv.Close()

		conn := dialUDS(t, path)
		conn.Close()
	})

	t.Run("client_reconnect", func(t *testing.T) {
		srv, path := startServer(t)

		// Client A connects.
		connA := dialUDS(t, path)
		time.Sleep(20 * time.Millisecond)

		srv.Send(sampleState())

		connA.SetReadDeadline(time.Now().Add(2 * time.Second))
		scanA := bufio.NewScanner(connA)
		if !scanA.Scan() {
			t.Fatalf("client A: expected line, got error: %v", scanA.Err())
		}
		var gotA memtrack.MemoryState
		if err := json.Unmarshal(scanA.Bytes(), &gotA); err != nil {
			t.Fatalf("client A unmarshal: %v", err)
		}
		connA.Close()

		// Client B connects.
		connB := dialUDS(t, path)
		defer connB.Close()
		time.Sleep(20 * time.Millisecond)

		srv.Send(sampleState())

		connB.SetReadDeadline(time.Now().Add(2 * time.Second))
		scanB := bufio.NewScanner(connB)
		if !scanB.Scan() {
			t.Fatalf("client B: expected line, got error: %v", scanB.Err())
		}
		var gotB memtrack.MemoryState
		if err := json.Unmarshal(scanB.Bytes(), &gotB); err != nil {
			t.Fatalf("client B unmarshal: %v", err)
		}

		// Both received valid data.
		if gotA.PID != 12345 || gotB.PID != 12345 {
			t.Errorf("PID mismatch: A=%d B=%d", gotA.PID, gotB.PID)
		}
	})

	t.Run("json_field_names_match_schema", func(t *testing.T) {
		srv, path := startServer(t)
		conn := dialUDS(t, path)
		defer conn.Close()
		time.Sleep(20 * time.Millisecond)

		srv.Send(sampleState())

		conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		scanner := bufio.NewScanner(conn)
		if !scanner.Scan() {
			t.Fatalf("expected line, got error: %v", scanner.Err())
		}

		var raw map[string]interface{}
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			t.Fatalf("unmarshal to map: %v", err)
		}

		expected := map[string]bool{
			"type":            true,
			"pid":             true,
			"gpu_id":          true,
			"allocated_bytes": true,
			"total_vram":      true,
			"utilization_pct": true,
			"last_alloc_size": true,
			"timestamp_ns":    true,
		}

		for k := range raw {
			if !expected[k] {
				t.Errorf("unexpected JSON key: %q", k)
			}
			delete(expected, k)
		}
		for k := range expected {
			t.Errorf("missing JSON key: %q", k)
		}
	})

	t.Run("type_field_value_is_memory", func(t *testing.T) {
		srv, path := startServer(t)
		conn := dialUDS(t, path)
		defer conn.Close()
		time.Sleep(20 * time.Millisecond)

		srv.Send(sampleState())

		conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		scanner := bufio.NewScanner(conn)
		if !scanner.Scan() {
			t.Fatalf("expected line, got error: %v", scanner.Err())
		}

		var raw map[string]interface{}
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			t.Fatalf("unmarshal to map: %v", err)
		}

		typeVal, ok := raw["type"]
		if !ok {
			t.Fatal("missing 'type' field in JSON")
		}
		if typeVal != "memory" {
			t.Errorf("type: got %v, want 'memory'", typeVal)
		}
	})

	t.Run("close_removes_socket_file", func(t *testing.T) {
		path := tempSockPath(t)
		srv := remediate.NewServer(path)
		if err := srv.Start(); err != nil {
			t.Fatalf("Start: %v", err)
		}

		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Fatal("socket file should exist after Start")
		}

		srv.Close()

		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatal("socket file should not exist after Close")
		}
	})

	// Default (no SetSocketGid call) keeps 0o700.
	t.Run("default_socket_mode_is_0700", func(t *testing.T) {
		path := tempSockPath(t)
		srv := remediate.NewServer(path)
		if err := srv.Start(); err != nil {
			t.Fatalf("Start: %v", err)
		}
		defer srv.Close()

		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("Stat: %v", err)
		}
		if mode := info.Mode().Perm(); mode != 0o700 {
			t.Errorf("socket mode=%o, want 0700", mode)
		}
	})

	// SetSocketGid with the current user's primary gid produces a 0o770
	// socket. Using the caller's own gid means chown always succeeds, even
	// in unprivileged CI (chown -1 to self is allowed).
	t.Run("socket_gid_grants_group_access", func(t *testing.T) {
		gid := os.Getgid()
		path := tempSockPath(t)
		srv := remediate.NewServer(path)
		srv.SetSocketGid(gid)
		if err := srv.Start(); err != nil {
			t.Fatalf("Start: %v", err)
		}
		defer srv.Close()

		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("Stat: %v", err)
		}
		if mode := info.Mode().Perm(); mode != 0o770 {
			t.Errorf("socket mode=%o, want 0770 when SetSocketGid is called", mode)
		}
	})

	// SetSocketGid with a negative value keeps owner-only.
	t.Run("negative_socket_gid_keeps_0700", func(t *testing.T) {
		path := tempSockPath(t)
		srv := remediate.NewServer(path)
		srv.SetSocketGid(-1)
		if err := srv.Start(); err != nil {
			t.Fatalf("Start: %v", err)
		}
		defer srv.Close()

		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("Stat: %v", err)
		}
		if mode := info.Mode().Perm(); mode != 0o700 {
			t.Errorf("socket mode=%o, want 0700 with negative gid", mode)
		}
	})
}
