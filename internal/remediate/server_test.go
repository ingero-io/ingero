package remediate_test

import (
	"bufio"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/memtrack"
	"github.com/ingero-io/ingero/internal/remediate"
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
}
