package tracing

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestInit_DisabledReturnsNoop(t *testing.T) {
	tracer, shutdown, err := Init(context.Background(), Config{Enabled: false})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	if tracer == nil {
		t.Fatal("nil tracer")
	}
	_, span := tracer.Start(context.Background(), "test")
	if span.IsRecording() {
		t.Fatal("disabled config produced a recording span")
	}
	span.End()
	if err := shutdown(context.Background()); err != nil {
		t.Fatalf("shutdown: %v", err)
	}
}

func TestInit_DisabledWithMalformedEndpointReturnsError(t *testing.T) {
	// Even when Enabled=false, a non-empty malformed endpoint must surface
	// at Init time so the typo is caught at startup. An empty endpoint is
	// the legitimate disabled-and-unconfigured case and stays a no-op.
	cases := []struct {
		name string
		ep   string
	}{
		{"no_port", "fleet.example"},
		{"bad_scheme", "ftp://fleet.example:4318"},
		{"missing_host", "http://"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := Init(context.Background(), Config{
				Enabled:  false,
				Endpoint: tc.ep,
			})
			if err == nil {
				t.Fatalf("expected error for disabled+%q endpoint", tc.ep)
			}
		})
	}
	// Empty endpoint with Enabled=false stays a no-op (no error).
	_, _, err := Init(context.Background(), Config{Enabled: false, Endpoint: ""})
	if err != nil {
		t.Fatalf("disabled+empty endpoint must not error: %v", err)
	}
}

func TestInit_BadEndpointReturnsError(t *testing.T) {
	cases := []struct {
		name string
		ep   string
	}{
		{"empty", ""},
		{"no_port", "fleet.example"},
		{"bad_scheme", "ftp://fleet.example:4318"},
		{"missing_host", "http://"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := Init(context.Background(), Config{
				Enabled:  true,
				Endpoint: tc.ep,
			})
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestInit_HappyPath(t *testing.T) {
	var (
		mu          sync.Mutex
		gotMethods  []string
		gotPaths    []string
		gotBodyLens []int
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		gotMethods = append(gotMethods, r.Method)
		gotPaths = append(gotPaths, r.URL.Path)
		gotBodyLens = append(gotBodyLens, len(body))
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	endpoint := strings.TrimPrefix(srv.URL, "http://")
	tracer, tp, _, err := initWithProvider(context.Background(), Config{
		Enabled:        true,
		Endpoint:       endpoint,
		Insecure:       true,
		NodeID:         "node-test",
		ClusterID:      "cluster-test",
		ServiceVersion: "v0.13.0-test",
	})
	if err != nil {
		t.Fatalf("initWithProvider: %v", err)
	}
	if tp == nil {
		t.Fatal("nil provider")
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = tp.Shutdown(ctx)
	}()

	_, span := tracer.Start(context.Background(), "ingero.detection.straggler")
	span.End()

	flushCtx, flushCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer flushCancel()
	if err := tp.ForceFlush(flushCtx); err != nil {
		t.Fatalf("ForceFlush: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(gotMethods) == 0 {
		t.Fatal("server did not receive any request")
	}
	if gotMethods[0] != "POST" {
		t.Fatalf("method = %q, want POST", gotMethods[0])
	}
	if !strings.Contains(gotPaths[0], "/v1/traces") {
		t.Fatalf("path = %q, want to contain /v1/traces", gotPaths[0])
	}
	if gotBodyLens[0] == 0 {
		t.Fatal("server received empty body")
	}
}
