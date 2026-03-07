package dashboard

import (
	"context"
	"crypto/tls"
	"io"
	"io/fs"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestServerStartsAndServesHTML(t *testing.T) {
	srv := New(nil, ":0", "", "")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Use a custom listener to get the port.
	tlsCfg := &tls.Config{MinVersion: tls.VersionTLS13}
	cert, _, err := generateSelfSignedCert()
	if err != nil {
		t.Fatal(err)
	}
	tlsCfg.Certificates = []tls.Certificate{cert}

	ln, err := tls.Listen("tcp", "localhost:0", tlsCfg)
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()

	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/capabilities", srv.handleCapabilities)
	mux.HandleFunc("/api/v1/overview", srv.handleOverview)

	// Serve embedded static files.
	staticSub, err := fs.Sub(staticFiles, "static")
	if err != nil {
		t.Fatal(err)
	}
	mux.Handle("/", http.FileServer(http.FS(staticSub)))

	httpSrv := &http.Server{Handler: hostGuard(mux), TLSConfig: tlsCfg}

	go func() {
		httpSrv.Serve(ln)
	}()

	go func() {
		<-ctx.Done()
		httpSrv.Close()
	}()

	time.Sleep(50 * time.Millisecond)

	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}

	// GET / should return HTML.
	resp, err := client.Get("https://" + addr + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("GET / status = %d, want 200", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "Ingero GPU Dashboard") {
		t.Error("GET / body does not contain expected title")
	}

	cancel()
}

func TestServerGracefulShutdown(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	srv := New(nil, "localhost:0", "", "")

	done := make(chan error, 1)
	go func() {
		done <- srv.Start(ctx)
	}()

	time.Sleep(100 * time.Millisecond)

	cancel()

	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Start() returned error on shutdown: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("server did not shut down within 3s")
	}
}

func TestHostGuardRejectsBadHost(t *testing.T) {
	handler := hostGuard(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
	}))

	tests := []struct {
		name   string
		host   string
		expect int
	}{
		{"localhost", "localhost", 200},
		{"localhost:8080", "localhost:8080", 200},
		{"127.0.0.1", "127.0.0.1", 200},
		{"127.0.0.1:9090", "127.0.0.1:9090", 200},
		{"::1", "::1", 200},
		{"evil.com", "evil.com", 403},
		{"attacker.local", "attacker.local", 403},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, _ := http.NewRequest("GET", "/", nil)
			req.Host = tt.host
			rec := &fakeResponseWriter{code: 200}
			handler.ServeHTTP(rec, req)
			if rec.code != tt.expect {
				t.Errorf("host=%q: got %d, want %d", tt.host, rec.code, tt.expect)
			}
		})
	}
}

// fakeResponseWriter captures the status code.
type fakeResponseWriter struct {
	code   int
	header http.Header
}

func (f *fakeResponseWriter) Header() http.Header {
	if f.header == nil {
		f.header = make(http.Header)
	}
	return f.header
}
func (f *fakeResponseWriter) Write(b []byte) (int, error) { return len(b), nil }
func (f *fakeResponseWriter) WriteHeader(code int)         { f.code = code }
