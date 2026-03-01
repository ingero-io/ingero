package k8s

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

func TestIsInCluster(t *testing.T) {
	// Save and restore env.
	orig := os.Getenv("KUBERNETES_SERVICE_HOST")
	defer os.Setenv("KUBERNETES_SERVICE_HOST", orig)

	os.Setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
	if !IsInCluster() {
		t.Error("expected IsInCluster()=true when KUBERNETES_SERVICE_HOST is set")
	}

	os.Unsetenv("KUBERNETES_SERVICE_HOST")
	if IsInCluster() {
		t.Error("expected IsInCluster()=false when KUBERNETES_SERVICE_HOST is unset")
	}
}

func TestClientGet(t *testing.T) {
	// Mock K8s API server.
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/pods" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		auth := r.Header.Get("Authorization")
		if auth != "Bearer test-token" {
			t.Errorf("unexpected auth: %s", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"items":[]}`))
	}))
	defer srv.Close()

	c := &Client{
		host:  srv.URL,
		token: "test-token",
		http:  srv.Client(),
	}

	data, err := c.Get("/api/v1/pods")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(data) != `{"items":[]}` {
		t.Errorf("unexpected response: %s", data)
	}
}

func TestClientGetError(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		w.Write([]byte(`{"message":"forbidden"}`))
	}))
	defer srv.Close()

	c := &Client{
		host:  srv.URL,
		token: "bad-token",
		http:  srv.Client(),
	}

	_, err := c.Get("/api/v1/pods")
	if err == nil {
		t.Fatal("expected error for 403 response")
	}
}

func TestTruncate(t *testing.T) {
	if got := truncate("hello", 10); got != "hello" {
		t.Errorf("truncate short: %q", got)
	}
	if got := truncate("hello world", 8); got != "hello..." {
		t.Errorf("truncate long: got %q, want %q", got, "hello...")
	}
	if got := truncate("abcdef", 3); got != "abc" {
		t.Errorf("truncate tiny maxLen: got %q, want %q", got, "abc")
	}
}

func TestNodeName(t *testing.T) {
	c := &Client{nodeName: "gpu-node-01"}
	if c.NodeName() != "gpu-node-01" {
		t.Errorf("NodeName() = %q, want gpu-node-01", c.NodeName())
	}

	c2 := &Client{}
	if c2.NodeName() != "" {
		t.Errorf("NodeName() = %q, want empty", c2.NodeName())
	}
}
