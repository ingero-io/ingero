package k8s

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// samplePodListJSON is a realistic K8s API pod list response.
// Contains one GPU pod (training-job-0) and one non-GPU pod (metrics-exporter).
const samplePodListJSON = `{
  "items": [
    {
      "metadata": {
        "name": "training-job-0",
        "namespace": "ml-team"
      },
      "spec": {
        "containers": [
          {
            "name": "pytorch",
            "resources": {
              "limits": {
                "nvidia.com/gpu": "1",
                "memory": "16Gi"
              }
            }
          }
        ]
      },
      "status": {
        "containerStatuses": [
          {
            "containerID": "containerd://a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
          }
        ]
      }
    },
    {
      "metadata": {
        "name": "metrics-exporter",
        "namespace": "monitoring"
      },
      "spec": {
        "containers": [
          {
            "name": "exporter",
            "resources": {
              "limits": {
                "memory": "256Mi"
              }
            }
          }
        ]
      },
      "status": {
        "containerStatuses": [
          {
            "containerID": "containerd://bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
          }
        ]
      }
    }
  ]
}`

func TestParsePodList(t *testing.T) {
	pods, err := parsePodList([]byte(samplePodListJSON))
	if err != nil {
		t.Fatalf("parsePodList: %v", err)
	}

	if len(pods) != 2 {
		t.Fatalf("got %d pods, want 2", len(pods))
	}

	// First pod: GPU-requesting training job.
	p := pods[0]
	if p.Name != "training-job-0" {
		t.Errorf("pod[0].Name = %q, want training-job-0", p.Name)
	}
	if p.Namespace != "ml-team" {
		t.Errorf("pod[0].Namespace = %q, want ml-team", p.Namespace)
	}
	if !p.GPURequested {
		t.Error("pod[0].GPURequested = false, want true")
	}
	if len(p.ContainerIDs) != 1 {
		t.Fatalf("pod[0] has %d container IDs, want 1", len(p.ContainerIDs))
	}
	if p.ContainerIDs[0] != "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2" {
		t.Errorf("pod[0].ContainerIDs[0] = %q", p.ContainerIDs[0])
	}

	// Second pod: non-GPU metrics exporter.
	p2 := pods[1]
	if p2.Name != "metrics-exporter" {
		t.Errorf("pod[1].Name = %q, want metrics-exporter", p2.Name)
	}
	if p2.GPURequested {
		t.Error("pod[1].GPURequested = true, want false")
	}
}

func TestStripRuntimePrefix(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"containerd://abc123", "abc123"},
		{"docker://abc123", "abc123"},
		{"cri-o://abc123", "abc123"},
		{"abc123", "abc123"},
		{"", ""},
	}
	for _, tt := range tests {
		got := stripRuntimePrefix(tt.input)
		if got != tt.want {
			t.Errorf("stripRuntimePrefix(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestPodCacheLookup(t *testing.T) {
	// Set up mock K8s API server.
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(samplePodListJSON))
	}))
	defer srv.Close()

	client := &Client{
		host:  srv.URL,
		token: "test",
		http:  srv.Client(),
	}

	cache := NewPodCache(client)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	go cache.Run(ctx)

	// Wait for first refresh.
	if err := cache.WaitReady(ctx, 3*time.Second); err != nil {
		t.Fatalf("WaitReady: %v", err)
	}

	// Lookup by container ID (prefix stripped).
	info := cache.Lookup("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2")
	if info == nil {
		t.Fatal("Lookup returned nil for known container ID")
	}
	if info.Name != "training-job-0" {
		t.Errorf("Lookup().Name = %q, want training-job-0", info.Name)
	}
	if info.Namespace != "ml-team" {
		t.Errorf("Lookup().Namespace = %q, want ml-team", info.Namespace)
	}
	if !info.GPURequested {
		t.Error("Lookup().GPURequested = false, want true")
	}

	// Lookup unknown container ID.
	if cache.Lookup("unknown") != nil {
		t.Error("Lookup returned non-nil for unknown container ID")
	}
}

func TestGPUPods(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(samplePodListJSON))
	}))
	defer srv.Close()

	client := &Client{
		host:  srv.URL,
		token: "test",
		http:  srv.Client(),
	}

	cache := NewPodCache(client)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	go cache.Run(ctx)
	if err := cache.WaitReady(ctx, 3*time.Second); err != nil {
		t.Fatalf("WaitReady: %v", err)
	}

	gpuPods := cache.GPUPods()
	if len(gpuPods) != 1 {
		t.Fatalf("GPUPods() returned %d pods, want 1", len(gpuPods))
	}
	if gpuPods[0].Name != "training-job-0" {
		t.Errorf("GPUPods()[0].Name = %q, want training-job-0", gpuPods[0].Name)
	}
}

func TestPodCacheGracefulDegradation(t *testing.T) {
	// API server that always returns 500.
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	client := &Client{
		host:  srv.URL,
		token: "test",
		http:  srv.Client(),
	}

	cache := NewPodCache(client)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	go cache.Run(ctx)

	// WaitReady should timeout — API is broken.
	err := cache.WaitReady(ctx, 500*time.Millisecond)
	if err == nil {
		t.Error("WaitReady should have returned error when API is unavailable")
	}

	// Lookup should return nil, not panic.
	if cache.Lookup("anything") != nil {
		t.Error("Lookup should return nil when cache is empty")
	}

	// GPUPods should return empty, not panic.
	if len(cache.GPUPods()) != 0 {
		t.Error("GPUPods should return empty when cache is empty")
	}
}

func TestParsePodListEmpty(t *testing.T) {
	pods, err := parsePodList([]byte(`{"items":[]}`))
	if err != nil {
		t.Fatalf("parsePodList: %v", err)
	}
	if len(pods) != 0 {
		t.Errorf("got %d pods, want 0", len(pods))
	}
}

func TestParsePodListInvalidJSON(t *testing.T) {
	_, err := parsePodList([]byte(`not json`))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}
