// Package k8s provides a lightweight Kubernetes API client for pod metadata
// enrichment and GPU pod discovery. Uses in-cluster ServiceAccount tokens —
// no k8s.io/client-go dependency (zero new imports; net/http + crypto/tls +
// encoding/json already linked via MCP SDK).
//
// Design decisions:
//   - Token refresh: Projected ServiceAccount tokens (K8s 1.24+) expire after
//     1 hour by default. We reload from disk every 50 minutes.
//   - HTTP timeout: 5s per request prevents PodCache goroutine from hanging.
//   - Detection: IsInCluster() checks KUBERNETES_SERVICE_HOST env var — the
//     standard way all in-cluster clients detect K8s (same as client-go).
//   - Graceful degradation: When not in K8s, NewInCluster() returns an error
//     and the agent runs without pod metadata (bare-metal mode).
package k8s

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

const (
	// Default paths for in-cluster ServiceAccount credentials.
	tokenPath = "/var/run/secrets/kubernetes.io/serviceaccount/token"
	caPath    = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

	// tokenRefreshInterval is how often we reload the token from disk.
	// Projected tokens expire at 1h by default; refresh at 50min for safety.
	tokenRefreshInterval = 50 * time.Minute

	// httpTimeout prevents API calls from blocking the PodCache goroutine.
	httpTimeout = 5 * time.Second
)

// Client is a lightweight Kubernetes API client using in-cluster credentials.
// It reads the ServiceAccount token from disk and refreshes it periodically
// to handle projected token rotation (K8s 1.24+).
type Client struct {
	host     string       // https://KUBERNETES_SERVICE_HOST:KUBERNETES_SERVICE_PORT
	token    string       // cached bearer token
	tokenMu  sync.Mutex   // protects token reads/writes
	http     *http.Client // TLS client with CA bundle
	nodeName string       // MY_NODE_NAME from Downward API
}

// IsInCluster returns true if running inside a Kubernetes pod.
// Checks the KUBERNETES_SERVICE_HOST env var, which kubelet always injects
// into pod containers. This is the same check client-go uses.
func IsInCluster() bool {
	return os.Getenv("KUBERNETES_SERVICE_HOST") != ""
}

// NewInCluster creates a Client using the pod's ServiceAccount credentials.
// Returns an error if not running in K8s or if credentials are missing.
//
// The client reads:
//   - Token from /var/run/secrets/kubernetes.io/serviceaccount/token
//   - CA cert from /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
//   - API server address from KUBERNETES_SERVICE_HOST + KUBERNETES_SERVICE_PORT
//   - Node name from MY_NODE_NAME (set via Downward API in DaemonSet)
func NewInCluster() (*Client, error) {
	host := os.Getenv("KUBERNETES_SERVICE_HOST")
	port := os.Getenv("KUBERNETES_SERVICE_PORT")
	if host == "" || port == "" {
		return nil, fmt.Errorf("not running in Kubernetes (KUBERNETES_SERVICE_HOST not set)")
	}

	// Read initial token.
	tokenBytes, err := os.ReadFile(tokenPath)
	if err != nil {
		return nil, fmt.Errorf("reading ServiceAccount token: %w", err)
	}

	// Read CA certificate for TLS verification against the API server.
	caCert, err := os.ReadFile(caPath)
	if err != nil {
		return nil, fmt.Errorf("reading CA certificate: %w", err)
	}
	caPool := x509.NewCertPool()
	if !caPool.AppendCertsFromPEM(caCert) {
		return nil, fmt.Errorf("failed to parse CA certificate")
	}

	c := &Client{
		host:     fmt.Sprintf("https://%s:%s", host, port),
		token:    string(tokenBytes),
		nodeName: os.Getenv("MY_NODE_NAME"),
		http: &http.Client{
			Timeout: httpTimeout,
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{
					RootCAs:    caPool,
					MinVersion: tls.VersionTLS12,
				},
			},
		},
	}
	return c, nil
}

// NodeName returns the node name from the Downward API (MY_NODE_NAME env var).
// Empty string if not set.
func (c *Client) NodeName() string {
	return c.nodeName
}

// Get performs an authenticated GET request to the K8s API.
// Path should start with "/" (e.g., "/api/v1/pods").
func (c *Client) Get(path string) ([]byte, error) {
	c.tokenMu.Lock()
	token := c.token
	c.tokenMu.Unlock()

	req, err := http.NewRequest("GET", c.host+path, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("K8s API request failed: %w", err)
	}
	defer resp.Body.Close()

	// Limit response size to 16MB to prevent OOM on large clusters
	// (e.g., empty MY_NODE_NAME causes cluster-wide pod listing).
	const maxResponseSize = 16 << 20
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseSize))
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("K8s API returned %d: %s", resp.StatusCode, truncate(string(body), 200))
	}
	return body, nil
}

// RefreshToken reloads the ServiceAccount token from disk.
// Called periodically by PodCache.Run() to handle projected token rotation.
func (c *Client) RefreshToken() error {
	tokenBytes, err := os.ReadFile(tokenPath)
	if err != nil {
		return fmt.Errorf("refreshing token: %w", err)
	}
	c.tokenMu.Lock()
	c.token = string(tokenBytes)
	c.tokenMu.Unlock()
	return nil
}

// truncate shortens a string to maxLen, appending "..." if truncated.
// The total output length is at most maxLen.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}
