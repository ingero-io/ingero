// Package fleet provides an HTTP fan-out client for querying multiple Ingero
// nodes and concatenating results. Used by `ingero query --nodes` and
// `ingero explain --nodes` to provide cross-node investigation from a single CLI.
//
// Architecture: client-side fan-out only. Each node runs its own dashboard API.
// The fleet client sends concurrent HTTP requests and concatenates the results.
// No central aggregation service — the user's CLI is the merge point.
package fleet

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	neturl "net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// DefaultTimeout is the per-node HTTP request timeout.
const DefaultTimeout = 5 * time.Second

// DefaultLimit is the max rows returned per node.
const DefaultLimit = 1000

// Config configures the fleet client.
type Config struct {
	Nodes      []string      // host:port addresses
	Timeout    time.Duration // per-node timeout (0 = DefaultTimeout)
	Limit      int           // per-node row limit (0 = DefaultLimit)
	CACert     string        // path to CA cert for mTLS (empty = plain HTTP)
	ClientCert string        // path to client cert for mTLS
	ClientKey  string        // path to client key for mTLS
}

// Client is an HTTP fan-out client for fleet queries.
type Client struct {
	nodes   []string
	timeout time.Duration
	limit   int
	http    *http.Client
	scheme  string // "http" or "https"
}

// New creates a fleet client from the given config.
func New(cfg Config) (*Client, error) {
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = DefaultTimeout
	}
	limit := cfg.Limit
	if limit == 0 {
		limit = DefaultLimit
	}

	scheme := "http"
	transport := http.DefaultTransport

	if cfg.CACert != "" || cfg.ClientCert != "" {
		scheme = "https"
		tlsCfg := &tls.Config{MinVersion: tls.VersionTLS12}

		if cfg.CACert != "" {
			caPEM, err := os.ReadFile(cfg.CACert)
			if err != nil {
				return nil, fmt.Errorf("fleet: reading CA cert: %w", err)
			}
			pool := x509.NewCertPool()
			if !pool.AppendCertsFromPEM(caPEM) {
				return nil, fmt.Errorf("fleet: invalid CA cert in %s", cfg.CACert)
			}
			tlsCfg.RootCAs = pool
		}

		if cfg.ClientCert != "" && cfg.ClientKey != "" {
			cert, err := tls.LoadX509KeyPair(cfg.ClientCert, cfg.ClientKey)
			if err != nil {
				return nil, fmt.Errorf("fleet: loading client cert: %w", err)
			}
			tlsCfg.Certificates = []tls.Certificate{cert}
		}

		transport = &http.Transport{TLSClientConfig: tlsCfg}
	}

	return &Client{
		nodes:   cfg.Nodes,
		timeout: timeout,
		limit:   limit,
		http:    &http.Client{Transport: transport},
		scheme:  scheme,
	}, nil
}

// QueryResult holds the merged result of a fan-out SQL query.
type QueryResult struct {
	Columns  []string   // column names (first is "node")
	Rows     [][]any    // row data
	Warnings []string   // errors from individual nodes
}

// ChainResult holds the merged result of a fan-out chain query.
type ChainResult struct {
	Chains   []ChainEntry // merged chains sorted by severity
	Warnings []string     // errors from individual nodes
}

// ChainEntry is a causal chain from a remote node.
type ChainEntry struct {
	Node            string   `json:"node"`
	ID              string   `json:"id"`
	DetectedAt      string   `json:"detected_at"`
	Severity        string   `json:"severity"`
	Summary         string   `json:"summary"`
	RootCause       string   `json:"root_cause"`
	Explanation     string   `json:"explanation"`
	Recommendations []string `json:"recommendations,omitempty"`
	CUDAOp          string   `json:"cuda_op,omitempty"`
	CUDAP99US       int64    `json:"cuda_p99_us,omitempty"`
	CUDAP50US       int64    `json:"cuda_p50_us,omitempty"`
	TailRatio       float64  `json:"tail_ratio,omitempty"`
}

// queryRequest is the POST body for /api/v1/query.
type queryRequest struct {
	SQL   string `json:"sql"`
	Limit int    `json:"limit"`
}

// queryResponse is the JSON response from /api/v1/query.
type queryResponse struct {
	Columns []string `json:"columns"`
	Rows    [][]any  `json:"rows"`
}

// chainAPIResponse is the JSON response from /api/v1/chains.
type chainAPIResponse struct {
	Chains []ChainEntry `json:"chains"`
}

// nodeResult holds the result from a single node.
type nodeResult struct {
	node string
	resp *queryResponse
	err  error
}

// nodeChainResult holds chain results from a single node.
type nodeChainResult struct {
	node   string
	chains []ChainEntry
	err    error
}

// QuerySQL fans out a SQL query to all configured nodes and concatenates results.
// The result has a "node" column prepended to identify each row's origin.
func (c *Client) QuerySQL(ctx context.Context, sql string) (*QueryResult, error) {
	results := make([]nodeResult, len(c.nodes))
	var wg sync.WaitGroup

	for i, node := range c.nodes {
		wg.Add(1)
		go func(idx int, addr string) {
			defer wg.Done()
			results[idx] = c.queryNode(ctx, addr, sql)
		}(i, node)
	}
	wg.Wait()

	return c.mergeQueryResults(results)
}

// QueryChains fans out a chain query to all nodes and merges results sorted by severity.
func (c *Client) QueryChains(ctx context.Context, since string) (*ChainResult, error) {
	results := make([]nodeChainResult, len(c.nodes))
	var wg sync.WaitGroup

	for i, node := range c.nodes {
		wg.Add(1)
		go func(idx int, addr string) {
			defer wg.Done()
			results[idx] = c.queryNodeChains(ctx, addr, since)
		}(i, node)
	}
	wg.Wait()

	return c.mergeChainResults(results)
}

// GenericResult holds the result of a fan-out GET request returning raw JSON per node.
type GenericResult struct {
	Nodes    map[string]json.RawMessage // node addr → raw JSON response
	Warnings []string
}

// QueryEndpoint fans out a GET request to a path on each node and returns raw JSON per node.
// Used for /api/v1/ops, /api/v1/overview, etc. where response schemas vary by endpoint.
func (c *Client) QueryEndpoint(ctx context.Context, path string) (*GenericResult, error) {
	type nodeGenericResult struct {
		node string
		data json.RawMessage
		err  error
	}

	results := make([]nodeGenericResult, len(c.nodes))
	var wg sync.WaitGroup

	for i, node := range c.nodes {
		wg.Add(1)
		go func(idx int, addr string) {
			defer wg.Done()
			reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
			defer cancel()

			url := fmt.Sprintf("%s://%s%s", c.scheme, addr, path)
			req, err := http.NewRequestWithContext(reqCtx, "GET", url, nil)
			if err != nil {
				results[idx] = nodeGenericResult{node: addr, err: err}
				return
			}

			resp, err := c.http.Do(req)
			if err != nil {
				results[idx] = nodeGenericResult{node: addr, err: err}
				return
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
				results[idx] = nodeGenericResult{node: addr, err: fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))}
				return
			}

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				results[idx] = nodeGenericResult{node: addr, err: fmt.Errorf("reading body: %w", err)}
				return
			}
			results[idx] = nodeGenericResult{node: addr, data: json.RawMessage(body)}
		}(i, node)
	}
	wg.Wait()

	merged := &GenericResult{Nodes: make(map[string]json.RawMessage)}
	for _, r := range results {
		if r.err != nil {
			merged.Warnings = append(merged.Warnings, fmt.Sprintf("node %s: %v", r.node, r.err))
			continue
		}
		merged.Nodes[r.node] = r.data
	}

	if len(merged.Nodes) == 0 && len(merged.Warnings) > 0 {
		return merged, fmt.Errorf("all nodes failed:\n  %s", strings.Join(merged.Warnings, "\n  "))
	}

	return merged, nil
}

func (c *Client) queryNode(ctx context.Context, addr, sql string) nodeResult {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	body, _ := json.Marshal(queryRequest{SQL: sql, Limit: c.limit})
	url := fmt.Sprintf("%s://%s/api/v1/query", c.scheme, addr)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nodeResult{node: addr, err: fmt.Errorf("creating request: %w", err)}
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nodeResult{node: addr, err: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nodeResult{node: addr, err: fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))}
	}

	var qr queryResponse
	if err := json.NewDecoder(resp.Body).Decode(&qr); err != nil {
		return nodeResult{node: addr, err: fmt.Errorf("decoding response: %w", err)}
	}

	return nodeResult{node: addr, resp: &qr}
}

func (c *Client) queryNodeChains(ctx context.Context, addr, since string) nodeChainResult {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	url := fmt.Sprintf("%s://%s/api/v1/chains", c.scheme, addr)
	if since != "" {
		url += "?since=" + neturl.QueryEscape(since)
	}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nodeChainResult{node: addr, err: fmt.Errorf("creating request: %w", err)}
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return nodeChainResult{node: addr, err: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nodeChainResult{node: addr, err: fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))}
	}

	var cr chainAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return nodeChainResult{node: addr, err: fmt.Errorf("decoding response: %w", err)}
	}

	// Tag each chain with the node address if not already set.
	for i := range cr.Chains {
		if cr.Chains[i].Node == "" {
			cr.Chains[i].Node = addr
		}
	}

	return nodeChainResult{node: addr, chains: cr.Chains}
}

func (c *Client) mergeQueryResults(results []nodeResult) (*QueryResult, error) {
	merged := &QueryResult{}

	var refCols []string
	for _, r := range results {
		if r.err != nil {
			merged.Warnings = append(merged.Warnings, fmt.Sprintf("node %s: %v", r.node, r.err))
			continue
		}
		if r.resp == nil || len(r.resp.Columns) == 0 {
			continue
		}

		// First successful result sets the reference columns.
		if refCols == nil {
			refCols = r.resp.Columns
			merged.Columns = append([]string{"node"}, refCols...)
		} else {
			// Check schema match.
			if !columnsEqual(refCols, r.resp.Columns) {
				merged.Warnings = append(merged.Warnings,
					fmt.Sprintf("node %s: column mismatch (got %v, expected %v) — skipped", r.node, r.resp.Columns, refCols))
				continue
			}
		}

		// Prepend node column to each row.
		for _, row := range r.resp.Rows {
			merged.Rows = append(merged.Rows, append([]any{r.node}, row...))
		}
	}

	// If all nodes failed, return error.
	if refCols == nil && len(merged.Warnings) > 0 {
		return merged, fmt.Errorf("all nodes failed:\n  %s", strings.Join(merged.Warnings, "\n  "))
	}

	return merged, nil
}

func (c *Client) mergeChainResults(results []nodeChainResult) (*ChainResult, error) {
	merged := &ChainResult{}

	for _, r := range results {
		if r.err != nil {
			merged.Warnings = append(merged.Warnings, fmt.Sprintf("node %s: %v", r.node, r.err))
			continue
		}
		merged.Chains = append(merged.Chains, r.chains...)
	}

	// Sort by severity: HIGH > MEDIUM > LOW.
	sort.SliceStable(merged.Chains, func(i, j int) bool {
		return severityRank(merged.Chains[i].Severity) > severityRank(merged.Chains[j].Severity)
	})

	// If all nodes failed, return error.
	if len(merged.Chains) == 0 && len(merged.Warnings) > 0 {
		return merged, fmt.Errorf("all nodes failed:\n  %s", strings.Join(merged.Warnings, "\n  "))
	}

	return merged, nil
}

func columnsEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func severityRank(s string) int {
	switch strings.ToUpper(s) {
	case "HIGH":
		return 3
	case "MEDIUM":
		return 2
	case "LOW":
		return 1
	default:
		return 0
	}
}
