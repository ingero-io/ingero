// Package mcp provides an MCP (Model Context Protocol) server for Ingero.
//
// The MCP server exposes ten tools and one prompt to AI agents:
//
// Tools:
//   - get_check: Run system diagnostics (kernel, BTF, NVIDIA, CUDA)
//   - get_trace_stats: Get CUDA/host stats (p50/p95/p99 or aggregate fallback)
//   - get_causal_chains: Analyze events and return causal chains with severity
//   - run_demo: Run a synthetic demo scenario
//   - get_test_report: Return the GPU integration test report (JSON)
//   - run_sql: Execute read-only SQL for ad-hoc analysis
//   - get_stacks: Get resolved call stacks for operations (symbol names, source files)
//   - graph_lifecycle: CUDA Graph lifecycle timeline for a PID
//   - graph_frequency: Graph launch frequency and hot/cold classification
//   - query_fleet: Fan-out query across multiple Ingero nodes (chains, ops, overview, sql)
//
// Prompts:
//   - /investigate: Guided investigation workflow for diagnosing GPU issues
//
// Usage:
//
//	ingero mcp
//
// This starts an MCP server on stdio, ready for Claude or other MCP clients.
package mcp

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"net"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	gomcp "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/ingero-io/ingero/internal/fleet"
	"github.com/ingero-io/ingero/internal/correlate"
	"github.com/ingero-io/ingero/internal/discover"
	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/internal/synth"
	"github.com/ingero-io/ingero/pkg/events"
)

// Server wraps the MCP server and its dependencies.
type Server struct {
	mcpServer   *gomcp.Server
	store       *store.Store
	fleetNodes  []string          // fleet.nodes from config (nil = fleet disabled)
	fleetClient *fleet.Client     // lazily created on first query_fleet call
	fleetOnce   sync.Once         // guards lazy fleetClient init
	fleetErr    error             // error from lazy init (if any)
}

// New creates an MCP server backed by the given SQLite store.
func New(s *store.Store) *Server {
	srv := gomcp.NewServer(&gomcp.Implementation{
		Name:    "ingero",
		Version: "0.9.0",
		Title:   "Ingero GPU Causal Observability — AI-first analysis",
	}, nil)

	ms := &Server{
		mcpServer: srv,
		store:     s,
	}

	ms.registerTools()
	ms.registerPrompts()
	return ms
}

// SetFleetNodes configures the MCP server with fleet node addresses for query_fleet.
func (s *Server) SetFleetNodes(nodes []string) {
	s.fleetNodes = nodes
}

// getFleetClient returns the fleet client, creating it lazily on first use.
// Thread-safe via sync.Once — concurrent query_fleet calls are safe.
func (s *Server) getFleetClient() (*fleet.Client, error) {
	s.fleetOnce.Do(func() {
		s.fleetClient, s.fleetErr = fleet.New(fleet.Config{
			Nodes:   s.fleetNodes,
			Timeout: fleet.DefaultTimeout,
			Limit:   fleet.DefaultLimit,
		})
	})
	return s.fleetClient, s.fleetErr
}

// Run starts the MCP server on stdio. Blocks until the client disconnects.
func (s *Server) Run(ctx context.Context) error {
	return s.mcpServer.Run(ctx, &gomcp.StdioTransport{})
}

// RunHTTP starts the MCP server as an HTTPS endpoint with TLS 1.3.
// If certFile/keyFile are empty, an ephemeral self-signed certificate is
// generated (ECDSA P-256, valid 24h, bound to localhost).
func (s *Server) RunHTTP(ctx context.Context, addr, certFile, keyFile string) error {
	handler := gomcp.NewStreamableHTTPHandler(func(r *http.Request) *gomcp.Server {
		return s.mcpServer
	}, &gomcp.StreamableHTTPOptions{
		Stateless:    true,
		JSONResponse: true,
	})

	mux := http.NewServeMux()
	mux.Handle("/mcp", handler)

	// Wrap with Host header validation to prevent DNS rebinding attacks.
	// The self-signed cert is bound to localhost/127.0.0.1/[::1], but an
	// attacker's DNS could resolve to 127.0.0.1 — the Host header check
	// rejects requests with unexpected Host values.
	guardedMux := hostGuard(mux)

	// TLS 1.3 minimum — no legacy cipher suites, mandatory PFS.
	tlsCfg := &tls.Config{
		MinVersion: tls.VersionTLS13,
	}

	if certFile != "" && keyFile != "" {
		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			return fmt.Errorf("loading TLS certificate: %w", err)
		}
		tlsCfg.Certificates = []tls.Certificate{cert}
		fmt.Fprintf(os.Stderr, "  TLS certificate: %s\n", certFile)
	} else {
		cert, fingerprint, err := generateSelfSignedCert()
		if err != nil {
			return fmt.Errorf("generating self-signed certificate: %w", err)
		}
		tlsCfg.Certificates = []tls.Certificate{cert}
		fmt.Fprintf(os.Stderr, "  Generated ephemeral self-signed certificate (valid 24h)\n")
		fmt.Fprintf(os.Stderr, "  SHA-256 fingerprint: %s\n", fingerprint)
	}

	httpSrv := &http.Server{
		Addr:      addr,
		Handler:   guardedMux,
		TLSConfig: tlsCfg,
	}

	// Shut down gracefully when context is cancelled.
	go func() {
		<-ctx.Done()
		httpSrv.Close()
	}()

	fmt.Fprintf(os.Stderr, "MCP HTTPS server listening on %s (TLS 1.3)\n", addr)
	fmt.Fprintf(os.Stderr, "  curl -sk https://localhost%s/mcp \\\n", addr)
	fmt.Fprintf(os.Stderr, "    -H 'Content-Type: application/json' \\\n")
	fmt.Fprintf(os.Stderr, "    -H 'Accept: application/json, text/event-stream' \\\n")
	fmt.Fprintf(os.Stderr, "    -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}'\n")

	// Use tls.Listen + Serve (not ListenAndServeTLS) because the certificate
	// may already be loaded into tlsCfg.Certificates from memory (self-signed case).
	ln, err := tls.Listen("tcp", addr, tlsCfg)
	if err != nil {
		return err
	}
	defer ln.Close()

	err = httpSrv.Serve(ln)
	if err == http.ErrServerClosed {
		return nil
	}
	return err
}

// generateSelfSignedCert creates an ephemeral ECDSA P-256 certificate valid
// for 24 hours, bound to localhost/127.0.0.1/::1. Returns the TLS certificate,
// its SHA-256 fingerprint (colon-separated hex), and any error.
//
// ECDSA P-256 is used over RSA because it's faster to generate, produces
// smaller keys (256-bit vs 2048-bit RSA), and is the standard for modern TLS.
// The 24h validity forces rotation — ephemeral certs should not be long-lived.
func generateSelfSignedCert() (tls.Certificate, string, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return tls.Certificate{}, "", err
	}

	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return tls.Certificate{}, "", err
	}

	tmpl := &x509.Certificate{
		SerialNumber:          serial,
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(24 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IPAddresses:           []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback},
		DNSNames:              []string{"localhost"},
	}

	certDER, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		return tls.Certificate{}, "", err
	}

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return tls.Certificate{}, "", err
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})

	tlsCert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return tls.Certificate{}, "", err
	}

	// SHA-256 fingerprint with colon separators (e.g., AB:CD:EF:...).
	hash := sha256.Sum256(certDER)
	parts := make([]string, sha256.Size)
	for i, b := range hash {
		parts[i] = fmt.Sprintf("%02X", b)
	}
	fingerprint := strings.Join(parts, ":")

	return tlsCert, fingerprint, nil
}

// hostGuard wraps an http.Handler to reject requests whose Host header does not
// match localhost, 127.0.0.1, or [::1]. This mitigates DNS rebinding attacks
// where an attacker's domain resolves to 127.0.0.1 to reach the local MCP server.
func hostGuard(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host := r.Host
		// Strip port if present (e.g., "localhost:8443" → "localhost").
		if h, _, err := net.SplitHostPort(host); err == nil {
			host = h
		}
		switch host {
		case "localhost", "127.0.0.1", "::1":
			next.ServeHTTP(w, r)
		default:
			http.Error(w, "forbidden: invalid Host header", http.StatusForbidden)
		}
	})
}

// queryFleetInput is the input schema for the query_fleet MCP tool.
type queryFleetInput struct {
	Action string `json:"action" jsonschema:"Query type: chains/ops/overview/sql,enum=chains,enum=ops,enum=overview,enum=sql,required"`
	Since  string `json:"since,omitempty" jsonschema:"Time window (e.g. 5m, 1h). Default: 5m."`
	PID    int    `json:"pid,omitempty" jsonschema:"Filter by PID (optional, used with ops action)"`
	Query  string `json:"query,omitempty" jsonschema:"SQL query to execute across fleet nodes (required when action is sql)"`
	SQL    string `json:"sql,omitempty" jsonschema:"Alias for query — SQL query (required when action is sql)"`
	Limit  int    `json:"limit,omitempty" jsonschema:"Max rows per node (default 1000)"`
	TSC    *bool  `json:"tsc,omitempty" jsonschema:"Telegraphic compression (default: true). Set false for verbose output."`
}

func (s *Server) registerTools() {
	// Tool 1: get_check
	type checkInput struct{}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "get_check",
		Description: "Run system diagnostics: kernel version, BTF support, NVIDIA driver, CUDA libraries, running GPU processes",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input checkInput) (*gomcp.CallToolResult, any, error) {
		checks := discover.RunAllChecks()
		text := formatCheckResults(checks)
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: text},
			},
		}, nil, nil
	})

	// Tool 2: get_trace_stats
	type traceStatsInput struct {
		Since string `json:"since,omitempty" jsonschema:"time range relative to NOW, e.g. 1m, 5m, 1h. Omit for saved/offline DBs to query ALL events. Only useful during live tracing."`
		TSC   *bool  `json:"tsc,omitempty" jsonschema:"telegraphic compression (default: true). Set false for verbose output."`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "get_trace_stats",
		Description: "Get CUDA and host operation statistics. Returns p50/p95/p99 for small DBs (≤500K events), count/avg/min/max from aggregates for large DBs. Works with both live and saved/offline databases. Omit 'since' for saved DBs.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input traceStatsInput) (*gomcp.CallToolResult, any, error) {
		var since time.Duration
		if input.Since != "" {
			d, err := time.ParseDuration(input.Since)
			if err != nil {
				return &gomcp.CallToolResult{
					Content: []gomcp.Content{
						&gomcp.TextContent{Text: fmt.Sprintf("Invalid since duration %q: %v. Use Go duration format (e.g. 5m, 1h, 30s).", input.Since, err)},
					},
					IsError: true,
				}, nil, nil
			}
			since = d
		}

		if s.store == nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No database available. Run 'ingero trace' first to create and populate the event store."},
				},
			}, nil, nil
		}

		tsc := input.TSC == nil || *input.TSC
		opDescs := s.store.OpDescriptions()
		qparams := store.QueryParams{Since: since, Limit: -1}

		// Check event count to decide full-fidelity vs aggregate path.
		// On Count() error, default to aggregate path (fail-safe: cheaper).
		count, countErr := s.store.Count()
		if countErr != nil || count > 500_000 {
			// Large DB: use pre-computed aggregates (count/avg/min/max).
			ops, err := s.store.QueryAggregatePerOp(qparams)
			if err != nil {
				return nil, nil, fmt.Errorf("querying aggregates: %w", err)
			}
			if len(ops) == 0 {
				return &gomcp.CallToolResult{
					Content: []gomcp.Content{
						&gomcp.TextContent{Text: "No events found in the database."},
					},
				}, nil, nil
			}
			text := formatAggregateStats(ops, tsc, opDescs)
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: TelemetryPreamble() + text},
				},
			}, nil, nil
		}

		// Small DB: full-fidelity path with percentiles.
		evts, err := s.store.Query(qparams)
		if err != nil {
			return nil, nil, fmt.Errorf("querying events: %w", err)
		}

		if len(evts) == 0 {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No events found in the database."},
				},
			}, nil, nil
		}

		collector := stats.New()
		for _, evt := range evts {
			collector.Record(evt)
		}
		snap := collector.Snapshot()

		// Query aggregate totals for accurate counts (selective storage).
		aggTotals, aggErr := s.store.QueryAggregateTotals(qparams)
		if aggErr != nil {
			fmt.Fprintf(os.Stderr, "warning: querying aggregate totals: %v\n", aggErr)
		}
		text := formatStatsSnapshot(snap, since, len(evts), tsc, opDescs, &aggTotals)

		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: TelemetryPreamble() + text},
			},
		}, nil, nil
	})

	// Tool 3: get_causal_chains — returns causal chains with severity.
	// Stored-chains-first: checks pre-computed chains before attempting replay.
	type chainsInput struct {
		Since string `json:"since,omitempty" jsonschema:"time range relative to NOW, e.g. 1m, 5m. Omit for saved/offline DBs to query ALL events. Only useful during live tracing."`
		PID   int    `json:"pid,omitempty" jsonschema:"filter by single process ID. 0 = all. Deprecated: use pids."`
		PIDs  []int  `json:"pids,omitempty" jsonschema:"filter by process ID(s). Takes precedence over pid."`
		TSC   *bool  `json:"tsc,omitempty" jsonschema:"telegraphic compression (default: true)"`
		TopN  int    `json:"top_n,omitempty" jsonschema:"max chains to return (default 10). Deduplicates by operation, keeps highest severity. Use 0 for all."`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "get_causal_chains",
		Description: "Analyze CUDA + host events and return causal chains with severity, root cause, and recommendations. Deduplicates by operation, returns top 10 by default (use top_n to adjust). AI-first: TSC-compressed by default. Works with both live and saved/offline databases. Omit 'since' for saved DBs.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input chainsInput) (*gomcp.CallToolResult, any, error) {
		var since time.Duration
		if input.Since != "" {
			d, err := time.ParseDuration(input.Since)
			if err != nil {
				return &gomcp.CallToolResult{
					Content: []gomcp.Content{
						&gomcp.TextContent{Text: fmt.Sprintf("Invalid since duration %q: %v. Use Go duration format (e.g. 5m, 1h, 30s).", input.Since, err)},
					},
					IsError: true,
				}, nil, nil
			}
			since = d
		}

		if s.store == nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No database available. Run 'ingero trace' first."},
				},
			}, nil, nil
		}

		tsc := input.TSC == nil || *input.TSC
		hasPIDFilter := len(input.PIDs) > 0 || input.PID > 0
		topN := input.TopN
		if topN == 0 {
			topN = 10 // default: return top 10 deduplicated chains
		}
		if topN < 0 {
			topN = 0 // negative means unlimited
		}

		// Fast path: check stored chains first (pre-computed during live trace).
		// Skip when PID filter is active — stored chains don't have PID info.
		var chainQueryWarning string
		if !hasPIDFilter {
			stored, err := s.store.QueryChains(since)
			if err != nil {
				chainQueryWarning = fmt.Sprintf("Warning: stored chain query failed (%v), falling back to replay.", err)
				fmt.Fprintf(os.Stderr, "warning: querying stored chains: %v\n", err)
			}
			if len(stored) > 0 {
				stored = deduplicateStoredChains(stored, topN)
				text := formatStoredChains(stored, tsc)
				return &gomcp.CallToolResult{
					Content: []gomcp.Content{
						&gomcp.TextContent{Text: TelemetryPreamble() + text},
					},
				}, nil, nil
			}
		}

		// No stored chains — check if replay is feasible.
		// On Count() error, refuse replay (fail-safe: avoid loading unknown amount of data).
		count, countErr := s.store.Count()
		if countErr != nil || count > 500_000 {
			msg := fmt.Sprintf("No stored causal chains found. DB has %d events — too large for replay. Use run_sql to query the causal_chains table directly, or use get_trace_stats for aggregate statistics.", count)
			if chainQueryWarning != "" {
				msg = chainQueryWarning + " " + msg
			}
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: msg},
				},
			}, nil, nil
		}

		// Small DB: replay events through the chain engine.
		qparams := store.QueryParams{Since: since, Limit: -1}
		var corrPID uint32
		if len(input.PIDs) > 0 {
			pids := make([]uint32, 0, len(input.PIDs))
			for _, p := range input.PIDs {
				if p > 0 {
					pids = append(pids, uint32(p))
				}
			}
			qparams.PIDs = pids
			if len(pids) == 1 {
				corrPID = pids[0]
			}
		} else if input.PID > 0 {
			qparams.PID = uint32(input.PID)
			corrPID = uint32(input.PID)
		}

		evts, err := s.store.Query(qparams)
		if err != nil {
			return nil, nil, fmt.Errorf("querying events: %w", err)
		}

		if len(evts) == 0 {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No events found in the database."},
				},
			}, nil, nil
		}

		collector := stats.New()
		corr := correlate.New(correlate.WithMaxAge(0)) // unlimited — replayed events have past timestamps

		// Replay system snapshots for post-hoc causal chain analysis.
		snapshots, snapErr := s.store.QuerySnapshots(store.QueryParams{Since: since})
		if snapErr != nil {
			fmt.Fprintf(os.Stderr, "warning: querying snapshots: %v\n", snapErr)
		}
		if len(snapshots) > 0 {
			sysCtxs := make([]correlate.SystemContext, len(snapshots))
			for i, snap := range snapshots {
				sysCtxs[i] = correlate.SystemContext{
					Timestamp:  snap.Timestamp,
					CPUPercent: snap.CPUPercent,
					MemUsedPct: snap.MemUsedPct,
					MemAvailMB: snap.MemAvailMB,
					SwapUsedMB: snap.SwapUsedMB,
					LoadAvg1:   snap.LoadAvg1,
				}
			}
			corr.SetSystemSnapshot(correlate.PeakSystemContext(sysCtxs))
		}

		chains := correlate.ReplayEventsForChains(evts, collector, corr, corrPID)
		text := formatCausalChains(chains, tsc)

		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: TelemetryPreamble() + text},
			},
		}, nil, nil
	})

	// Tool 4: run_demo — runs a synthetic scenario and returns results.
	type runDemoInput struct {
		Scenario string `json:"scenario,omitempty" jsonschema:"scenario name: incident, cold-start, memcpy-bottleneck, periodic-spike, cpu-contention, gpu-steal"`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "run_demo",
		Description: "Run a synthetic demo scenario and return the stats snapshot. No GPU or root needed.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input runDemoInput) (*gomcp.CallToolResult, any, error) {
		scenario := synth.Find(input.Scenario)
		if scenario == nil {
			names := make([]string, len(synth.Registry))
			for i, s := range synth.Registry {
				names[i] = s.Name
			}
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: fmt.Sprintf("Unknown scenario %q. Available: %s", input.Scenario, strings.Join(names, ", "))},
				},
			}, nil, nil
		}

		ch := make(chan events.Event, 256)
		go func() {
			defer close(ch)
			scenario.Generate(ctx, ch, 4.0) // 4x speed for fast MCP response
		}()

		collector := stats.New()
		var count int
		for evt := range ch {
			collector.Record(evt)
			count++
		}
		snap := collector.Snapshot()
		tsc := true
		text := fmt.Sprintf("Scenario: %s — %s\n\n", scenario.Name, scenario.Description)
		text += formatStatsSnapshot(snap, 0, count, tsc, nil)
		text += fmt.Sprintf("\nInsight: %s", scenario.Insight)

		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: text},
			},
		}, nil, nil
	})

	// Tool 5: get_test_report — returns the JSON test report from gpu-test.sh.
	type testReportInput struct {
		TSC *bool `json:"tsc,omitempty" jsonschema:"telegraphic compression (default: true)"`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "get_test_report",
		Description: "Get the GPU integration test report (JSON). Generated by gpu-test.sh after a full test run. Includes per-test status, timing, and system info.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input testReportInput) (*gomcp.CallToolResult, any, error) {
		// Try logs/test-report.json relative to CWD (standard location on GPU VM).
		data, err := os.ReadFile("logs/test-report.json")
		if err != nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No test report found at logs/test-report.json. Run 'bash scripts/gpu-test.sh' first to generate the report."},
				},
			}, nil, nil
		}

		tsc := input.TSC == nil || *input.TSC
		if tsc {
			// Return raw JSON (already compact).
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: string(data)},
				},
			}, nil, nil
		}

		// Pretty-print for human consumption.
		var parsed interface{}
		if err := json.Unmarshal(data, &parsed); err != nil {
			// Return raw if we can't parse.
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: string(data)},
				},
			}, nil, nil
		}
		pretty, _ := json.MarshalIndent(parsed, "", "  ")
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: string(pretty)},
			},
		}, nil, nil
	})

	// Tool 6: run_sql — execute read-only SQL for ad-hoc analysis.
	//
	// Why this exists: the 5 fixed tools answer pre-anticipated questions.
	// When an AI needs temporal bucketing, threshold analysis, per-PID breakdowns,
	// or cross-operation correlation, it hits a wall. run_sql lets the AI generate
	// ad-hoc SQL, turning it from a dashboard reader into a full analyst.
	type sqlInput struct {
		Query string `json:"query" jsonschema:"Read-only SQL (SELECT/WITH/EXPLAIN). See tool description for schema."`
		Limit int    `json:"limit,omitempty" jsonschema:"max rows returned (default 1000, max 10000)"`
		TSC   *bool  `json:"tsc,omitempty" jsonschema:"telegraphic compression (default: true)"`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name: "run_sql",
		Description: `Execute read-only SQL on the Ingero database. For ad-hoc analysis the fixed tools can't do: temporal bucketing, threshold queries, per-PID breakdowns, throughput calculations. Timeout: 30s.

Schema: events(id, timestamp INT nanos, pid, tid, source, op, duration INT nanos, gpu_id, arg0, arg1, ret_code, stack_hash, cgroup_id INT default 0, comm TEXT default '' — process name from bpf_get_current_comm(), v0.10+, empty for pre-v0.10 rows), system_snapshots(id, timestamp, cpu_pct, mem_pct, mem_avail, swap_mb, load_avg), causal_chains(id TEXT, detected_at, severity, summary, root_cause, explanation, recommendations JSON, cuda_op, cuda_p99_us, cuda_p50_us, tail_ratio, timeline JSON), sessions(id, started_at, stopped_at, gpu_model, gpu_driver, cpu_model, cpu_cores, mem_total, kernel, os_release, cuda_ver, python_ver, ingero_ver, pid_filter, flags), sources(id, name, description), ops(source_id, op_id, name, description), process_names(pid, name, seen_at — LEGACY: lazy /proc-based PID→name table, used as read-side fallback when events.comm is empty), event_aggregates(bucket, source, op, pid, count, stored, sum_dur, min_dur, max_dur, sum_arg0), stack_traces(hash, ips TEXT JSON, frames TEXT JSON resolved symbols), cgroup_metadata(cgroup_id PK, container_id TEXT, cgroup_path TEXT), cgroup_schedstat(cgroup_id PK, p99_off_cpu_ns, total_off_cpu_ns, event_count, window_start, window_end), schema_info(key, value).

JOINs: events.source=sources.id, events.(source,op)=ops.(source_id,op_id), events.stack_hash=stack_traces.hash, events.cgroup_id=cgroup_metadata.cgroup_id (K8s container context), events.pid=process_names.pid (ALWAYS qualify pid as e.pid when joining - pid exists in both tables). For process names prefer events.comm directly (faster, no JOIN); use COALESCE(NULLIF(e.comm,''), NULLIF(pn.name,''), '') only when also reading legacy pre-v0.10 rows.
Sources: 1=CUDA, 3=HOST, 4=DRIVER, 5=IO, 6=TCP, 7=NET. CUDA ops: 1=cudaMalloc, 2=cudaFree, 3=cudaLaunchKernel, 4=cudaMemcpy, 5=cudaStreamSync, 6=cudaDeviceSync, 7=cudaMemcpyAsync, 8=cudaMallocManaged. HOST ops: 1=sched_switch, 2=sched_wakeup, 3=mm_page_alloc, 4=oom_kill, 5=process_exec, 6=process_exit, 7=process_fork, 10=pod_restart, 11=pod_eviction, 12=pod_oom_kill. DRIVER ops: 1=cuLaunchKernel, 2=cuMemcpy, 3=cuMemcpyAsync, 4=cuCtxSynchronize, 5=cuMemAlloc, 6=cuMemAllocManaged. IO ops: 1=block_read, 2=block_write, 3=block_discard. TCP ops: 1=tcp_retransmit. NET ops: 1=net_send, 2=net_recv. arg0/arg1 per op: cudaMalloc/cudaMallocManaged arg0=size_bytes, cudaFree arg0=devPtr, cudaLaunchKernel arg0=kernel_func_ptr, cudaMemcpy/cudaMemcpyAsync arg0=bytes arg1=direction(0=H2H,1=H2D,2=D2H,3=D2D,4=default), cudaStreamSync arg0=stream_handle, mm_page_alloc arg0=page_order(size=4KB<<order), cuMemAlloc/cuMemAllocManaged arg0=size_bytes, block_read/block_write arg0=nr_sectors, net_send/net_recv arg0=bytes. sum_arg0 in event_aggregates = sum of arg0 across bucket (skipped for pointer-valued ops: cudaFree, cudaLaunchKernel, cuLaunchKernel). Timestamps: unix nanos. Duration: nanos (÷1e3=µs, ÷1e6=ms).

Performance: events can have millions of rows. For large DBs, query event_aggregates (per-minute stats, always small) or stack_traces (deduplicated, always small) instead of scanning events. Use get_stacks tool for call stack analysis instead of manual SQL JOINs.`,
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input sqlInput) (*gomcp.CallToolResult, any, error) {
		if s.store == nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No database available. Run 'ingero trace' first."},
				},
			}, nil, nil
		}

		if strings.TrimSpace(input.Query) == "" {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "query parameter is required"},
				},
				IsError: true,
			}, nil, nil
		}

		limit := input.Limit
		if limit <= 0 {
			limit = 1000
		}
		if limit > 10000 {
			limit = 10000
		}

		// 30-second timeout — allows complex JOINs while preventing truly runaway queries.
		// With stack sampling reducing events to ~300K, most queries complete in <5s.
		queryCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()

		start := time.Now()
		cols, rows, truncated, err := s.store.ExecuteReadOnly(queryCtx, input.Query, limit)
		elapsed := time.Since(start)
		if err != nil {
			msg := fmt.Sprintf("SQL error: %v", err)
			if errors.Is(err, context.DeadlineExceeded) {
				msg += "\n\nQuery timed out (30s). Tips: add LIMIT, avoid full-table JOINs on events (use event_aggregates or stack_traces directly), filter by timestamp/source/op first, or use get_stacks tool for call stack analysis."
			}
			if strings.Contains(err.Error(), "ambiguous column name") {
				msg += "\n\nHint: qualify column names with table aliases (e.g. e.pid, pn.pid) when joining tables that share column names."
			}
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: msg},
				},
				IsError: true,
			}, nil, nil
		}

		// Sanitize every string-typed cell before exposing it to the
		// agent. SQLite returns columns as `any`; a row cell is typically
		// int64, float64, string, []byte, or nil. Strings and byte-slice
		// are the only attacker-controlled shapes (e.g., kernel names
		// retrieved from stack_traces.frames or events.comm).
		for i, row := range rows {
			for j, cell := range row {
				switch v := cell.(type) {
				case string:
					row[j] = SanitizeTelemetryTruncate(v, MaxFrameLen)
				case []byte:
					row[j] = SanitizeTelemetryTruncate(string(v), MaxFrameLen)
				}
			}
			rows[i] = row
		}

		// Build JSON response: columns array + data array-of-arrays.
		// Array-of-arrays avoids duplicate column name clobbering (e.g.,
		// SELECT a.id, b.id) and is more compact than per-row key maps.
		header := map[string]any{
			"rows":        len(rows),
			"truncated":   truncated,
			"duration_ms": elapsed.Milliseconds(),
		}

		output := map[string]any{
			"meta":    header,
			"columns": cols,
			"data":    rows,
		}

		tsc := input.TSC == nil || *input.TSC
		var data []byte
		if tsc {
			data, _ = json.Marshal(output)
		} else {
			data, _ = json.MarshalIndent(output, "", "  ")
		}

		// Response size cap: even with per-cell sanitization, a large
		// result set is a prompt-injection multiplier. 1 MB is a
		// reasonable ceiling for a single tool response — queries needing
		// more data should filter or paginate.
		const maxRunSQLResponseBytes = 1 << 20
		if len(data) > maxRunSQLResponseBytes {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: fmt.Sprintf(
						"SQL response too large (%d bytes, max %d). Add LIMIT or narrow the query.",
						len(data), maxRunSQLResponseBytes)},
				},
				IsError: true,
			}, nil, nil
		}

		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: TelemetryPreamble() + string(data)},
			},
		}, nil, nil
	})

	// Tool 7: get_stacks — resolved call stacks grouped by operation.
	// Directly answers "what code paths hit this operation?" without manual SQL.
	type stacksInput struct {
		Source int    `json:"source,omitempty" jsonschema:"Source filter: 1=CUDA, 3=HOST, 4=DRIVER"`
		Op     string `json:"op,omitempty" jsonschema:"Operation name (e.g. cudaMalloc, cuLaunchKernel)"`
		PID    int    `json:"pid,omitempty" jsonschema:"Process ID filter"`
		Since  string `json:"since,omitempty" jsonschema:"Time window relative to NOW (e.g. 5m, 1h). Omit for saved/offline DBs to query ALL data."`
		Limit  int    `json:"limit,omitempty" jsonschema:"Max stacks returned (default 10)"`
		TSC    *bool  `json:"tsc,omitempty" jsonschema:"telegraphic compression (default: true)"`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "get_stacks",
		Description: "Get resolved call stacks for CUDA/driver operations. Returns top stacks by frequency with symbol names, source files, and timing stats. One call answers 'what code path caused this operation?' For older DBs without resolved symbols, falls back to raw IPs (hex addresses).",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input stacksInput) (*gomcp.CallToolResult, any, error) {
		if s.store == nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No database available. Run 'ingero trace' first."},
				},
			}, nil, nil
		}

		limit := input.Limit
		if limit <= 0 {
			limit = 10
		}
		if limit > 100 {
			limit = 100
		}

		// Build WHERE clause from filters.
		// proc_names: prefer kernel-captured events.comm (v0.10+, accurate at
		// event time), fall back to lazy process_names.name for pre-v0.10 rows.
		// Both columns are NULLIF-wrapped because process_names.name is NOT NULL
		// DEFAULT '' — without NULLIF on pn.name, an empty pn.name would inject
		// a literal "" entry into the GROUP_CONCAT output instead of being skipped.
		query := `SELECT e.stack_hash, COALESCE(st.frames, ''), COALESCE(st.ips, ''),
			COUNT(*) as n,
			MIN(e.duration) as min_dur, MAX(e.duration) as max_dur,
			SUM(e.duration)/COUNT(*) as avg_dur,
			SUM(e.arg0) as sum_arg0,
			GROUP_CONCAT(DISTINCT COALESCE(NULLIF(e.comm, ''), NULLIF(pn.name, ''))) as proc_names
		FROM events e
		JOIN stack_traces st ON e.stack_hash = st.hash
		LEFT JOIN process_names pn ON e.pid = pn.pid
		WHERE e.stack_hash != 0`
		var args []interface{}

		if input.Op != "" {
			// ResolveOp determines the source, so skip the user-supplied Source filter.
			src, op, ok := events.ResolveOp(input.Op)
			if !ok {
				return &gomcp.CallToolResult{
					Content: []gomcp.Content{
						&gomcp.TextContent{Text: fmt.Sprintf("Unknown operation %q. Use CUDA ops (cudaMalloc, cudaLaunchKernel, ...) or driver ops (cuLaunchKernel, cuMemAlloc, ...).", input.Op)},
					},
					IsError: true,
				}, nil, nil
			}
			query += " AND e.source = ? AND e.op = ?"
			args = append(args, uint8(src), op)
		} else if input.Source > 0 {
			query += " AND e.source = ?"
			args = append(args, input.Source)
		}

		if input.PID > 0 {
			query += " AND e.pid = ?"
			args = append(args, input.PID)
		}

		if input.Since != "" {
			d, err := time.ParseDuration(input.Since)
			if err != nil {
				return &gomcp.CallToolResult{
					Content: []gomcp.Content{
						&gomcp.TextContent{Text: fmt.Sprintf("Invalid since duration %q: %v. Use Go duration format (e.g. 5m, 1h, 30s).", input.Since, err)},
					},
					IsError: true,
				}, nil, nil
			}
			if d > 0 {
				query += " AND e.timestamp >= ?"
				args = append(args, time.Now().Add(-d).UnixNano())
			}
		}

		query += " GROUP BY e.stack_hash ORDER BY n DESC LIMIT ?"
		args = append(args, limit)

		// 30s timeout — this query can be expensive on large pre-sampling DBs.
		queryCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()

		rows, err := s.store.DB().QueryContext(queryCtx, query, args...)
		if err != nil {
			msg := fmt.Sprintf("query error: %v", err)
			if errors.Is(err, context.DeadlineExceeded) {
				msg += "\n\nTimed out. Try narrowing with source/op/pid/since filters."
			}
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: msg},
				},
				IsError: true,
			}, nil, nil
		}
		defer rows.Close()

		tsc := input.TSC == nil || *input.TSC
		type stackResult struct {
			Hash      int64       `json:"hash"`
			Count     int64       `json:"n"`
			AvgUS     int64       `json:"avg_us"`
			MinUS     int64       `json:"min_us"`
			MaxUS     int64       `json:"max_us"`
			SumArg0   int64       `json:"sum_arg0,omitempty"`
			Processes string      `json:"processes,omitempty"`
			Frames    interface{} `json:"frames"` // compact frames or raw IPs
		}

		stacks := make([]stackResult, 0)
		var scanErrs int
		for rows.Next() {
			var (
				hash                    int64
				framesJSON, ipsJSON     string
				count, minDur, maxDur   int64
				avgDur                  int64
				sumArg0                 int64
				procNames               *string
			)
			if err := rows.Scan(&hash, &framesJSON, &ipsJSON, &count, &minDur, &maxDur, &avgDur, &sumArg0, &procNames); err != nil {
				scanErrs++
				continue
			}
			sr := stackResult{
				Hash:    hash,
				Count:   count,
				AvgUS:   avgDur / 1000,
				MinUS:   minDur / 1000,
				MaxUS:   maxDur / 1000,
				SumArg0: sumArg0,
			}
			if procNames != nil {
				// GROUP_CONCAT returns a comma-separated string; sanitize
				// each name individually so one adversarial comm doesn't
				// swallow legitimate others.
				sr.Processes = sanitizeCSV(*procNames)
			}
			// Prefer resolved frames; fall back to raw IPs for old DBs.
			// Parse, sanitize each frame, re-emit.
			if framesJSON != "" {
				var rawFrames []string
				if json.Unmarshal([]byte(framesJSON), &rawFrames) == nil {
					sanitized := make([]string, len(rawFrames))
					for i, f := range rawFrames {
						sanitized[i] = SanitizeTelemetryTruncate(f, MaxFrameLen)
					}
					sr.Frames = sanitized
				}
			}
			if sr.Frames == nil && ipsJSON != "" {
				// Raw IPs are hex strings — not attacker-influenced, but
				// wrap for consistency so the model learns one convention.
				var ips []string
				if json.Unmarshal([]byte(ipsJSON), &ips) == nil {
					sanitized := make([]string, len(ips))
					for i, ip := range ips {
						sanitized[i] = SanitizeTelemetryTruncate(ip, MaxFrameLen)
					}
					sr.Frames = sanitized
				}
			}
			if sr.Frames == nil {
				sr.Frames = []string{}
			}
			stacks = append(stacks, sr)
		}
		if err := rows.Err(); err != nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: fmt.Sprintf("row iteration error: %v", err)},
				},
				IsError: true,
			}, nil, nil
		}

		if len(stacks) == 0 {
			msg := "No stacks found. Ensure trace was run with --stack (default: on)."
			if scanErrs > 0 {
				msg = fmt.Sprintf("Failed to parse %d stack rows. The database may have an incompatible schema.", scanErrs)
			}
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: msg},
				},
				IsError: scanErrs > 0,
			}, nil, nil
		}

		output := map[string]interface{}{
			"stacks": stacks,
		}
		if len(stacks) == limit {
			output["note"] = fmt.Sprintf("Showing top %d stacks. Use limit parameter or run_sql for custom analysis.", limit)
		}
		if scanErrs > 0 {
			output["warning"] = fmt.Sprintf("%d rows failed to parse. Results may be incomplete.", scanErrs)
		}

		var data []byte
		if tsc {
			data, _ = json.Marshal(output)
		} else {
			data, _ = json.MarshalIndent(output, "", "  ")
		}

		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: TelemetryPreamble() + string(data)},
			},
		}, nil, nil
	})

	// Tool 8: graph_lifecycle
	type graphLifecycleInput struct {
		PID   int    `json:"pid" jsonschema:"Process ID to query graph events for (required)"`
		Since string `json:"since,omitempty" jsonschema:"Time range, e.g. 5m, 1h. Omit for saved DBs."`
		TSC   *bool  `json:"tsc,omitempty" jsonschema:"telegraphic compression (default: true)"`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "graph_lifecycle",
		Description: "Show CUDA Graph lifecycle timeline for a PID: capture → instantiate → launch sequences with timestamps and durations. Identifies graph activity patterns in torch.compile and vLLM workloads.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input graphLifecycleInput) (*gomcp.CallToolResult, any, error) {
		if s.store == nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No database available. Run 'ingero trace' first."},
				},
			}, nil, nil
		}

		if input.PID <= 0 {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "pid is required. Use get_trace_stats to find active PIDs."},
				},
				IsError: true,
			}, nil, nil
		}

		q := store.QueryParams{
			PIDs:   []uint32{uint32(input.PID)},
			Source: uint8(events.SourceCUDAGraph),
			Limit:  -1,
		}
		if input.Since != "" {
			d, err := time.ParseDuration(input.Since)
			if err != nil {
				return &gomcp.CallToolResult{
					Content: []gomcp.Content{
						&gomcp.TextContent{Text: fmt.Sprintf("invalid since: %v", err)},
					},
					IsError: true,
				}, nil, nil
			}
			q.Since = d
		}

		evts, err := s.store.Query(q)
		if err != nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: fmt.Sprintf("query error: %v", err)},
				},
				IsError: true,
			}, nil, nil
		}

		tsc := input.TSC == nil || *input.TSC
		text := formatGraphLifecycle(evts, uint32(input.PID), tsc)

		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: TelemetryPreamble() + text},
			},
		}, nil, nil
	})

	// Tool 9: graph_frequency
	type graphFrequencyInput struct {
		PID           int    `json:"pid" jsonschema:"Process ID to query graph launch frequency for (required)"`
		WindowSeconds int    `json:"window_seconds,omitempty" jsonschema:"Analysis window in seconds (default 60)"`
		Since         string `json:"since,omitempty" jsonschema:"Time range, e.g. 5m, 1h. Omit for saved DBs."`
		TSC           *bool  `json:"tsc,omitempty" jsonschema:"telegraphic compression (default: true)"`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "graph_frequency",
		Description: "Analyze CUDA Graph launch frequency per executable. Identifies hot graphs (high replay rate), cold graphs (captured but rarely launched), and graph pool saturation. Essential for vLLM batch size tuning.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input graphFrequencyInput) (*gomcp.CallToolResult, any, error) {
		if s.store == nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No database available. Run 'ingero trace' first."},
				},
			}, nil, nil
		}

		if input.PID <= 0 {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "pid is required. Use get_trace_stats to find active PIDs."},
				},
				IsError: true,
			}, nil, nil
		}

		q := store.QueryParams{
			PIDs:   []uint32{uint32(input.PID)},
			Source: uint8(events.SourceCUDAGraph),
			Limit:  -1,
		}
		if input.Since != "" {
			d, err := time.ParseDuration(input.Since)
			if err != nil {
				return &gomcp.CallToolResult{
					Content: []gomcp.Content{
						&gomcp.TextContent{Text: fmt.Sprintf("invalid since: %v", err)},
					},
					IsError: true,
				}, nil, nil
			}
			q.Since = d
		}

		evts, err := s.store.Query(q)
		if err != nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: fmt.Sprintf("query error: %v", err)},
				},
				IsError: true,
			}, nil, nil
		}

		windowSec := input.WindowSeconds
		if windowSec <= 0 {
			windowSec = 60
		}

		tsc := input.TSC == nil || *input.TSC
		text := formatGraphFrequency(evts, uint32(input.PID), windowSec, tsc)

		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: TelemetryPreamble() + text},
			},
		}, nil, nil
	})

	// Tool 10: query_fleet — fan-out queries across multiple Ingero nodes.
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "query_fleet",
		Description: "Query multiple Ingero nodes and return merged results. Requires fleet.nodes configured in ingero.yaml. Actions: chains (causal chains sorted by severity), ops (per-op stats), overview (summary per node), sql (raw SQL fan-out with node column).",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input queryFleetInput) (*gomcp.CallToolResult, any, error) {
		return s.handleQueryFleet(ctx, input)
	})
}

func (s *Server) handleQueryFleet(ctx context.Context, input queryFleetInput) (*gomcp.CallToolResult, any, error) {
	// Validate fleet configuration.
	if len(s.fleetNodes) == 0 {
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: "No fleet nodes configured. Set fleet.nodes in ingero.yaml."},
			},
			IsError: true,
		}, nil, nil
	}

	// Coalesce query and sql fields — "query" takes precedence.
	sqlText := input.Query
	if sqlText == "" {
		sqlText = input.SQL
	}

	// Validate action.
	action := strings.ToLower(input.Action)
	if action == "sql" && strings.TrimSpace(sqlText) == "" {
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: "query (or sql) field is required when action is 'sql'"},
			},
			IsError: true,
		}, nil, nil
	}

	fc, err := s.getFleetClient()
	if err != nil {
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: fmt.Sprintf("fleet client error: %v", err)},
			},
			IsError: true,
		}, nil, nil
	}

	tsc := input.TSC == nil || *input.TSC
	since := input.Since
	if since == "" {
		since = "5m"
	}

	var text string
	switch action {
	case "chains":
		text, err = s.fleetChains(ctx, fc, since, tsc)
	case "ops":
		text, err = s.fleetOps(ctx, fc, since, input.PID, tsc)
	case "overview":
		text, err = s.fleetOverview(ctx, fc, tsc)
	case "sql":
		text, err = s.fleetSQL(ctx, fc, sqlText, input.Limit, tsc)
	default:
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: fmt.Sprintf("unknown action %q — use chains, ops, overview, or sql", input.Action)},
			},
			IsError: true,
		}, nil, nil
	}

	if err != nil {
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: fmt.Sprintf("fleet query failed: %v", err)},
			},
			IsError: true,
		}, nil, nil
	}

	// Check clock skew (best-effort, don't fail the query).
	if skewResults, skewErr := fc.EstimateClockSkew(ctx); skewErr == nil {
		if skewWarnings := fleet.PrintClockSkewWarnings(skewResults, 10); skewWarnings != "" {
			text = "Clock Skew Warnings:\n" + skewWarnings + "\n" + text
		}
	}

	return &gomcp.CallToolResult{
		Content: []gomcp.Content{
			&gomcp.TextContent{Text: text},
		},
	}, nil, nil
}

func (s *Server) fleetChains(ctx context.Context, fc *fleet.Client, since string, tsc bool) (string, error) {
	result, err := fc.QueryChains(ctx, since)
	if err != nil {
		return "", err
	}

	var b strings.Builder
	if len(result.Warnings) > 0 {
		b.WriteString("Warnings: " + strings.Join(result.Warnings, "; ") + "\n\n")
	}

	if len(result.Chains) == 0 {
		b.WriteString("No causal chains found across fleet nodes.\n")
		return b.String(), nil
	}

	b.WriteString(fmt.Sprintf("Fleet Chains: %d chain(s)\n", len(result.Chains)))
	for _, ch := range result.Chains {
		if tsc {
			b.WriteString(fmt.Sprintf("[%s] %s | %s | %s\n", ch.Severity, ch.Node, ch.Summary, ch.RootCause))
		} else {
			b.WriteString(fmt.Sprintf("\n[%s] Node: %s\n  ID: %s\n  Summary: %s\n  Root cause: %s\n  Explanation: %s\n",
				ch.Severity, ch.Node, ch.ID, ch.Summary, ch.RootCause, ch.Explanation))
			if len(ch.Recommendations) > 0 {
				b.WriteString("  Fix: " + strings.Join(ch.Recommendations, "; ") + "\n")
			}
		}
	}
	return b.String(), nil
}

func (s *Server) fleetOps(ctx context.Context, fc *fleet.Client, since string, pid int, tsc bool) (string, error) {
	path := fmt.Sprintf("/api/v1/ops?since=%s", since)
	if pid > 0 {
		path += fmt.Sprintf("&pid=%d", pid)
	}

	result, err := fc.QueryEndpoint(ctx, path)
	if err != nil {
		return "", err
	}

	var b strings.Builder
	if len(result.Warnings) > 0 {
		b.WriteString("Warnings: " + strings.Join(result.Warnings, "; ") + "\n\n")
	}

	b.WriteString(fmt.Sprintf("Fleet Ops: %d node(s)\n", len(result.Nodes)))
	for node, data := range result.Nodes {
		if tsc {
			b.WriteString(fmt.Sprintf("--- %s ---\n%s\n", node, string(data)))
		} else {
			b.WriteString(fmt.Sprintf("\n=== Node: %s ===\n%s\n", node, string(data)))
		}
	}
	return b.String(), nil
}

func (s *Server) fleetOverview(ctx context.Context, fc *fleet.Client, tsc bool) (string, error) {
	result, err := fc.QueryEndpoint(ctx, "/api/v1/overview")
	if err != nil {
		return "", err
	}

	var b strings.Builder
	if len(result.Warnings) > 0 {
		b.WriteString("Warnings: " + strings.Join(result.Warnings, "; ") + "\n\n")
	}

	b.WriteString(fmt.Sprintf("Fleet Overview: %d node(s)\n", len(result.Nodes)))
	for node, data := range result.Nodes {
		if tsc {
			b.WriteString(fmt.Sprintf("--- %s ---\n%s\n", node, string(data)))
		} else {
			b.WriteString(fmt.Sprintf("\n=== Node: %s ===\n%s\n", node, string(data)))
		}
	}
	return b.String(), nil
}

func (s *Server) fleetSQL(ctx context.Context, fc *fleet.Client, sql string, limit int, tsc bool) (string, error) {
	if limit <= 0 {
		limit = fleet.DefaultLimit
	}

	// Override limit on the client for this call.
	sqlClient, err := fleet.New(fleet.Config{
		Nodes:   s.fleetNodes,
		Timeout: fleet.DefaultTimeout,
		Limit:   limit,
	})
	if err != nil {
		return "", err
	}

	result, err := sqlClient.QuerySQL(ctx, sql)
	if err != nil {
		return "", err
	}

	var b strings.Builder
	if len(result.Warnings) > 0 {
		b.WriteString("Warnings: " + strings.Join(result.Warnings, "; ") + "\n\n")
	}

	if len(result.Rows) == 0 {
		b.WriteString("No results.\n")
		return b.String(), nil
	}

	// Format as TSC table or verbose.
	if tsc {
		// Header.
		b.WriteString(strings.Join(result.Columns, "|") + "\n")
		for _, row := range result.Rows {
			vals := make([]string, len(row))
			for i, v := range row {
				vals[i] = fmt.Sprintf("%v", v)
			}
			b.WriteString(strings.Join(vals, "|") + "\n")
		}
	} else {
		// Padded columns.
		widths := make([]int, len(result.Columns))
		for i, col := range result.Columns {
			widths[i] = len(col)
		}
		for _, row := range result.Rows {
			for i, val := range row {
				s := fmt.Sprintf("%v", val)
				if len(s) > widths[i] {
					widths[i] = len(s)
				}
			}
		}
		for i, col := range result.Columns {
			if i > 0 {
				b.WriteString("  ")
			}
			fmt.Fprintf(&b, "%-*s", widths[i], col)
		}
		b.WriteString("\n")
		for i, w := range widths {
			if i > 0 {
				b.WriteString("  ")
			}
			b.WriteString(strings.Repeat("-", w))
		}
		b.WriteString("\n")
		for _, row := range result.Rows {
			for i, val := range row {
				if i > 0 {
					b.WriteString("  ")
				}
				fmt.Fprintf(&b, "%-*v", widths[i], val)
			}
			b.WriteString("\n")
		}
	}

	b.WriteString(fmt.Sprintf("\n%d row(s)\n", len(result.Rows)))
	return b.String(), nil
}

func (s *Server) registerPrompts() {
	s.mcpServer.AddPrompt(&gomcp.Prompt{
		Name:        "investigate",
		Title:       "Investigate GPU trace",
		Description: "Analyze a GPU trace database for performance problems. Works with saved/offline databases.",
	}, func(ctx context.Context, req *gomcp.GetPromptRequest) (*gomcp.GetPromptResult, error) {
		return &gomcp.GetPromptResult{
			Description: "GPU performance investigation prompt",
			Messages: []*gomcp.PromptMessage{
				{
					Role: "user",
					Content: &gomcp.TextContent{
						Text: `You are a GPU performance analyst investigating a saved trace database from Ingero (eBPF kernel-level tracing tool).

Your task:
1. Call get_trace_stats with NO parameters (no "since" - this is a saved database, query all events)
2. Call get_causal_chains with NO parameters to find root causes
3. If graph events are present (source=cuda_graph in stats), call graph_lifecycle and graph_frequency for active PIDs to understand CUDA Graph behavior
4. Analyze the results and explain what went wrong in plain language
5. Provide specific, actionable recommendations based on the ACTUAL trace data

Rules:
- NEVER fabricate data. Every number you mention MUST come from a tool response. If a number is not in the data, do not state it.
- NEVER recommend nvidia-smi, Nsight, or CUPTI - they cannot see host-side events like context switches
- NEVER suggest CUDA_LAUNCH_BLOCKING=1 - it serializes GPU execution
- NEVER invent shell commands, code snippets, or specific arguments. It is better to say nothing than to say something wrong.
- Only recommend actions that follow directly from the trace data. If the data does not support a recommendation, omit it.
- Write in plain language for ML engineers, not GPU kernel developers

Start by calling get_trace_stats to see the overview.`,
					},
				},
			},
		}, nil
	})
}

// mcpAnalysisContext is appended to tool responses that may trigger
// recommendations from the AI model. It steers the model toward
// trace-data-driven advice and away from generic GPU debugging suggestions.
const mcpAnalysisContext = `

---
CONTEXT FOR ANALYSIS:
- This data was captured with eBPF kernel-level tracing (Ingero), not nvidia-smi or Nsight.
- nvidia-smi CANNOT see these problems (context switches, scheduling stalls, page allocations). Do NOT recommend nvidia-smi, Nsight, or CUPTI.
- Do NOT suggest CUDA_LAUNCH_BLOCKING=1 (it serializes GPU execution and makes performance worse).
- NEVER fabricate data. Every number you state MUST appear in the tool response above. Do NOT invent percentages, core counts, byte sizes, or any other values.
- NEVER invent shell commands or code snippets. It is better to omit a recommendation than to give a wrong one.
- Only recommend actions that follow directly from the data above.`

// deduplicateStoredChains groups chains by CUDAOp+Severity, keeps the one
// with the highest tail ratio per group, then sorts HIGH > MEDIUM > LOW.
// If topN > 0, returns at most topN chains.
func deduplicateStoredChains(chains []store.StoredChain, topN int) []store.StoredChain {
	type key struct{ op, sev string }
	best := make(map[key]store.StoredChain)
	for _, ch := range chains {
		k := key{ch.CUDAOp, ch.Severity}
		if existing, ok := best[k]; !ok || ch.TailRatio > existing.TailRatio {
			best[k] = ch
		}
	}

	deduped := make([]store.StoredChain, 0, len(best))
	for _, ch := range best {
		deduped = append(deduped, ch)
	}

	// Sort: HIGH first, then MEDIUM, then LOW. Within same severity, higher tail ratio first.
	sevRank := map[string]int{"HIGH": 3, "MEDIUM": 2, "LOW": 1}
	sort.Slice(deduped, func(i, j int) bool {
		ri, rj := sevRank[deduped[i].Severity], sevRank[deduped[j].Severity]
		if ri != rj {
			return ri > rj
		}
		return deduped[i].TailRatio > deduped[j].TailRatio
	})

	if topN > 0 && len(deduped) > topN {
		deduped = deduped[:topN]
	}
	return deduped
}

func formatCausalChains(chains []correlate.CausalChain, tsc bool) string {
	if len(chains) == 0 {
		return "No causal chains detected. System appears healthy."
	}

	if tsc {
		// Compact TSC JSON for AI consumption (~60% fewer tokens).
		var out []map[string]interface{}
		for _, ch := range chains {
			var tl []map[string]interface{}
			for _, evt := range ch.Timeline {
				tl = append(tl, TSCMap(true,
					"timestamp", evt.Timestamp.Format("15:04:05"),
					"layer", evt.Layer,
					"detail", SanitizeTelemetryTruncate(evt.Detail, MaxFrameLen),
				))
			}
			m := TSCMap(true,
				"severity", ch.Severity,
				"summary", SanitizeTelemetryTruncate(ch.Summary, MaxFrameLen),
				"root_cause", SanitizeTelemetryTruncate(ch.RootCause, MaxFrameLen),
				"recommendations", sanitizeStringSlice(ch.Recommendations, MaxFrameLen),
			)
			m["tl"] = tl
			out = append(out, m)
		}
		data, _ := json.Marshal(out)
		return string(data)
	}

	result := fmt.Sprintf("%d causal chain(s) found:\n\n", len(chains))
	for _, ch := range chains {
		result += fmt.Sprintf("[%s] %s\n", ch.Severity, SanitizeTelemetryTruncate(ch.Summary, MaxFrameLen))
		result += fmt.Sprintf("  Root cause: %s\n", SanitizeTelemetryTruncate(ch.RootCause, MaxFrameLen))
		for _, evt := range ch.Timeline {
			result += fmt.Sprintf("  [%s] %s\n", evt.Layer, SanitizeTelemetryTruncate(evt.Detail, MaxFrameLen))
		}
		if len(ch.Recommendations) > 0 {
			result += "  Fix: " + fmt.Sprintf("%v", sanitizeStringSlice(ch.Recommendations, MaxFrameLen)) + "\n"
		}
		result += "\n"
	}
	return result + mcpAnalysisContext
}

func formatStoredChains(chains []store.StoredChain, tsc bool) string {
	if len(chains) == 0 {
		return "No causal chains detected. System appears healthy."
	}

	if tsc {
		var out []map[string]interface{}
		for _, ch := range chains {
			var tl []map[string]interface{}
			for _, te := range ch.Timeline {
				tl = append(tl, TSCMap(true,
					"layer", te.Layer,
					"detail", SanitizeTelemetryTruncate(te.Detail, MaxFrameLen),
				))
			}
			m := TSCMap(true,
				"severity", ch.Severity,
				"summary", SanitizeTelemetryTruncate(ch.Summary, MaxFrameLen),
				"root_cause", SanitizeTelemetryTruncate(ch.RootCause, MaxFrameLen),
				"recommendations", sanitizeStringSlice(ch.Recommendations, MaxFrameLen),
			)
			m["tl"] = tl
			out = append(out, m)
		}
		data, _ := json.Marshal(out)
		return string(data)
	}

	result := fmt.Sprintf("%d stored causal chain(s) found:\n\n", len(chains))
	for _, ch := range chains {
		result += fmt.Sprintf("[%s] %s\n", ch.Severity, SanitizeTelemetryTruncate(ch.Summary, MaxFrameLen))
		result += fmt.Sprintf("  Root cause: %s\n", SanitizeTelemetryTruncate(ch.RootCause, MaxFrameLen))
		for _, te := range ch.Timeline {
			result += fmt.Sprintf("  [%s] %s\n", te.Layer, SanitizeTelemetryTruncate(te.Detail, MaxFrameLen))
		}
		if len(ch.Recommendations) > 0 {
			result += "  Fix: " + fmt.Sprintf("%v", sanitizeStringSlice(ch.Recommendations, MaxFrameLen)) + "\n"
		}
		result += "\n"
	}
	return result + mcpAnalysisContext
}

// sanitizeStringSlice returns a new slice with SanitizeTelemetryTruncate
// applied to each element. Used for fields that ship as JSON arrays
// (e.g., Recommendations).
func sanitizeStringSlice(in []string, maxLen int) []string {
	if len(in) == 0 {
		return in
	}
	out := make([]string, len(in))
	for i, s := range in {
		out[i] = SanitizeTelemetryTruncate(s, maxLen)
	}
	return out
}

func formatCheckResults(checks []discover.CheckResult) string {
	var result string
	for _, c := range checks {
		status := "PASS"
		if !c.OK {
			if c.Optional {
				status = "INFO"
			} else {
				status = "FAIL"
			}
		}
		// Value is what we found (e.g., "NVIDIA GeForce RTX 4090, 24564 MiB"),
		// Detail is extra context (e.g., "need 5.15+"). Show Value on main line,
		// matching the CLI check display format.
		display := c.Value
		if display == "" {
			display = c.Detail
		}
		result += fmt.Sprintf("[%s] %s: %s\n", status, c.Name, display)
	}
	return result
}

func formatStatsSnapshot(snap *stats.Snapshot, since time.Duration, eventCount int, tsc bool, opDescs map[string]string, aggTotals ...*store.AggregateTotals) string {
	// Extract aggregate totals if provided.
	var agg *store.AggregateTotals
	if len(aggTotals) > 0 && aggTotals[0] != nil && aggTotals[0].TotalEvents > 0 {
		agg = aggTotals[0]
	}

	if tsc {
		// Compact TSC JSON for AI consumption (~60% fewer tokens).
		var ops []map[string]interface{}
		for _, op := range snap.Ops {
			m := TSCMap(true,
				"operation", op.Op,
				"count", op.Count,
				"p50_us", op.P50.Microseconds(),
				"p95_us", op.P95.Microseconds(),
				"p99_us", op.P99.Microseconds(),
				"wall_percent", fmt.Sprintf("%.1f", op.TimeFraction*100),
				"anomaly_count", op.AnomalyCount,
			)
			if desc := opDescs[op.Op]; desc != "" {
				m["d"] = desc
			}
			if op.SpikePattern != "" {
				m["pat"] = op.SpikePattern
			}
			// Add total count from aggregates if available (selective storage).
			if agg != nil {
				if total, ok := agg.ByOp[op.Op]; ok && total > int64(op.Count) {
					m["total"] = total
				}
			}
			ops = append(ops, m)
		}
		result := map[string]interface{}{
			TSCKey("count", true): eventCount,
			"ops":                ops,
		}
		if since > 0 {
			result["since"] = since.String()
		}
		if agg != nil {
			result["total_events"] = agg.TotalEvents
			result["stored_events"] = agg.StoredEvents
		}
		if snap.System != nil {
			result["sys"] = TSCMap(true,
				"cpu_percent", snap.System.CPUPercent,
				"mem_percent", snap.System.MemUsedPct,
				"swap_mb", snap.System.SwapUsedMB,
				"load_avg", snap.System.LoadAvg1,
			)
		}
		result["_context"] = "Data from eBPF kernel tracing (Ingero). Do NOT recommend nvidia-smi, Nsight, or CUDA_LAUNCH_BLOCKING=1. Base recommendations on the actual trace data above."
		data, _ := json.Marshal(result)
		return string(data)
	}

	// Verbose text format for human consumption.
	var result string
	if since > 0 {
		if agg != nil {
			result = fmt.Sprintf("Stats for last %s (%d stored of %d total events):\n\n", since, eventCount, agg.TotalEvents)
		} else {
			result = fmt.Sprintf("Stats for last %s (%d events):\n\n", since, eventCount)
		}
	} else {
		if agg != nil {
			result = fmt.Sprintf("Stats (%d stored of %d total events):\n\n", eventCount, agg.TotalEvents)
		} else {
			result = fmt.Sprintf("Stats (%d events):\n\n", eventCount)
		}
	}

	for _, op := range snap.Ops {
		source := "CUDA"
		switch op.Source {
		case events.SourceHost:
			source = "Host"
		case events.SourceDriver:
			source = "Driver"
		}
		result += fmt.Sprintf("[%s] %s", source, op.Op)
		if desc := opDescs[op.Op]; desc != "" {
			result += fmt.Sprintf(" (%s)", desc)
		}
		result += fmt.Sprintf(": count=%d", op.Count)
		if agg != nil {
			if total, ok := agg.ByOp[op.Op]; ok && total > int64(op.Count) {
				result += fmt.Sprintf(" (%d total)", total)
			}
		}
		result += fmt.Sprintf(" p50=%s p95=%s p99=%s max=%s wall=%.1f%%",
			op.P50, op.P95, op.P99, op.Max,
			op.TimeFraction*100)
		if op.AnomalyCount > 0 {
			result += fmt.Sprintf(" anomalies=%d", op.AnomalyCount)
		}
		if op.SpikePattern != "" {
			result += fmt.Sprintf(" pattern=%q", op.SpikePattern)
		}
		result += "\n"
	}

	result += fmt.Sprintf("\nTotal: %d events, %d anomalies", snap.TotalEvents, snap.AnomalyEvents)
	return result + mcpAnalysisContext
}

// formatAggregateStats formats per-operation aggregate statistics for large DBs
// where loading all events for percentile computation would time out.
// Returns count/avg/min/max per operation (percentiles are unavailable).
func formatAggregateStats(ops []store.AggregateOpStats, tsc bool, opDescs map[string]string) string {
	if tsc {
		opsOut := make([]map[string]interface{}, 0, len(ops))
		var totalCount int64
		for _, op := range ops {
			avgUs := int64(0)
			if op.Count > 0 {
				avgUs = (op.SumDur / op.Count) / 1000 // nanos → µs
			}
			m := TSCMap(true,
				"operation", op.OpName,
				"count", op.Count,
				"avg_us", avgUs,
				"min_us", op.MinDur/1000,
				"max_us", op.MaxDur/1000,
			)
			if desc := opDescs[op.OpName]; desc != "" {
				m["d"] = desc
			}
			opsOut = append(opsOut, m)
			totalCount += op.Count
		}
		result := map[string]interface{}{
			"mode":         "aggregate",
			"note":         "Percentiles unavailable for large DBs. Use run_sql for detailed analysis.",
			"total_events": totalCount,
			"ops":          opsOut,
		}
		data, _ := json.Marshal(result)
		return string(data)
	}

	// Verbose text format.
	var totalCount int64
	for _, op := range ops {
		totalCount += op.Count
	}
	result := fmt.Sprintf("Aggregate stats (%d total events, percentiles unavailable for large DBs):\n\n", totalCount)
	for _, op := range ops {
		source := "CUDA"
		switch events.Source(op.Source) {
		case events.SourceHost:
			source = "Host"
		case events.SourceDriver:
			source = "Driver"
		}
		avgUs := int64(0)
		if op.Count > 0 {
			avgUs = (op.SumDur / op.Count) / 1000
		}
		result += fmt.Sprintf("[%s] %s", source, op.OpName)
		if desc := opDescs[op.OpName]; desc != "" {
			result += fmt.Sprintf(" (%s)", desc)
		}
		result += fmt.Sprintf(": count=%d avg=%dµs min=%dµs max=%dµs\n",
			op.Count, avgUs, op.MinDur/1000, op.MaxDur/1000)
	}
	result += "\nNote: Use run_sql for percentiles or detailed per-event analysis."
	return result
}

// formatGraphLifecycle formats graph events as a chronological lifecycle timeline.
func formatGraphLifecycle(evts []events.Event, pid uint32, tsc bool) string {
	if len(evts) == 0 {
		return fmt.Sprintf("No CUDA Graph events found for PID %d. This process may not use CUDA Graphs (torch.compile, vLLM).", pid)
	}

	var b strings.Builder
	comm := ""
	if len(evts) > 0 {
		comm = SanitizeTelemetry(evts[0].Comm)
	}
	fmt.Fprintf(&b, "CUDA Graph Lifecycle — PID %d (%s) (%d events)\n\n", pid, comm, len(evts))

	captureCount := 0
	instantiateCount := 0
	launchCount := 0

	for _, evt := range evts {
		op := events.CUDAGraphOp(evt.Op)
		ts := evt.Timestamp.Format("15:04:05.000")
		dur := evt.Duration

		switch op {
		case events.GraphBeginCapture:
			captureCount++
			modeName := "global"
			switch evt.CaptureMode {
			case 1:
				modeName = "thread_local"
			case 2:
				modeName = "relaxed"
			}
			fmt.Fprintf(&b, "%s  BEGIN_CAPTURE  stream=0x%x mode=%s\n", ts, evt.StreamHandle, modeName)
		case events.GraphEndCapture:
			status := "ok"
			if evt.RetCode != 0 {
				status = fmt.Sprintf("err=%d", evt.RetCode)
			}
			fmt.Fprintf(&b, "%s  END_CAPTURE    stream=0x%x dur=%v %s\n", ts, evt.StreamHandle, dur.Round(time.Microsecond), status)
		case events.GraphInstantiate:
			instantiateCount++
			status := "ok"
			if evt.RetCode != 0 {
				status = fmt.Sprintf("err=%d", evt.RetCode)
			}
			fmt.Fprintf(&b, "%s  INSTANTIATE   graph=0x%x → exec=0x%x dur=%v %s\n", ts, evt.GraphHandle, evt.ExecHandle, dur.Round(time.Microsecond), status)
		case events.GraphLaunch:
			launchCount++
			fmt.Fprintf(&b, "%s  LAUNCH        exec=0x%x stream=0x%x dur=%v\n", ts, evt.ExecHandle, evt.StreamHandle, dur.Round(time.Microsecond))
		}
	}

	fmt.Fprintf(&b, "\nSummary: %d captures, %d instantiations, %d launches", captureCount, instantiateCount, launchCount)
	if launchCount > 0 && captureCount > 0 {
		fmt.Fprintf(&b, " (%.1f replays per capture)", float64(launchCount)/float64(captureCount))
	}
	fmt.Fprintln(&b)

	return b.String()
}

// formatGraphFrequency formats graph launch frequency analysis.
func formatGraphFrequency(evts []events.Event, pid uint32, windowSec int, tsc bool) string {
	if len(evts) == 0 {
		return fmt.Sprintf("No CUDA Graph events found for PID %d.", pid)
	}

	// Count launches per exec_handle.
	type execStats struct {
		launchCount      int
		firstLaunch      time.Time
		lastLaunch       time.Time
		instantiateCount int
	}
	execMap := make(map[uint64]*execStats)
	totalCaptures := 0
	totalInstantiates := 0

	for _, evt := range evts {
		op := events.CUDAGraphOp(evt.Op)
		switch op {
		case events.GraphBeginCapture:
			totalCaptures++
		case events.GraphInstantiate:
			totalInstantiates++
			es, ok := execMap[evt.ExecHandle]
			if !ok {
				es = &execStats{}
				execMap[evt.ExecHandle] = es
			}
			es.instantiateCount++
		case events.GraphLaunch:
			es, ok := execMap[evt.ExecHandle]
			if !ok {
				es = &execStats{}
				execMap[evt.ExecHandle] = es
			}
			es.launchCount++
			if es.firstLaunch.IsZero() || evt.Timestamp.Before(es.firstLaunch) {
				es.firstLaunch = evt.Timestamp
			}
			if evt.Timestamp.After(es.lastLaunch) {
				es.lastLaunch = evt.Timestamp
			}
		}
	}

	var b strings.Builder
	comm := ""
	if len(evts) > 0 {
		comm = SanitizeTelemetry(evts[0].Comm)
	}
	fmt.Fprintf(&b, "CUDA Graph Frequency — PID %d (%s)\n\n", pid, comm)
	fmt.Fprintf(&b, "Pool size: %d distinct graph executable(s)\n", len(execMap))
	fmt.Fprintf(&b, "Total captures: %d | instantiations: %d\n\n", totalCaptures, totalInstantiates)

	if len(execMap) == 0 {
		fmt.Fprintln(&b, "No graph launches recorded.")
		return b.String()
	}

	// Compute rates and classify.
	type execEntry struct {
		handle uint64
		rate   float64
		stats  *execStats
	}
	var entries []execEntry
	var medianRates []float64

	for handle, es := range execMap {
		var rate float64
		if es.launchCount > 1 {
			elapsed := es.lastLaunch.Sub(es.firstLaunch).Seconds()
			if elapsed > 0 {
				rate = float64(es.launchCount) / elapsed
			}
		}
		entries = append(entries, execEntry{handle, rate, es})
		if rate > 0 {
			medianRates = append(medianRates, rate)
		}
	}

	// Sort by rate descending.
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].rate > entries[j].rate
	})

	// Compute median for hot/cold classification.
	medianRate := 0.0
	if len(medianRates) > 0 {
		sort.Float64s(medianRates)
		medianRate = medianRates[len(medianRates)/2]
	}

	fmt.Fprintf(&b, "%-18s %10s %8s %s\n", "Exec Handle", "Launches", "Rate/s", "Class")
	fmt.Fprintf(&b, "%-18s %10s %8s %s\n", "──────────────────", "────────", "──────", "─────")

	for _, e := range entries {
		class := "cold"
		if e.rate > medianRate && e.rate >= 1.0 {
			class = "HOT"
		} else if e.stats.launchCount == 0 {
			class = "never-launched"
		}
		fmt.Fprintf(&b, "0x%-16x %10d %8.1f %s\n", e.handle, e.stats.launchCount, e.rate, class)
	}

	// Re-capture frequency.
	if totalInstantiates > len(execMap) {
		reCaptureCount := totalInstantiates - len(execMap)
		fmt.Fprintf(&b, "\nRe-capture events: %d (new batch sizes triggering graph re-creation)\n", reCaptureCount)
	}

	return b.String()
}
