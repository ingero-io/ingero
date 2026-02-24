// Package mcp provides an MCP (Model Context Protocol) server for Ingero.
//
// The MCP server exposes six tools to AI agents:
//   - get_check: Run system diagnostics (kernel, BTF, NVIDIA, CUDA)
//   - get_trace_stats: Get recent CUDA/host stats from the SQLite database
//   - query_events: Query stored events with filters
//   - get_causal_chains: Analyze events and return causal chains with severity
//   - run_demo: Run a synthetic demo scenario
//   - get_test_report: Return the GPU integration test report (JSON)
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
	"fmt"
	"math/big"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	gomcp "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/ingero-io/ingero/internal/correlate"
	"github.com/ingero-io/ingero/internal/discover"
	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/internal/synth"
	"github.com/ingero-io/ingero/pkg/events"
)

// Server wraps the MCP server and its dependencies.
type Server struct {
	mcpServer *gomcp.Server
	store     *store.Store
}

// New creates an MCP server backed by the given SQLite store.
func New(s *store.Store) *Server {
	srv := gomcp.NewServer(&gomcp.Implementation{
		Name:    "ingero",
		Version: "0.6.0",
		Title:   "Ingero GPU Causal Observability — AI-first analysis",
	}, nil)

	ms := &Server{
		mcpServer: srv,
		store:     s,
	}

	ms.registerTools()
	return ms
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

func (s *Server) registerTools() {
	// Tool 1: get_check
	type checkInput struct{}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "get_check",
		Description: "Run system diagnostics: kernel version, BTF support, NVIDIA driver, CUDA libraries, running GPU processes",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input checkInput) (*gomcp.CallToolResult, struct{}, error) {
		checks := discover.RunAllChecks()
		text := formatCheckResults(checks)
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: text},
			},
		}, struct{}{}, nil
	})

	// Tool 2: get_trace_stats
	type traceStatsInput struct {
		Since string `json:"since,omitempty" jsonschema:"time range to query, e.g. 1m, 5m, 1h. Default: 1m"`
		TSC   *bool  `json:"tsc,omitempty" jsonschema:"telegraphic compression (default: true). Set false for verbose output."`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "get_trace_stats",
		Description: "Get CUDA and host operation statistics from stored events. Returns p50/p95/p99 latency, time fractions, and anomaly counts.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input traceStatsInput) (*gomcp.CallToolResult, struct{}, error) {
		since := 1 * time.Minute
		if input.Since != "" {
			d, err := time.ParseDuration(input.Since)
			if err == nil {
				since = d
			}
		}

		if s.store == nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No database available. Run 'ingero trace' first to create and populate the event store."},
				},
			}, struct{}{}, nil
		}

		evts, err := s.store.Query(store.QueryParams{Since: since})
		if err != nil {
			return nil, struct{}{}, fmt.Errorf("querying events: %w", err)
		}

		if len(evts) == 0 {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: fmt.Sprintf("No events found in the last %s. Is 'ingero trace' running?", since)},
				},
			}, struct{}{}, nil
		}

		collector := stats.New()
		for _, evt := range evts {
			collector.Record(evt)
		}
		snap := collector.Snapshot()
		tsc := input.TSC == nil || *input.TSC

		// Enrich stats with op descriptions from lookup tables.
		var opDescs map[string]string
		if s.store != nil {
			opDescs = s.store.OpDescriptions()
		}
		text := formatStatsSnapshot(snap, since, len(evts), tsc, opDescs)

		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: text},
			},
		}, struct{}{}, nil
	})

	// Tool 3: query_events
	type queryInput struct {
		Since string `json:"since,omitempty" jsonschema:"time range, e.g. 5m, 1h. Default: 1h"`
		PID   int    `json:"pid,omitempty" jsonschema:"filter by process ID. 0 = all"`
		Op    string `json:"op,omitempty" jsonschema:"filter by operation name, e.g. cudaMemcpy, sched_switch"`
		Limit int    `json:"limit,omitempty" jsonschema:"max results. Default: 100"`
		TSC   *bool  `json:"tsc,omitempty" jsonschema:"telegraphic compression (default: true)"`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "query_events",
		Description: "Query raw events from the Ingero database. Returns individual event records with timestamps, durations, and operation details.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input queryInput) (*gomcp.CallToolResult, struct{}, error) {
		since := 1 * time.Hour
		if input.Since != "" {
			d, err := time.ParseDuration(input.Since)
			if err == nil {
				since = d
			}
		}

		limit := 100
		if input.Limit > 0 {
			limit = input.Limit
		}

		if s.store == nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No database available. Run 'ingero trace' first."},
				},
			}, struct{}{}, nil
		}

		params := store.QueryParams{
			Since: since,
			PID:   uint32(input.PID),
			Limit: limit,
		}

		richEvts, err := s.store.QueryRich(params)
		if err != nil {
			return nil, struct{}{}, fmt.Errorf("querying events: %w", err)
		}

		tsc := input.TSC == nil || *input.TSC
		text := formatEventList(richEvts, since, tsc)
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: text},
			},
		}, struct{}{}, nil
	})

	// Tool 4: get_causal_chains — returns active causal chains with severity.
	type chainsInput struct {
		Since string `json:"since,omitempty" jsonschema:"time range, e.g. 1m, 5m. Default: 1m"`
		PID   int    `json:"pid,omitempty" jsonschema:"filter by process ID. 0 = all"`
		TSC   *bool  `json:"tsc,omitempty" jsonschema:"telegraphic compression (default: true)"`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "get_causal_chains",
		Description: "Analyze CUDA + host events and return causal chains with severity, root cause, and recommendations. AI-first: TSC-compressed by default.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input chainsInput) (*gomcp.CallToolResult, struct{}, error) {
		since := 1 * time.Minute
		if input.Since != "" {
			d, err := time.ParseDuration(input.Since)
			if err == nil {
				since = d
			}
		}

		if s.store == nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No database available. Run 'ingero trace' first."},
				},
			}, struct{}{}, nil
		}

		evts, err := s.store.Query(store.QueryParams{Since: since, PID: uint32(input.PID)})
		if err != nil {
			return nil, struct{}{}, fmt.Errorf("querying events: %w", err)
		}

		if len(evts) == 0 {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No events found. Is 'ingero trace' running?"},
				},
			}, struct{}{}, nil
		}

		collector := stats.New()
		corr := correlate.New(correlate.WithMaxAge(0)) // unlimited — replayed events have past timestamps

		// Replay system snapshots for post-hoc causal chain analysis.
		snapshots, _ := s.store.QuerySnapshots(store.QueryParams{Since: since})
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

		// Incremental replay: process events chronologically and snapshot
		// chains at 1-second boundaries. Preserves temporal dynamics so
		// baseline→anomaly transitions produce high tail ratios (matching
		// live trace behavior). Single-pass replay averages them away.
		chains := correlate.ReplayEventsForChains(evts, collector, corr, uint32(input.PID))
		tsc := input.TSC == nil || *input.TSC
		text := formatCausalChains(chains, tsc)

		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: text},
			},
		}, struct{}{}, nil
	})

	// Tool 5: run_demo — runs a synthetic scenario and returns results.
	type runDemoInput struct {
		Scenario string `json:"scenario,omitempty" jsonschema:"scenario name: incident, cold-start, memcpy-bottleneck, periodic-spike, cpu-contention, gpu-steal"`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "run_demo",
		Description: "Run a synthetic demo scenario and return the stats snapshot. No GPU or root needed.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input runDemoInput) (*gomcp.CallToolResult, struct{}, error) {
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
			}, struct{}{}, nil
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
		}, struct{}{}, nil
	})

	// Tool 6: get_test_report — returns the JSON test report from gpu-test.sh.
	type testReportInput struct {
		TSC *bool `json:"tsc,omitempty" jsonschema:"telegraphic compression (default: true)"`
	}
	gomcp.AddTool(s.mcpServer, &gomcp.Tool{
		Name:        "get_test_report",
		Description: "Get the GPU integration test report (JSON). Generated by gpu-test.sh after a full test run. Includes per-test status, timing, and system info.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input testReportInput) (*gomcp.CallToolResult, struct{}, error) {
		// Try logs/test-report.json relative to CWD (standard location on GPU VM).
		data, err := os.ReadFile("logs/test-report.json")
		if err != nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: "No test report found at logs/test-report.json. Run 'bash scripts/gpu-test.sh' first to generate the report."},
				},
			}, struct{}{}, nil
		}

		tsc := input.TSC == nil || *input.TSC
		if tsc {
			// Return raw JSON (already compact).
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: string(data)},
				},
			}, struct{}{}, nil
		}

		// Pretty-print for human consumption.
		var parsed interface{}
		if err := json.Unmarshal(data, &parsed); err != nil {
			// Return raw if we can't parse.
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{
					&gomcp.TextContent{Text: string(data)},
				},
			}, struct{}{}, nil
		}
		pretty, _ := json.MarshalIndent(parsed, "", "  ")
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{
				&gomcp.TextContent{Text: string(pretty)},
			},
		}, struct{}{}, nil
	})
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
					"detail", evt.Detail,
				))
			}
			m := TSCMap(true,
				"severity", ch.Severity,
				"summary", ch.Summary,
				"root_cause", ch.RootCause,
				"recommendations", ch.Recommendations,
			)
			m["tl"] = tl
			out = append(out, m)
		}
		data, _ := json.Marshal(out)
		return string(data)
	}

	result := fmt.Sprintf("%d causal chain(s) found:\n\n", len(chains))
	for _, ch := range chains {
		result += fmt.Sprintf("[%s] %s\n", ch.Severity, ch.Summary)
		result += fmt.Sprintf("  Root cause: %s\n", ch.RootCause)
		for _, evt := range ch.Timeline {
			result += fmt.Sprintf("  [%s] %s\n", evt.Layer, evt.Detail)
		}
		if len(ch.Recommendations) > 0 {
			result += "  Fix: " + fmt.Sprintf("%v", ch.Recommendations) + "\n"
		}
		result += "\n"
	}
	return result
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

func formatStatsSnapshot(snap *stats.Snapshot, since time.Duration, eventCount int, tsc bool, opDescs map[string]string) string {
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
			ops = append(ops, m)
		}
		result := map[string]interface{}{
			TSCKey("count", true):  eventCount,
			"since":               since.String(),
			"ops":                  ops,
		}
		if snap.System != nil {
			result["sys"] = TSCMap(true,
				"cpu_percent", snap.System.CPUPercent,
				"mem_percent", snap.System.MemUsedPct,
				"swap_mb", snap.System.SwapUsedMB,
				"load_avg", snap.System.LoadAvg1,
			)
		}
		data, _ := json.Marshal(result)
		return string(data)
	}

	// Verbose text format for human consumption.
	result := fmt.Sprintf("Stats for last %s (%d events):\n\n", since, eventCount)

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
		result += fmt.Sprintf(": count=%d p50=%s p95=%s p99=%s max=%s wall=%.1f%%",
			op.Count,
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
	return result
}

func formatEventList(evts []store.RichEvent, since time.Duration, tsc bool) string {
	if len(evts) == 0 {
		return fmt.Sprintf("No events found in the last %s.", since)
	}

	if tsc {
		// Compact TSC JSON — abbreviated keys, enriched with lookup table data.
		var output []map[string]interface{}
		for _, evt := range evts {
			m := TSCMap(true,
				"ts", evt.Timestamp.Format("15:04:05"),
				"pid", evt.PID,
				"src", evt.SourceName,
				"op", evt.OpName,
				"dur_us", evt.Duration.Microseconds(),
			)
			if evt.OpDesc != "" {
				m["d"] = evt.OpDesc
			}
			output = append(output, m)
		}
		data, _ := json.Marshal(output)
		return string(data)
	}

	// Verbose JSON — full field names, enriched with lookup table descriptions.
	type jsonEvt struct {
		Timestamp  string `json:"timestamp"`
		PID        uint32 `json:"pid"`
		Source     string `json:"source"`
		SourceDesc string `json:"source_desc"`
		Op         string `json:"op"`
		OpDesc     string `json:"op_desc"`
		DurationUs int64  `json:"duration_us"`
	}

	var output []jsonEvt
	for _, evt := range evts {
		output = append(output, jsonEvt{
			Timestamp:  evt.Timestamp.Format(time.RFC3339Nano),
			PID:        evt.PID,
			Source:     evt.SourceName,
			SourceDesc: evt.SourceDesc,
			Op:         evt.OpName,
			OpDesc:     evt.OpDesc,
			DurationUs: evt.Duration.Microseconds(),
		})
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	return string(data)
}
