// Package cli — explain.go implements `ingero explain`, which produces
// automated incident reports with multi-layer causal chains.
//
// Reads from the SQLite database populated by `ingero trace`.
// No eBPF probes, no root required.
//
// Call chain: explainRunE → open DB → query events →
//   replay through stats + correlator → renderIncidentReport
package cli

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ingero-io/ingero/internal/correlate"
	"github.com/ingero-io/ingero/internal/discover"
	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/internal/sysinfo"
)

var (
	explainDBPath     string
	explainPIDs       []int
	explainSince      time.Duration
	explainFrom       string
	explainTo         string
	explainJSON       bool
	explainLast       int
	explainChains     bool
	explainPerProcess bool
)

var explainCmd = &cobra.Command{
	Use:   "explain",
	Short: "Automated root cause analysis with causal chains",
	Long: `Analyze recorded CUDA + host events and produce an incident report with
causal chains, root cause identification, and fix recommendations.

Reads from the SQLite database populated by 'ingero trace'. No root needed.

  ingero explain                    # analyze last 5 minutes
  ingero explain --since 1h        # last hour
  ingero explain --last 100        # last 100 events
  ingero explain --pid 4821        # filter by process
  ingero explain --chains          # show stored causal chains (no re-analysis)
  ingero explain --per-process     # per-process CUDA API breakdown (RAG/multi-process)
  ingero explain --from "15:40" --to "15:45"  # absolute time range`,

	RunE: explainRunE,
}

func init() {
	explainCmd.Flags().StringVar(&explainDBPath, "db", "", "database path (default: ~/.ingero/ingero.db)")
	explainCmd.Flags().IntSliceVarP(&explainPIDs, "pid", "p", nil, "filter by process ID(s), comma-separated (default: all)")
	explainCmd.Flags().DurationVar(&explainSince, "since", 5*time.Minute, "analyze events from the last duration")
	explainCmd.Flags().StringVar(&explainFrom, "from", "", "start time (e.g., '2026-02-20 15:40' or '15:40')")
	explainCmd.Flags().StringVar(&explainTo, "to", "", "end time (e.g., '2026-02-20 15:45' or '15:45')")
	explainCmd.Flags().BoolVar(&explainJSON, "json", false, "output as JSON")
	explainCmd.Flags().IntVar(&explainLast, "last", 0, "analyze the N most recent events (0 = use --since)")
	explainCmd.Flags().BoolVar(&explainChains, "chains", false, "show stored causal chains from DB (skip re-analysis)")
	explainCmd.Flags().BoolVar(&explainPerProcess, "per-process", false, "per-process CUDA API breakdown (RAG/multi-process contention)")

	rootCmd.AddCommand(explainCmd)
}

func resolveExplainDB() string {
	if explainDBPath != "" {
		return explainDBPath
	}
	return store.DefaultDBPath()
}

func explainRunE(cmd *cobra.Command, args []string) error {
	// --chains mode: show pre-computed causal chains from DB.
	if explainChains {
		return explainStoredChains()
	}

	// --per-process mode: per-process CUDA API breakdown.
	if explainPerProcess {
		return explainPerProcessBreakdown()
	}

	// Open DB.
	s, err := store.New(resolveExplainDB())
	if err != nil {
		return fmt.Errorf("opening database: %w\n\nHint: run 'ingero trace' first to collect events", err)
	}
	defer s.Close()

	// Build query params. Unlimited scan — explain needs all events
	// in the time range for accurate anomaly detection.
	params := store.QueryParams{
		Limit: -1,
	}

	if explainLast > 0 {
		// --last N: fetch N most recent events.
		params.Limit = explainLast
	} else if explainFrom != "" || explainTo != "" {
		// --from/--to: absolute time range.
		if explainFrom != "" {
			t, err := parseTime(explainFrom)
			if err != nil {
				return fmt.Errorf("parsing --from: %w", err)
			}
			params.From = t
		}
		if explainTo != "" {
			t, err := parseTime(explainTo)
			if err != nil {
				return fmt.Errorf("parsing --to: %w", err)
			}
			params.To = t
		}
	} else {
		// Default: --since duration.
		params.Since = explainSince
	}

	params.PIDs = toUint32Slice(explainPIDs)

	evts, err := s.Query(params)
	if err != nil {
		return fmt.Errorf("querying events: %w", err)
	}

	if len(evts) == 0 {
		fmt.Println("  No events found in the specified time range.")
		fmt.Println("  Run 'ingero trace' first to collect events.")
		return nil
	}

	debugf("explain: replaying %d stored events", len(evts))

	// Query aggregate totals so the report shows accurate total event counts
	// even when selective storage discarded most individual events.
	aggTotals, _ := s.QueryAggregateTotals(params)
	if aggTotals.TotalEvents > 0 {
		debugf("explain: aggregates show %d total events (%d stored individually)",
			aggTotals.TotalEvents, aggTotals.StoredEvents)
	}

	// Replay events through stats + correlator.
	// maxAge=0 disables pruning — replayed events have past timestamps and
	// would all be discarded by a time.Now()-based cutoff.
	collector := stats.New()
	corr := correlate.New(correlate.WithMaxAge(0))
	procCache := discover.NewProcCache()

	// Replay system snapshots for post-hoc causal chain analysis.
	// When --last N is used, params has no time filter, so scope snapshot
	// query to the actual event time range. Otherwise PeakSystemContext
	// would pick peaks from the entire DB history.
	snapshotParams := params
	if explainLast > 0 && len(evts) > 0 {
		snapshotParams.From = evts[0].Timestamp
		snapshotParams.To = evts[len(evts)-1].Timestamp
	}
	snapshots, _ := s.QuerySnapshots(snapshotParams)
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

	// Incremental replay with 1-second windowed chain detection.
	// Preserves temporal dynamics (baseline→anomaly transition).
	chains := correlate.ReplayEventsForChains(evts, collector, corr, singlePIDOrZero(explainPIDs))
	snap := collector.Snapshot()

	debugf("explain: %d events → %d causal chains", len(evts), len(chains))

	if explainJSON {
		return renderChainsJSON(chains)
	}
	renderIncidentReport(chains, snap, nil, procCache, toUint32Slice(explainPIDs), &aggTotals)
	return nil
}

// explainStoredChains renders pre-computed causal chains from the DB
// without re-analyzing events.
func explainStoredChains() error {
	s, err := store.New(resolveExplainDB())
	if err != nil {
		return fmt.Errorf("opening database: %w\n\nHint: run 'ingero trace' first to collect events", err)
	}
	defer s.Close()

	stored, err := s.QueryChains(explainSince)
	if err != nil {
		return fmt.Errorf("querying chains: %w", err)
	}

	if len(stored) == 0 {
		fmt.Println("  No stored causal chains found.")
		fmt.Println("  Run 'ingero trace' to collect events and detect chains.")
		return nil
	}

	if explainJSON {
		return renderStoredChainsJSON(stored)
	}

	// Render stored chains in human-readable format.
	var b strings.Builder
	b.WriteString(fmt.Sprintf("STORED CAUSAL CHAINS — %d chain(s)\n\n", len(stored)))

	for _, ch := range stored {
		b.WriteString(fmt.Sprintf("[%s] %s\n", ch.Severity, ch.Summary))
		b.WriteString(fmt.Sprintf("  Detected: %s\n", ch.DetectedAt.Local().Format("2006-01-02 15:04:05")))
		if len(ch.Timeline) > 0 {
			b.WriteString("  Timeline:\n")
			for _, te := range ch.Timeline {
				detail := te.Detail
				if te.DurationUS > 0 {
					detail += fmt.Sprintf(" (%s)", formatDuration(time.Duration(te.DurationUS)*time.Microsecond))
				}
				b.WriteString(fmt.Sprintf("    [%-6s]  %s\n", te.Layer, detail))
			}
		}
		b.WriteString(fmt.Sprintf("  Root cause: %s\n", ch.RootCause))
		if len(ch.Recommendations) > 0 {
			b.WriteString("  Fix: ")
			b.WriteString(strings.Join(ch.Recommendations, "; "))
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}

	fmt.Print(b.String())
	return nil
}

// explainPerProcessBreakdown shows per-process CUDA API usage from aggregates.
// Useful for RAG pipelines and multi-process GPU sharing diagnosis.
func explainPerProcessBreakdown() error {
	s, err := store.New(resolveExplainDB())
	if err != nil {
		return fmt.Errorf("opening database: %w\n\nHint: run 'ingero trace' first to collect events", err)
	}
	defer s.Close()

	params := store.QueryParams{}
	if explainLast > 0 {
		params.Limit = explainLast
	} else if explainFrom != "" || explainTo != "" {
		if explainFrom != "" {
			t, err := parseTime(explainFrom)
			if err != nil {
				return fmt.Errorf("parsing --from: %w", err)
			}
			params.From = t
		}
		if explainTo != "" {
			t, err := parseTime(explainTo)
			if err != nil {
				return fmt.Errorf("parsing --to: %w", err)
			}
			params.To = t
		}
	} else {
		params.Since = explainSince
	}
	params.PIDs = toUint32Slice(explainPIDs)

	perProc, err := s.QueryAggregatePerProcess(params)
	if err != nil {
		return fmt.Errorf("querying per-process stats: %w", err)
	}

	if len(perProc) == 0 {
		fmt.Println("  No per-process aggregate data found.")
		fmt.Println("  Run 'ingero trace' first to collect events.")
		return nil
	}

	if explainJSON {
		return renderPerProcessJSON(perProc)
	}
	renderPerProcessReport(perProc, s)
	return nil
}

func renderPerProcessReport(stats []store.ProcessOpStats, s *store.Store) {
	var b strings.Builder
	b.WriteString("PER-PROCESS GPU API BREAKDOWN\n\n")

	// Group by PID.
	type pidGroup struct {
		pid      uint32
		name     string
		ops      []store.ProcessOpStats
		totalOps int64
	}
	groups := make(map[uint32]*pidGroup)
	var order []uint32
	for _, st := range stats {
		g, ok := groups[st.PID]
		if !ok {
			g = &pidGroup{pid: st.PID}
			groups[st.PID] = g
			order = append(order, st.PID)
		}
		g.ops = append(g.ops, st)
		g.totalOps += st.Count
	}

	// Resolve process names from DB.
	for _, pid := range order {
		g := groups[pid]
		var name string
		s.DB().QueryRow("SELECT name FROM process_names WHERE pid = ?", pid).Scan(&name)
		g.name = name
	}

	// Render each process.
	for _, pid := range order {
		g := groups[pid]
		if g.name != "" {
			b.WriteString(fmt.Sprintf("  PID %d (%s) — %d total ops\n", pid, g.name, g.totalOps))
		} else {
			b.WriteString(fmt.Sprintf("  PID %d — %d total ops\n", pid, g.totalOps))
		}
		for _, op := range g.ops {
			avgNs := int64(0)
			if op.Count > 0 {
				avgNs = op.SumDur / op.Count
			}
			b.WriteString(fmt.Sprintf("    %-24s %8d calls   avg=%-10s max=%s\n",
				op.OpName, op.Count,
				formatDuration(time.Duration(avgNs)),
				formatDuration(time.Duration(op.MaxDur))))
		}
		b.WriteString("\n")
	}

	// Multi-process contention summary.
	cudaProcs := 0
	for _, pid := range order {
		g := groups[pid]
		for _, op := range g.ops {
			if op.Source == 1 || op.Source == 4 { // CUDA or Driver
				cudaProcs++
				break
			}
		}
	}
	if cudaProcs > 1 {
		b.WriteString(fmt.Sprintf("  GPU CONTENTION: %d processes sharing GPU with concurrent CUDA calls\n", cudaProcs))
		b.WriteString("  Recommendation: check for serialization on default stream, consider CUDA MPS\n")
	}

	fmt.Print(b.String())
}

func renderPerProcessJSON(stats []store.ProcessOpStats) error {
	fmt.Println("[")
	for i, st := range stats {
		avgNs := int64(0)
		if st.Count > 0 {
			avgNs = st.SumDur / st.Count
		}
		fmt.Printf("  {\"pid\":%d,\"op\":%q,\"count\":%d,\"avg_ns\":%d,\"max_ns\":%d}",
			st.PID, st.OpName, st.Count, avgNs, st.MaxDur)
		if i < len(stats)-1 {
			fmt.Println(",")
		} else {
			fmt.Println()
		}
	}
	fmt.Println("]")
	return nil
}

// renderStoredChainsJSON outputs stored chains as JSON.
func renderStoredChainsJSON(chains []store.StoredChain) error {
	fmt.Println("[")
	for i, ch := range chains {
		fmt.Printf("  {\"id\":%q,\"severity\":%q,\"summary\":%q,\"root_cause\":%q,\"detected_at\":%q}",
			ch.ID, ch.Severity, ch.Summary, ch.RootCause, ch.DetectedAt.Format(time.RFC3339))
		if i < len(chains)-1 {
			fmt.Println(",")
		} else {
			fmt.Println()
		}
	}
	fmt.Println("]")
	return nil
}

// renderIncidentReport prints a human-readable incident report with causal chains.
// aggTotals is optional — when non-nil and populated, shows accurate total event
// counts from aggregates (selective storage may have discarded most individual events).
func renderIncidentReport(chains []correlate.CausalChain, snap *stats.Snapshot, sys *sysinfo.SystemSnapshot, procCache *discover.ProcCache, pids []uint32, aggTotals ...*store.AggregateTotals) {
	var b strings.Builder

	// Header.
	highCount := 0
	medCount := 0
	lowCount := 0
	for _, ch := range chains {
		switch ch.Severity {
		case "HIGH":
			highCount++
		case "MEDIUM":
			medCount++
		case "LOW":
			lowCount++
		}
	}

	// Extract aggregate totals if provided.
	var agg *store.AggregateTotals
	if len(aggTotals) > 0 && aggTotals[0] != nil && aggTotals[0].TotalEvents > 0 {
		agg = aggTotals[0]
	}

	b.WriteString("INCIDENT REPORT")
	if len(chains) == 0 {
		b.WriteString(" — no causal chains detected\n\n")
		if agg != nil {
			b.WriteString(fmt.Sprintf("  Analyzed %d events (%d stored) over %s. No anomalous patterns found.\n",
				agg.TotalEvents, snap.TotalEvents, formatDuration(snap.WallClock)))
		} else {
			b.WriteString(fmt.Sprintf("  Analyzed %d events over %s. No anomalous patterns found.\n",
				snap.TotalEvents, formatDuration(snap.WallClock)))
		}
	} else {
		b.WriteString(fmt.Sprintf(" — %d causal chain(s) found", len(chains)))
		var parts []string
		if highCount > 0 {
			parts = append(parts, fmt.Sprintf("%d HIGH", highCount))
		}
		if medCount > 0 {
			parts = append(parts, fmt.Sprintf("%d MEDIUM", medCount))
		}
		if lowCount > 0 {
			parts = append(parts, fmt.Sprintf("%d LOW", lowCount))
		}
		if len(parts) > 0 {
			b.WriteString(fmt.Sprintf(" (%s)", strings.Join(parts, ", ")))
		}
		b.WriteString("\n")
	}

	// System context summary (if available).
	if sys != nil {
		b.WriteString(fmt.Sprintf("\n  System: CPU %.0f%% | Mem %.0f%% (%d MB free) | Load %.1f | Swap %d MB\n",
			sys.CPUPercent, sys.MemUsedPct, sys.MemAvailMB, sys.LoadAvg1, sys.SwapUsedMB))
	}

	// Process info.
	if len(pids) == 1 {
		b.WriteString(fmt.Sprintf("  Process: %s\n", procCache.FormatPID(pids[0])))
	} else if len(pids) > 1 {
		parts := make([]string, len(pids))
		for i, p := range pids {
			parts[i] = procCache.FormatPID(p)
		}
		b.WriteString(fmt.Sprintf("  Processes: %s\n", strings.Join(parts, ", ")))
	}

	b.WriteString("\n")

	// Render each chain.
	for _, ch := range chains {
		b.WriteString(fmt.Sprintf("[%s] %s\n", ch.Severity, ch.Summary))
		b.WriteString("  Timeline:\n")
		for _, evt := range ch.Timeline {
			detail := evt.Detail
			if evt.Duration > 0 {
				detail += fmt.Sprintf(" (%s)", formatDuration(evt.Duration))
			}
			b.WriteString(fmt.Sprintf("    [%-6s]  %s\n", evt.Layer, detail))
		}
		b.WriteString(fmt.Sprintf("\n  Root cause: %s\n", ch.RootCause))
		if len(ch.Recommendations) > 0 {
			b.WriteString("  Fix: ")
			b.WriteString(strings.Join(ch.Recommendations, "; "))
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}

	fmt.Print(b.String())
}

// renderChainsJSON outputs causal chains as JSON.
func renderChainsJSON(chains []correlate.CausalChain) error {
	// Simple JSON output using fmt since we don't need encoding/json import
	// overhead for this rarely-used path.
	fmt.Println("[")
	for i, ch := range chains {
		fmt.Printf("  {\"id\":%q,\"severity\":%q,\"summary\":%q,\"root_cause\":%q}",
			ch.ID, ch.Severity, ch.Summary, ch.RootCause)
		if i < len(chains)-1 {
			fmt.Println(",")
		} else {
			fmt.Println()
		}
	}
	fmt.Println("]")
	return nil
}

// parseTime parses a time string in common formats.
func parseTime(s string) (time.Time, error) {
	formats := []string{
		"2006-01-02 15:04:05",
		"2006-01-02 15:04",
		"2006-01-02T15:04:05",
		"15:04:05",
		"15:04",
	}
	for _, f := range formats {
		if t, err := time.Parse(f, s); err == nil {
			// If only time was given, assume today.
			if t.Year() == 0 {
				now := time.Now()
				t = time.Date(now.Year(), now.Month(), now.Day(), t.Hour(), t.Minute(), t.Second(), 0, time.Local)
			}
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("cannot parse %q — expected format: '2006-01-02 15:04' or '15:04'", s)
}

// resolvePIDsForUser finds all PIDs owned by the given username.
func resolvePIDsForUser(username string) ([]int, error) {
	// Read /etc/passwd to resolve username → UID, then scan /proc.
	// For simplicity, use os/user.Lookup if available.
	procs, err := discover.FindCUDAProcesses()
	if err != nil {
		return nil, err
	}

	var pids []int
	for _, p := range procs {
		// In a full implementation, we'd check /proc/<pid>/status for Uid.
		// For now, include all CUDA processes when --user is specified.
		pids = append(pids, p.PID)
	}
	return pids, nil
}
