package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ingero-io/ingero/internal/fleet"
	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/internal/sysinfo"
	"github.com/ingero-io/ingero/pkg/events"
)

var (
	queryDBPath    string
	querySince     string
	queryPIDs      []int
	queryOp        string
	queryJSON      bool
	queryLimit     int
	queryNodes     string
	queryTimeout   string
	queryCACert    string
	queryClientCert string
	queryClientKey  string
	queryClockSkew  string
)

var queryCmd = &cobra.Command{
	Use:   "query [sql]",
	Short: "Query stored events from the SQLite database",
	Long: `Query events previously recorded with 'ingero trace'.

When --nodes is specified (or fleet.nodes is configured in ingero.yaml),
the query is fanned out to each node's dashboard API and results are
concatenated with a "node" column prepended.

Examples:
  ingero query --since 1h
  ingero query --since 1h --pid 4821
  ingero query --since 1h --op cudaMemcpy --json
  ingero query --since 30m --limit 100
  ingero query --nodes host1:8443,host2:8443 "SELECT source, count(*) FROM events GROUP BY source"`,

	RunE: queryRunE,
}

func init() {
	queryCmd.Flags().StringVar(&queryDBPath, "db", "", "database path (default: ~/.ingero/ingero.db)")
	queryCmd.Flags().StringVar(&querySince, "since", "1h", "query events from the last duration (e.g., 30m, 1h, 24h)")
	queryCmd.Flags().IntSliceVarP(&queryPIDs, "pid", "p", nil, "filter by process ID(s), comma-separated (default: all)")
	queryCmd.Flags().StringVar(&queryOp, "op", "", "filter by operation name (e.g., cudaMemcpy, sched_switch)")
	queryCmd.Flags().BoolVar(&queryJSON, "json", false, "output as JSON")
	queryCmd.Flags().IntVar(&queryLimit, "limit", 0, "max results (0 = 10000, applies per-node for fleet queries)")
	queryCmd.Flags().StringVar(&queryNodes, "nodes", "", "comma-separated node addresses (host:port) for fleet fan-out query")
	queryCmd.Flags().StringVar(&queryTimeout, "timeout", "5s", "per-node timeout for fleet queries")
	queryCmd.Flags().StringVar(&queryCACert, "ca-cert", "", "CA certificate for mTLS (optional)")
	queryCmd.Flags().StringVar(&queryClientCert, "client-cert", "", "client certificate for mTLS (optional)")
	queryCmd.Flags().StringVar(&queryClientKey, "client-key", "", "client key for mTLS (optional)")
	queryCmd.Flags().StringVar(&queryClockSkew, "clock-skew-threshold", "10ms", "clock skew warning threshold for fleet queries")

	rootCmd.AddCommand(queryCmd)
}

func queryRunE(cmd *cobra.Command, args []string) error {
	// Resolve fleet nodes: CLI --nodes > config fleet.nodes > empty (local mode).
	nodes := resolveFleetNodes(queryNodes)

	// Fleet fan-out path: if nodes are configured and a SQL argument is provided.
	if len(nodes) > 0 && len(args) > 0 {
		return queryFleetSQL(cmd.Context(), nodes, args[0])
	}
	if len(nodes) > 0 && len(args) == 0 {
		return fmt.Errorf("fleet query requires a SQL argument: ingero query --nodes host:port \"SELECT ...\"")
	}

	// Local path: query local SQLite (existing behavior).
	dbPath := queryDBPath
	if dbPath == "" {
		dbPath = store.DefaultDBPath()
	}
	debugf("query: db=%s since=%s pids=%v op=%q limit=%d", dbPath, querySince, queryPIDs, queryOp, queryLimit)

	s, err := store.New(dbPath)
	if err != nil {
		return fmt.Errorf("opening database at %s: %w\nHint: run 'ingero trace' first to create the database", dbPath, err)
	}
	defer s.Close()

	since, err := parseSince(querySince)
	if err != nil {
		return err
	}

	// Build query params.
	params := store.QueryParams{
		Since: since,
		PIDs:  toUint32Slice(queryPIDs),
		Limit: queryLimit,
	}

	// Resolve op name to Source+Op code.
	if queryOp != "" {
		source, op, ok := events.ResolveOp(queryOp)
		if !ok {
			return fmt.Errorf("unknown operation %q (examples: cudaMemcpy, cuLaunchKernel, sched_switch)", queryOp)
		}
		params.Source = uint8(source)
		params.Op = op
	}

	evts, err := s.Query(params)
	if err != nil {
		return fmt.Errorf("querying events: %w", err)
	}

	// Query aggregate totals for accurate event counts (selective storage).
	aggTotals, _ := s.QueryAggregateTotals(params)
	debugf("query: returned %d stored events (aggregates: %d total)", len(evts), aggTotals.TotalEvents)

	if queryJSON {
		return queryOutputJSON(evts)
	}
	return queryOutputTable(evts, params, dbPath, &aggTotals)
}

// queryFleetSQL fans out a SQL query to multiple nodes and displays concatenated results.
func queryFleetSQL(ctx context.Context, nodes []string, sql string) error {
	timeout, err := time.ParseDuration(queryTimeout)
	if err != nil {
		return fmt.Errorf("invalid --timeout: %w", err)
	}

	limit := queryLimit
	if limit <= 0 {
		limit = fleet.DefaultLimit
	}

	client, err := fleet.New(fleet.Config{
		Nodes:      nodes,
		Timeout:    timeout,
		Limit:      limit,
		CACert:     queryCACert,
		ClientCert: queryClientCert,
		ClientKey:  queryClientKey,
	})
	if err != nil {
		return fmt.Errorf("creating fleet client: %w", err)
	}

	// Run query and clock skew estimation in parallel.
	type queryOut struct {
		result *fleet.QueryResult
		err    error
	}
	qch := make(chan queryOut, 1)
	go func() {
		r, e := client.QuerySQL(ctx, sql)
		qch <- queryOut{r, e}
	}()

	skewResults, _ := client.EstimateClockSkew(ctx)
	qo := <-qch

	result := qo.result

	// Print warnings to stderr.
	if result != nil {
		for _, w := range result.Warnings {
			fmt.Fprintf(os.Stderr, "WARNING: %s\n", w)
		}
	}

	// Print clock skew warnings.
	skewThreshold := parseClockSkewThresholdMs(queryClockSkew)
	if skewWarnings := fleet.PrintClockSkewWarnings(skewResults, skewThreshold); skewWarnings != "" {
		fmt.Fprint(os.Stderr, skewWarnings)
	}

	if qo.err != nil {
		return qo.err
	}

	if queryJSON {
		return fleetOutputJSON(result)
	}
	return fleetOutputTable(result)
}

// fleetOutputTable renders a fleet query result as a text table.
func fleetOutputTable(result *fleet.QueryResult) error {
	if len(result.Rows) == 0 {
		fmt.Println("  No results.")
		return nil
	}

	// Compute column widths.
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

	// Print header.
	var header strings.Builder
	for i, col := range result.Columns {
		if i > 0 {
			header.WriteString("  ")
		}
		fmt.Fprintf(&header, "%-*s", widths[i], col)
	}
	fmt.Println(header.String())
	// Separator.
	var sep strings.Builder
	for i, w := range widths {
		if i > 0 {
			sep.WriteString("  ")
		}
		sep.WriteString(strings.Repeat("-", w))
	}
	fmt.Println(sep.String())

	// Print rows.
	for _, row := range result.Rows {
		var line strings.Builder
		for i, val := range row {
			if i > 0 {
				line.WriteString("  ")
			}
			fmt.Fprintf(&line, "%-*v", widths[i], val)
		}
		fmt.Println(line.String())
	}

	fmt.Fprintf(os.Stderr, "\n  %d rows from %d node(s)\n", len(result.Rows), countNodes(result))
	return nil
}

func countNodes(r *fleet.QueryResult) int {
	seen := make(map[string]bool)
	for _, row := range r.Rows {
		if len(row) > 0 {
			if n, ok := row[0].(string); ok {
				seen[n] = true
			}
		}
	}
	return len(seen)
}

// fleetOutputJSON renders a fleet query result as JSON.
func fleetOutputJSON(result *fleet.QueryResult) error {
	out := make([]map[string]any, 0, len(result.Rows))
	for _, row := range result.Rows {
		m := make(map[string]any, len(result.Columns))
		for i, col := range result.Columns {
			if i < len(row) {
				m[col] = row[i]
			}
		}
		out = append(out, m)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

// resolveFleetNodes returns the node list from CLI flag or config.
// CLI --nodes takes precedence over config fleet.nodes.
func resolveFleetNodes(cliFlag string) []string {
	if cliFlag != "" {
		var nodes []string
		for _, n := range strings.Split(cliFlag, ",") {
			n = strings.TrimSpace(n)
			if n != "" {
				nodes = append(nodes, n)
			}
		}
		return nodes
	}
	return ReadFleetNodes()
}

func queryOutputJSON(evts []events.Event) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")

	type jsonEvt struct {
		Timestamp  string           `json:"timestamp"`
		PID        uint32           `json:"pid"`
		TID        uint32           `json:"tid"`
		Source     string           `json:"source"`
		Op         string           `json:"op"`
		DurationNs int64            `json:"duration_ns"`
		Duration   string           `json:"duration"`
		GPUID      uint32           `json:"gpu_id,omitempty"`
		Args       [2]uint64        `json:"args"`
		RetCode    int32            `json:"return_code,omitempty"`
		CGroupID   uint64           `json:"cgroup_id,omitempty"`
		Stack      []jsonStackFrame `json:"stack,omitempty"`
	}

	output := make([]jsonEvt, 0, len(evts))
	for _, evt := range evts {
		je := jsonEvt{
			Timestamp:  evt.Timestamp.Format(time.RFC3339Nano),
			PID:        evt.PID,
			TID:        evt.TID,
			Source:     evt.Source.String(),
			Op:         evt.OpName(),
			DurationNs: evt.Duration.Nanoseconds(),
			Duration:   formatDuration(evt.Duration),
			GPUID:      evt.GPUID,
			Args:       evt.Args,
			RetCode:    evt.RetCode,
			CGroupID:   evt.CGroupID,
		}
		if len(evt.Stack) > 0 {
			je.Stack = make([]jsonStackFrame, len(evt.Stack))
			for i, f := range evt.Stack {
				je.Stack[i] = jsonStackFrame{
					Symbol: f.SymbolName,
					File:   f.File,
					Line:   f.Line,
					PyFile: f.PyFile,
					PyFunc: f.PyFunc,
					PyLine: f.PyLine,
				}
				// Include IP only if no symbol resolved (raw IPs from old DBs).
				if f.SymbolName == "" && f.IP != 0 {
					je.Stack[i].IP = fmt.Sprintf("0x%x", f.IP)
				}
			}
		}
		output = append(output, je)
	}

	return enc.Encode(output)
}

func queryOutputTable(evts []events.Event, params store.QueryParams, dbPath string, aggTotals ...*store.AggregateTotals) error {
	if len(evts) == 0 {
		fmt.Println("  No events found.")
		fmt.Printf("  Database: %s\n", dbPath)
		fmt.Printf("  Time range: last %s\n", params.Since)
		return nil
	}

	// Compute stats over the query results.
	collector := stats.New()
	for _, evt := range evts {
		collector.Record(evt)
	}
	snap := collector.Snapshot()

	fmt.Printf("  Query: last %s", params.Since)
	if len(params.PIDs) == 1 {
		fmt.Printf(" | PID %d", params.PIDs[0])
	} else if len(params.PIDs) > 1 {
		pidStrs := make([]string, len(params.PIDs))
		for i, p := range params.PIDs {
			pidStrs[i] = fmt.Sprintf("%d", p)
		}
		fmt.Printf(" | PIDs %s", strings.Join(pidStrs, ","))
	}
	// Show aggregate totals if selective storage was active.
	if len(aggTotals) > 0 && aggTotals[0] != nil && aggTotals[0].TotalEvents > 0 {
		fmt.Printf(" | %d events (%d total, selective storage)\n\n", len(evts), aggTotals[0].TotalEvents)
	} else {
		fmt.Printf(" | %d events\n\n", len(evts))
	}

	// Print stats table.
	var b strings.Builder
	var lines int

	var cudaOps, driverOps, hostOps []stats.OpStats
	for _, op := range snap.Ops {
		switch op.Source {
		case events.SourceHost:
			hostOps = append(hostOps, op)
		case events.SourceDriver:
			driverOps = append(driverOps, op)
		default:
			cudaOps = append(cudaOps, op)
		}
	}

	if len(cudaOps) > 0 {
		renderOpsSection(&b, &lines, "CUDA Runtime API", cudaOps)
	}
	if len(driverOps) > 0 {
		if len(cudaOps) > 0 {
			fmt.Fprintf(&b, "\n")
		}
		renderOpsSection(&b, &lines, "CUDA Driver API", driverOps)
	}
	if len(hostOps) > 0 {
		if len(cudaOps) > 0 || len(driverOps) > 0 {
			fmt.Fprintf(&b, "\n")
		}
		renderOpsSection(&b, &lines, "Host Context", hostOps)
	}

	// Current system context (instant read — memory, load, swap).
	sysColl := sysinfo.New()
	sys := sysColl.ReadOnce()
	fmt.Fprintf(&b, "\n  System: Mem %.0f%% (%d MB free) | Load %.1f",
		sys.MemUsedPct, sys.MemAvailMB, sys.LoadAvg1)
	if sys.SwapUsedMB > 0 {
		fmt.Fprintf(&b, " | Swap %d MB", sys.SwapUsedMB)
	}
	fmt.Fprintf(&b, "\n")

	fmt.Print(b.String())
	fmt.Println()

	return nil
}

// parseClockSkewThresholdMs parses the --clock-skew-threshold flag as milliseconds.
func parseClockSkewThresholdMs(s string) float64 {
	d, err := time.ParseDuration(s)
	if err != nil {
		return 10 // default 10ms
	}
	return float64(d) / float64(time.Millisecond)
}
