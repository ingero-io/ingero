package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/internal/sysinfo"
	"github.com/ingero-io/ingero/pkg/events"
)

var (
	queryDBPath string
	querySince  time.Duration
	queryPIDs   []int
	queryOp     string
	queryJSON   bool
	queryLimit  int
)

var queryCmd = &cobra.Command{
	Use:   "query",
	Short: "Query stored events from the SQLite database",
	Long: `Query events previously recorded with 'ingero trace'.

Examples:
  ingero query --since 1h
  ingero query --since 1h --pid 4821
  ingero query --since 1h --op cudaMemcpy --json
  ingero query --since 30m --limit 100`,

	RunE: queryRunE,
}

func init() {
	queryCmd.Flags().StringVar(&queryDBPath, "db", "", "database path (default: ~/.ingero/ingero.db)")
	queryCmd.Flags().DurationVar(&querySince, "since", 1*time.Hour, "query events from the last duration (e.g., 30m, 1h, 24h)")
	queryCmd.Flags().IntSliceVarP(&queryPIDs, "pid", "p", nil, "filter by process ID(s), comma-separated (default: all)")
	queryCmd.Flags().StringVar(&queryOp, "op", "", "filter by operation name (e.g., cudaMemcpy, sched_switch)")
	queryCmd.Flags().BoolVar(&queryJSON, "json", false, "output as JSON")
	queryCmd.Flags().IntVar(&queryLimit, "limit", 0, "max results (0 = 10000)")

	rootCmd.AddCommand(queryCmd)
}

func queryRunE(cmd *cobra.Command, args []string) error {
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

	// Build query params.
	params := store.QueryParams{
		Since: querySince,
		PIDs:  toUint32Slice(queryPIDs),
		Limit: queryLimit,
	}

	// Resolve op name to Source+Op code.
	if queryOp != "" {
		source, op, ok := events.ResolveOp(queryOp)
		if ok {
			params.Source = uint8(source)
			params.Op = op
		}
	}

	evts, err := s.Query(params)
	if err != nil {
		return fmt.Errorf("querying events: %w", err)
	}
	debugf("query: returned %d events", len(evts))

	if queryJSON {
		return queryOutputJSON(evts)
	}
	return queryOutputTable(evts, params)
}

func queryOutputJSON(evts []events.Event) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")

	type jsonStackFrame struct {
		IP string `json:"ip"`
	}
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
		Stack      []jsonStackFrame `json:"stack,omitempty"`
	}

	var output []jsonEvt
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
		}
		if len(evt.Stack) > 0 {
			je.Stack = make([]jsonStackFrame, len(evt.Stack))
			for i, f := range evt.Stack {
				je.Stack[i] = jsonStackFrame{IP: fmt.Sprintf("0x%x", f.IP)}
			}
		}
		output = append(output, je)
	}

	return enc.Encode(output)
}

func queryOutputTable(evts []events.Event, params store.QueryParams) error {
	if len(evts) == 0 {
		fmt.Println("  No events found.")
		fmt.Printf("  Database: %s\n", store.DefaultDBPath())
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
	fmt.Printf(" | %d events\n\n", len(evts))

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
