package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/spf13/cobra"

	perfetto "github.com/ingero-io/ingero/internal/export"
	"github.com/ingero-io/ingero/internal/fleet"
	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/pkg/events"
)

var (
	exportFormat     string
	exportDBPath     string
	exportOutput     string
	exportNodes      string
	exportSince      string
	exportPIDs       []int
	exportLimit      int
	exportTimeout    string
	exportCACert     string
	exportClientCert string
	exportClientKey  string
)

var exportCmd = &cobra.Command{
	Use:   "export",
	Short: "Export event data to visualization formats",
	Long: `Export Ingero event data to external visualization formats.

Currently supports Perfetto/Chrome Trace Event Format for timeline visualization
in https://ui.perfetto.dev or chrome://tracing.

Modes:
  Local DB:    ingero export --format perfetto -d ingero.db -o trace.json
  Merged DB:   ingero export --format perfetto -d cluster.db -o trace.json
  Fan-out:     ingero export --format perfetto --nodes host1:8080,host2:8080 -o trace.json

Examples:
  ingero export --format perfetto -d ~/.ingero/ingero.db -o trace.json
  ingero export --format perfetto -d cluster.db -o trace.json --since 5m
  ingero export --format perfetto --nodes node-1:8080,node-2:8080 -o trace.json`,

	RunE: exportRunE,
}

func init() {
	exportCmd.Flags().StringVar(&exportFormat, "format", "", "output format (required: perfetto)")
	exportCmd.MarkFlagRequired("format")
	exportCmd.Flags().StringVarP(&exportDBPath, "db", "d", "", "database path (local or merged)")
	exportCmd.Flags().StringVarP(&exportOutput, "output", "o", "", "output file path (required)")
	exportCmd.MarkFlagRequired("output")
	exportCmd.Flags().StringVar(&exportNodes, "nodes", "", "comma-separated node addresses for fan-out export")
	exportCmd.Flags().StringVar(&exportSince, "since", "1h", "time range filter")
	exportCmd.Flags().IntSliceVarP(&exportPIDs, "pid", "p", nil, "filter by process ID(s)")
	exportCmd.Flags().IntVar(&exportLimit, "limit", 1000, "max events per node (fan-out mode)")
	exportCmd.Flags().StringVar(&exportTimeout, "timeout", "10s", "per-node timeout (fan-out mode)")
	exportCmd.Flags().StringVar(&exportCACert, "ca-cert", "", "CA certificate for mTLS (optional)")
	exportCmd.Flags().StringVar(&exportClientCert, "client-cert", "", "client certificate for mTLS (optional)")
	exportCmd.Flags().StringVar(&exportClientKey, "client-key", "", "client key for mTLS (optional)")

	rootCmd.AddCommand(exportCmd)
}

func exportRunE(cmd *cobra.Command, args []string) error {
	if exportFormat != "perfetto" {
		return fmt.Errorf("unsupported format %q — currently only 'perfetto' is supported", exportFormat)
	}

	nodes := resolveFleetNodes(exportNodes)
	hasDB := exportDBPath != ""
	hasNodes := len(nodes) > 0

	if !hasDB && !hasNodes {
		return fmt.Errorf("either --db or --nodes (or fleet.nodes in config) is required")
	}
	if hasDB && hasNodes {
		return fmt.Errorf("specify either --db or --nodes, not both")
	}

	var evts []events.Event
	var chains []store.StoredChain

	if hasDB {
		var err error
		evts, chains, err = loadFromDB(exportDBPath)
		if err != nil {
			return err
		}
	} else {
		var err error
		evts, chains, err = loadFromFleet(cmd.Context(), nodes)
		if err != nil {
			return err
		}
	}

	// Write output.
	f, err := os.Create(exportOutput)
	if err != nil {
		return fmt.Errorf("creating output file: %w", err)
	}
	defer f.Close()

	if err := perfetto.WritePerfetto(evts, chains, f); err != nil {
		return fmt.Errorf("writing Perfetto trace: %w", err)
	}

	info, _ := f.Stat()
	sizeMB := float64(0)
	if info != nil {
		sizeMB = float64(info.Size()) / (1024 * 1024)
	}
	fmt.Fprintf(os.Stderr, "  Exported %d events + %d chains → %s (%.1f MB)\n",
		len(evts), len(chains), exportOutput, sizeMB)

	return nil
}

func loadFromDB(dbPath string) ([]events.Event, []store.StoredChain, error) {
	s, err := store.New(dbPath)
	if err != nil {
		return nil, nil, fmt.Errorf("opening database: %w", err)
	}
	defer s.Close()

	since, err := parseSince(exportSince)
	if err != nil {
		return nil, nil, err
	}

	params := store.QueryParams{
		Since: since,
		PIDs:  toUint32Slice(exportPIDs),
		Limit: -1, // all events in range
	}

	evts, err := s.Query(params)
	if err != nil {
		return nil, nil, fmt.Errorf("querying events: %w", err)
	}

	chains, err := s.QueryChains(since)
	if err != nil {
		return nil, nil, fmt.Errorf("querying chains: %w", err)
	}

	return evts, chains, nil
}

func loadFromFleet(ctx context.Context, nodes []string) ([]events.Event, []store.StoredChain, error) {
	timeout, err := time.ParseDuration(exportTimeout)
	if err != nil {
		return nil, nil, fmt.Errorf("invalid --timeout: %w", err)
	}

	client, err := fleet.New(fleet.Config{
		Nodes:      nodes,
		Timeout:    timeout,
		Limit:      exportLimit,
		CACert:     exportCACert,
		ClientCert: exportClientCert,
		ClientKey:  exportClientKey,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("creating fleet client: %w", err)
	}

	// Fetch events via SQL fan-out.
	since, err := parseSince(exportSince)
	if err != nil {
		return nil, nil, err
	}
	cutoff := time.Now().Add(-since).UnixNano()

	sql := fmt.Sprintf("SELECT timestamp, pid, tid, source, op, duration, gpu_id, arg0, arg1, ret_code, node, rank, local_rank, world_size FROM events WHERE timestamp > %d ORDER BY timestamp", cutoff)
	qResult, err := client.QuerySQL(ctx, sql)
	if err != nil {
		if qResult != nil {
			for _, w := range qResult.Warnings {
				fmt.Fprintf(os.Stderr, "WARNING: %s\n", w)
			}
		}
		return nil, nil, err
	}
	for _, w := range qResult.Warnings {
		fmt.Fprintf(os.Stderr, "WARNING: %s\n", w)
	}

	// Convert fleet rows to events.
	var evts []events.Event
	for _, row := range qResult.Rows {
		evt := fleetRowToEvent(row, qResult.Columns)
		if evt != nil {
			evts = append(evts, *evt)
		}
	}

	// Fetch chains.
	cResult, err := client.QueryChains(ctx, exportSince)
	if err != nil {
		// Non-fatal — export without chains.
		fmt.Fprintf(os.Stderr, "WARNING: chain query failed: %v\n", err)
	}

	var chains []store.StoredChain
	if cResult != nil {
		for _, c := range cResult.Chains {
			chains = append(chains, store.StoredChain{
				ID:       c.ID,
				Severity: c.Severity,
				Summary:  c.Summary,
				RootCause: c.RootCause,
				Node:     c.Node,
			})
		}
	}

	return evts, chains, nil
}

// fleetRowToEvent converts a fleet query row to an Event.
// Column order: node, timestamp, pid, tid, source, op, duration, gpu_id, arg0, arg1, ret_code, node, rank, local_rank, world_size
// (first column is "node" prepended by fleet client)
func fleetRowToEvent(row []any, cols []string) *events.Event {
	if len(row) < 11 {
		return nil
	}

	// Helper to extract numeric value from any type.
	toInt64 := func(v any) int64 {
		switch x := v.(type) {
		case float64:
			return int64(x)
		case int64:
			return x
		case json.Number:
			n, _ := x.Int64()
			return n
		default:
			return 0
		}
	}
	toString := func(v any) string {
		if s, ok := v.(string); ok {
			return s
		}
		return ""
	}

	// row[0] is the fleet-prepended "node" column (address).
	// row[1..] are the SQL columns.
	idx := 1 // skip fleet node column
	evt := events.Event{
		Timestamp: time.Unix(0, toInt64(row[idx])),   // timestamp
		PID:       uint32(toInt64(row[idx+1])),         // pid
		TID:       uint32(toInt64(row[idx+2])),         // tid
		Source:    events.Source(toInt64(row[idx+3])),   // source
		Op:        uint8(toInt64(row[idx+4])),           // op
		Duration:  time.Duration(toInt64(row[idx+5])),   // duration (nanos)
		GPUID:     uint32(toInt64(row[idx+6])),         // gpu_id
		Args:      [2]uint64{uint64(toInt64(row[idx+7])), uint64(toInt64(row[idx+8]))},
		RetCode:   int32(toInt64(row[idx+9])),          // ret_code
	}

	// Node and rank columns.
	if len(row) > idx+10 {
		evt.Node = toString(row[idx+10])
	}
	if evt.Node == "" {
		evt.Node = toString(row[0]) // fallback to fleet address
	}

	return &evt
}
