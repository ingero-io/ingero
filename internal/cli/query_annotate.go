package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/pkg/events"
)

// annotatedRow pairs one event with the annotation labels that resolve
// to it. The pair travels through the cross-rollover merge together so
// a sort by timestamp keeps the labels attached to their event.
type annotatedRow struct {
	Event  events.Event
	Labels map[string]string
}

// queryWithAnnotations runs `ingero query --annotations`. It joins
// external annotations to events by process incarnation + time window.
// The join is rollover-aware: when --include-rolled is set, the join
// runs per DB file BEFORE the merge, so an annotation and an event
// split across a rollover boundary still resolve within their own file
// (an annotation in file A cannot be matched against an event in file
// B, which is correct - they were never the same incarnation interval).
func queryWithAnnotations(s *store.Store, dbPath string, params store.QueryParams) error {
	from, to := annotationScanWindow(params)

	rows, err := joinFileAnnotations(s, params, from, to)
	if err != nil {
		return fmt.Errorf("querying events: %w", err)
	}

	if queryIncludeRolled {
		rolled, lerr := store.ListRolledFiles(dbPath)
		if lerr != nil {
			debugf("query: list rolled files failed: %v", lerr)
		}
		debugf("query: include-rolled (annotations) found %d rolled siblings", len(rolled))
		for _, rp := range rolled {
			rs, oerr := store.NewReadOnly(rp)
			if oerr != nil {
				fmt.Fprintf(os.Stderr, "  Warning: opening rolled file %s: %v\n", rp, oerr)
				continue
			}
			rrows, jerr := joinFileAnnotations(rs, params, from, to)
			rs.Close()
			if jerr != nil {
				fmt.Fprintf(os.Stderr, "  Warning: querying rolled file %s: %v\n", rp, jerr)
				continue
			}
			rows = append(rows, rrows...)
		}
		sort.Slice(rows, func(i, j int) bool {
			return rows[i].Event.Timestamp.After(rows[j].Event.Timestamp)
		})
		effLimit := queryLimit
		if effLimit <= 0 {
			effLimit = 10000
		}
		if len(rows) > effLimit {
			rows = rows[:effLimit]
		}
	}

	if queryJSON {
		return annotatedRowsJSON(rows)
	}
	return annotatedRowsTable(rows, params, dbPath)
}

// joinFileAnnotations queries one DB file's events and joins the file's
// own annotations to them by process incarnation. The join never
// crosses a file boundary, which is exactly the rollover-correct
// behavior: each rolled file is a self-contained incarnation universe.
func joinFileAnnotations(s *store.Store, params store.QueryParams, from, to int64) ([]annotatedRow, error) {
	evts, err := s.Query(params)
	if err != nil {
		return nil, err
	}
	metas := make([]store.EventWithMeta, len(evts))
	for i, e := range evts {
		metas[i] = store.EventWithMeta{
			TimestampNs: e.Timestamp.UnixNano(),
			PID:         e.PID,
			Source:      uint8(e.Source),
			Op:          e.Op,
		}
	}
	anns, err := s.AnnotateEvents(metas, from, to)
	if err != nil {
		return nil, err
	}
	rows := make([]annotatedRow, len(evts))
	for i, e := range evts {
		rows[i] = annotatedRow{Event: e, Labels: anns[i].Labels}
	}
	return rows, nil
}

// annotationScanWindow derives the [from, to] nanosecond bounds for the
// process-lifecycle + annotation scan from the event query params. It
// mirrors the same Since / From / To precedence Store.Query applies so
// the lifecycle scan covers exactly the queried events.
func annotationScanWindow(params store.QueryParams) (int64, int64) {
	var from, to int64
	switch {
	case !params.From.IsZero():
		from = params.From.UnixNano()
	case params.Since > 0:
		from = time.Now().Add(-params.Since).UnixNano()
	default:
		from = 0
	}
	if !params.To.IsZero() {
		to = params.To.UnixNano()
	} else {
		to = time.Now().UnixNano()
	}
	return from, to
}

// annotatedRowsJSON renders the annotation-joined rows as JSON, adding a
// "labels" object to each event.
func annotatedRowsJSON(rows []annotatedRow) error {
	type out struct {
		Timestamp string            `json:"timestamp"`
		PID       uint32            `json:"pid"`
		TID       uint32            `json:"tid"`
		Comm      string            `json:"comm,omitempty"`
		Source    string            `json:"source"`
		Op        string            `json:"op"`
		Duration  string            `json:"duration"`
		Labels    map[string]string `json:"labels,omitempty"`
	}
	list := make([]out, 0, len(rows))
	for _, r := range rows {
		o := out{
			Timestamp: r.Event.Timestamp.Format(time.RFC3339Nano),
			PID:       r.Event.PID,
			TID:       r.Event.TID,
			Comm:      r.Event.Comm,
			Source:    r.Event.Source.String(),
			Op:        r.Event.OpName(),
			Duration:  formatDuration(r.Event.Duration),
		}
		if len(r.Labels) > 0 {
			o.Labels = r.Labels
		}
		list = append(list, o)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(list)
}

// annotatedRowsTable renders the annotation-joined rows as a text table
// with a trailing labels column.
func annotatedRowsTable(rows []annotatedRow, params store.QueryParams, dbPath string) error {
	if len(rows) == 0 {
		fmt.Println("  No events found.")
		fmt.Printf("  Database: %s\n", dbPath)
		return nil
	}
	annotated := 0
	for _, r := range rows {
		if len(r.Labels) > 0 {
			annotated++
		}
	}
	fmt.Printf("  Query: %d events, %d carry annotations\n\n", len(rows), annotated)
	fmt.Printf("  %-30s %-8s %-20s %s\n", "TIME", "PID", "OP", "LABELS")
	fmt.Printf("  %s\n", "------------------------------------------------------------------------")
	for _, r := range rows {
		fmt.Printf("  %-30s %-8d %-20s %s\n",
			r.Event.Timestamp.Format(time.RFC3339Nano),
			r.Event.PID,
			r.Event.OpName(),
			formatLabels(r.Labels))
	}
	fmt.Println()
	return nil
}

// formatLabels renders a label map as a stable, sorted "k=v k=v" string.
func formatLabels(labels map[string]string) string {
	if len(labels) == 0 {
		return "-"
	}
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := ""
	for i, k := range keys {
		if i > 0 {
			out += " "
		}
		out += k + "=" + labels[k]
	}
	return out
}
