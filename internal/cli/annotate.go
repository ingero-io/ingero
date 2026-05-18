package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ingero-io/ingero/pkg/annotate"
	"github.com/ingero-io/ingero/pkg/contract"
)

var (
	annotateFromFile string
	annotateLabels   []string
	annotatePID      int
	annotateSocket   string
)

var annotateCmd = &cobra.Command{
	Use:   "annotate",
	Short: "Inject external annotations into the running trace",
	Long: `Send annotation lines to a running 'ingero trace' so a recorded trace
can later be sliced by external workload identity (step, epoch, task id).

The agent must be running 'ingero trace' with the annotation ingest
socket enabled. Annotations are written to the agent's SQLite store and
joined to events by 'ingero query' and 'ingero explain'.

Input is NDJSON (one JSON annotation object per line). Each object:

  {"labels":{"step":"42"},"pid":1234,"ts":1700000000000000000}

  labels      required, non-empty map of key/value strings
  ts          optional unix nanos; the agent stamps receive time if absent
  pid         optional process scope; the agent resolves the incarnation
  start_time  optional /proc start-time ticks (paired with pid)
  span_start  optional unix nanos (mark a phase, paired with span_end)
  span_end    optional unix nanos

Primary forms (no leak):
  echo '{"labels":{"step":"42"}}' | ingero annotate
  ingero annotate --from-file run-markers.ndjson

Convenience form (--label) - see the caveat below:
  ingero annotate --label step=42 --label epoch=3 --pid 4821

CAVEAT: --label and --pid values are visible to any process on the host
via /proc/<pid>/cmdline, shell history, and process audit logs. If an
annotation value carries sensitive workload identity (internal job
names, customer task ids), use stdin or --from-file instead - those do
not leak.`,

	RunE: annotateRunE,
}

func init() {
	annotateCmd.Flags().StringVar(&annotateFromFile, "from-file", "",
		"read NDJSON annotations from a file ('-' for stdin)")
	annotateCmd.Flags().StringArrayVar(&annotateLabels, "label", nil,
		"convenience key=value label (leaks via /proc/<pid>/cmdline; prefer stdin/--from-file)")
	annotateCmd.Flags().IntVar(&annotatePID, "pid", 0,
		"process scope for --label annotations (leaks via /proc/<pid>/cmdline)")
	annotateCmd.Flags().StringVar(&annotateSocket, "socket", "",
		"annotation ingest socket path (default: the agent-owned path)")

	rootCmd.AddCommand(annotateCmd)
}

// annotateRunE drives the one-shot annotate subcommand. It assembles
// annotation lines from one of three sources (--label flags, --from-file,
// or stdin), connects to the agent's ingest socket, and writes them as
// NDJSON. Exactly one source is used: --label takes precedence, then
// --from-file, then stdin.
func annotateRunE(cmd *cobra.Command, args []string) error {
	socket := annotateSocket
	if socket == "" {
		socket = contract.AnnotationSocketDir + "/" + contract.AnnotationSocketName
	}

	lines, err := annotateCollectLines()
	if err != nil {
		return err
	}
	if len(lines) == 0 {
		return fmt.Errorf("no annotations to send: pipe NDJSON on stdin, pass --from-file, or use --label")
	}

	conn, err := net.DialTimeout("unix", socket, 5*time.Second)
	if err != nil {
		return fmt.Errorf("connecting to annotation socket %s: %w\nHint: run 'ingero trace' with annotation ingest enabled first", socket, err)
	}
	defer conn.Close()

	sent := 0
	for _, line := range lines {
		if _, err := conn.Write([]byte(line + "\n")); err != nil {
			return fmt.Errorf("writing annotation to socket: %w", err)
		}
		sent++
	}
	debugf("annotate: sent %d annotation line(s) to %s", sent, socket)
	fmt.Printf("  Sent %d annotation(s) to %s\n", sent, socket)
	return nil
}

// annotateCollectLines returns the NDJSON lines to send. Source
// precedence: --label flags, then --from-file, then stdin. Each line is
// validated locally before sending so an obvious mistake is caught
// without a round-trip; the agent re-validates on ingest regardless.
func annotateCollectLines() ([]string, error) {
	// --label convenience form: build a single annotation object.
	if len(annotateLabels) > 0 {
		labels := map[string]string{}
		for _, kv := range annotateLabels {
			k, v, ok := strings.Cut(kv, "=")
			if !ok {
				return nil, fmt.Errorf("invalid --label %q: expected key=value", kv)
			}
			labels[k] = v
		}
		obj := map[string]any{contract.AnnotationFieldLabels: labels}
		if annotatePID > 0 {
			obj[contract.AnnotationFieldPID] = annotatePID
		}
		line, err := json.Marshal(obj)
		if err != nil {
			return nil, fmt.Errorf("encoding --label annotation: %w", err)
		}
		if err := validateAnnotationLine(line); err != nil {
			return nil, err
		}
		return []string{string(line)}, nil
	}

	// --from-file / stdin NDJSON form.
	var r io.Reader
	if annotateFromFile == "" || annotateFromFile == "-" {
		r = os.Stdin
	} else {
		f, err := os.Open(annotateFromFile)
		if err != nil {
			return nil, fmt.Errorf("opening --from-file %s: %w", annotateFromFile, err)
		}
		defer f.Close()
		r = f
	}

	var lines []string
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 4096), contract.AnnotationMaxLineBytes)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		raw := strings.TrimSpace(scanner.Text())
		if raw == "" {
			continue
		}
		if err := validateAnnotationLine([]byte(raw)); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		lines = append(lines, raw)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading annotations: %w", err)
	}
	return lines, nil
}

// validateAnnotationLine decodes one NDJSON line and runs the same
// contract validation the agent applies on ingest, so a malformed
// annotation is reported with a local line number instead of being
// silently rejected by the agent.
func validateAnnotationLine(line []byte) error {
	if len(line) > contract.AnnotationMaxLineBytes {
		return fmt.Errorf("annotation line is %d bytes, max %d",
			len(line), contract.AnnotationMaxLineBytes)
	}
	var w struct {
		Timestamp int64             `json:"ts"`
		Labels    map[string]string `json:"labels"`
		PID       uint32            `json:"pid"`
		StartTime uint64            `json:"start_time"`
		SpanStart int64             `json:"span_start"`
		SpanEnd   int64             `json:"span_end"`
	}
	if err := json.Unmarshal(line, &w); err != nil {
		return fmt.Errorf("not valid JSON: %w", err)
	}
	a := annotate.Annotation{
		TimestampNs: w.Timestamp,
		Labels:      w.Labels,
		Process:     annotate.ProcessIncarnation{PID: w.PID, StartTime: w.StartTime},
		SpanStartNs: w.SpanStart,
		SpanEndNs:   w.SpanEnd,
	}
	return a.Validate()
}
