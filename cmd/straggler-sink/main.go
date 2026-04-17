// Command straggler-sink is a reference consumer for the ingero remediation
// UDS event stream. It connects to the `/tmp/ingero-remediate.sock` socket
// (written by `ingero fleet-push` or `ingero trace --fleet-remediate`),
// decodes the NDJSON message stream, and exposes Prometheus metrics:
//
//	ingero_sink_events_total{type="..."}                       counter
//	ingero_sink_parse_errors_total                             counter
//	ingero_sink_connected                                       gauge (0|1)
//	ingero_sink_last_event_timestamp_seconds                    gauge
//	ingero_sink_active_stragglers{cluster_id,node_id}           gauge (0|1)
//
// The sink reconnects automatically when the producer restarts or the
// connection is dropped.
//
// Intended deployment: Helm sidecar on the ingero agent DaemonSet, sharing
// an emptyDir at /tmp so both containers see the same socket. See
// deploy/helm/ingero/templates/daemonset.yaml for the wire-up.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

func main() {
	var (
		socketPath     = flag.String("socket-path", "/tmp/ingero-remediate.sock", "path to the ingero remediation UDS")
		listenAddr     = flag.String("listen", ":9090", "HTTP listen address for /metrics")
		reconnectDelay = flag.Duration("reconnect-delay", 2*time.Second, "delay between reconnect attempts when the socket is unavailable")
	)
	flag.Parse()

	m := newMetrics()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Printf("straggler-sink: signal_received signal=%v", sig)
		cancel()
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/metrics", m.handleMetrics)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, "ok\n")
	})
	httpSrv := &http.Server{
		Addr:              *listenAddr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		log.Printf("straggler-sink: http_listening addr=%s", *listenAddr)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("straggler-sink: http_error err=%v", err)
			cancel()
		}
	}()

	runConsumer(ctx, *socketPath, *reconnectDelay, m)

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	_ = httpSrv.Shutdown(shutdownCtx)
	log.Printf("straggler-sink: stopped")
}

// runConsumer is the outer loop: connect, drain, sleep on failure, repeat
// until ctx is cancelled.
func runConsumer(ctx context.Context, socketPath string, reconnectDelay time.Duration, m *metrics) {
	for {
		if ctx.Err() != nil {
			return
		}
		conn, err := net.DialTimeout("unix", socketPath, 2*time.Second)
		if err != nil {
			m.setConnected(false)
			log.Printf("straggler-sink: dial_failed path=%s err=%v retry_in=%s", socketPath, err, reconnectDelay)
			select {
			case <-ctx.Done():
				return
			case <-time.After(reconnectDelay):
				continue
			}
		}
		m.setConnected(true)
		log.Printf("straggler-sink: connected path=%s", socketPath)

		// When ctx is cancelled, close the connection to unblock the reader.
		go func() {
			<-ctx.Done()
			conn.Close()
		}()

		drain(conn, m)

		m.setConnected(false)
		conn.Close()
		if ctx.Err() != nil {
			return
		}
		log.Printf("straggler-sink: disconnected, reconnecting in %s", reconnectDelay)
		select {
		case <-ctx.Done():
			return
		case <-time.After(reconnectDelay):
		}
	}
}

// drain reads NDJSON lines until EOF or error. Each line is a single JSON
// message discriminated by a "type" field.
func drain(conn net.Conn, m *metrics) {
	scanner := bufio.NewScanner(conn)
	// Max NDJSON line: a typedMessage with all fields is ~300 B. Give headroom
	// for future fields + any unanticipated large PreemptingPIDs list.
	scanner.Buffer(make([]byte, 0, 4096), 64*1024)

	for scanner.Scan() {
		handleLine(scanner.Bytes(), m)
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		log.Printf("straggler-sink: read_error err=%v", err)
	}
}

// eventHeader captures the shared fields that dispatch depends on. Each
// concrete type parses into its own struct if needed; the sink only cares
// about bumping counters and tracking straggler state, so we pull just the
// fields we need.
type eventHeader struct {
	Type      string `json:"type"`
	NodeID    string `json:"node_id"`
	ClusterID string `json:"cluster_id"`
}

func handleLine(line []byte, m *metrics) {
	if len(line) == 0 {
		return
	}
	var h eventHeader
	if err := json.Unmarshal(line, &h); err != nil {
		m.bumpParseError()
		log.Printf("straggler-sink: parse_error err=%v", err)
		return
	}
	if h.Type == "" {
		m.bumpParseError()
		return
	}
	m.bumpEvent(h.Type)
	m.markActivity()

	switch h.Type {
	case "straggler_state":
		m.setStraggler(h.ClusterID, h.NodeID, true)
	case "straggler_resolved":
		m.setStraggler(h.ClusterID, h.NodeID, false)
	}
}

// metrics holds the tiny set of gauges and counters the sink exports. All
// writers are goroutines from runConsumer; the /metrics handler is the
// reader. Sync via mutex for the label-keyed maps, atomics for scalars.
type metrics struct {
	mu                sync.Mutex
	eventsByType      map[string]uint64
	stragglerState    map[stragglerKey]uint8 // 1 = active, 0 = resolved
	parseErrors       uint64
	connected         uint32
	lastEventUnixNano int64
}

type stragglerKey struct{ ClusterID, NodeID string }

func newMetrics() *metrics {
	return &metrics{
		eventsByType:   map[string]uint64{},
		stragglerState: map[stragglerKey]uint8{},
	}
}

func (m *metrics) bumpEvent(t string) {
	m.mu.Lock()
	m.eventsByType[t]++
	m.mu.Unlock()
}

func (m *metrics) bumpParseError() {
	atomic.AddUint64(&m.parseErrors, 1)
}

func (m *metrics) setConnected(v bool) {
	var n uint32
	if v {
		n = 1
	}
	atomic.StoreUint32(&m.connected, n)
}

func (m *metrics) markActivity() {
	atomic.StoreInt64(&m.lastEventUnixNano, time.Now().UnixNano())
}

func (m *metrics) setStraggler(clusterID, nodeID string, active bool) {
	k := stragglerKey{ClusterID: clusterID, NodeID: nodeID}
	m.mu.Lock()
	if active {
		m.stragglerState[k] = 1
	} else {
		m.stragglerState[k] = 0
	}
	m.mu.Unlock()
}

// handleMetrics writes a minimal Prometheus text-exposition response. No
// dependency on client_golang so the sink stays tiny.
func (m *metrics) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")

	m.mu.Lock()
	events := make([]struct {
		t string
		n uint64
	}, 0, len(m.eventsByType))
	for k, v := range m.eventsByType {
		events = append(events, struct {
			t string
			n uint64
		}{k, v})
	}
	stragglers := make([]struct {
		k stragglerKey
		v uint8
	}, 0, len(m.stragglerState))
	for k, v := range m.stragglerState {
		stragglers = append(stragglers, struct {
			k stragglerKey
			v uint8
		}{k, v})
	}
	m.mu.Unlock()

	// Stable output order so scrape diffing / tests are deterministic.
	sort.Slice(events, func(i, j int) bool { return events[i].t < events[j].t })
	sort.Slice(stragglers, func(i, j int) bool {
		if stragglers[i].k.ClusterID != stragglers[j].k.ClusterID {
			return stragglers[i].k.ClusterID < stragglers[j].k.ClusterID
		}
		return stragglers[i].k.NodeID < stragglers[j].k.NodeID
	})

	fmt.Fprintln(w, "# HELP ingero_sink_events_total Total remediation events received by type.")
	fmt.Fprintln(w, "# TYPE ingero_sink_events_total counter")
	for _, e := range events {
		fmt.Fprintf(w, "ingero_sink_events_total{type=%q} %d\n", e.t, e.n)
	}

	fmt.Fprintln(w, "# HELP ingero_sink_parse_errors_total Lines that failed NDJSON parsing.")
	fmt.Fprintln(w, "# TYPE ingero_sink_parse_errors_total counter")
	fmt.Fprintf(w, "ingero_sink_parse_errors_total %d\n", atomic.LoadUint64(&m.parseErrors))

	fmt.Fprintln(w, "# HELP ingero_sink_connected 1 if connected to the UDS, 0 otherwise.")
	fmt.Fprintln(w, "# TYPE ingero_sink_connected gauge")
	fmt.Fprintf(w, "ingero_sink_connected %d\n", atomic.LoadUint32(&m.connected))

	fmt.Fprintln(w, "# HELP ingero_sink_last_event_timestamp_seconds Unix timestamp of the most recent event (0 if none).")
	fmt.Fprintln(w, "# TYPE ingero_sink_last_event_timestamp_seconds gauge")
	if last := atomic.LoadInt64(&m.lastEventUnixNano); last > 0 {
		fmt.Fprintf(w, "ingero_sink_last_event_timestamp_seconds %.3f\n", float64(last)/1e9)
	} else {
		fmt.Fprintln(w, "ingero_sink_last_event_timestamp_seconds 0")
	}

	fmt.Fprintln(w, "# HELP ingero_sink_active_stragglers 1 when a straggler_state has been received without a matching straggler_resolved, 0 otherwise.")
	fmt.Fprintln(w, "# TYPE ingero_sink_active_stragglers gauge")
	for _, s := range stragglers {
		fmt.Fprintf(w, "ingero_sink_active_stragglers{cluster_id=%q,node_id=%q} %d\n", s.k.ClusterID, s.k.NodeID, s.v)
	}
}
