// Package export — prometheus.go provides a Prometheus /metrics HTTP endpoint.
//
// Serves the same metrics as OTLP in Prometheus exposition format.
// Disabled by default — enabled via --prometheus <addr> flag.
//
// Call chain: watch.go creates export.PrometheusServer →
//   HTTP server at /metrics → scrape handler renders current snapshot
package export

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"

	"github.com/ingero-io/ingero/internal/stats"
)

// PrometheusServer serves /metrics for Prometheus scraping.
type PrometheusServer struct {
	addr string
	mu   sync.RWMutex
	snap *stats.Snapshot
}

// NewPrometheus creates a Prometheus metrics endpoint. Returns nil if addr is empty.
func NewPrometheus(addr string) *PrometheusServer {
	if addr == "" {
		return nil
	}
	return &PrometheusServer{addr: addr}
}

// UpdateSnapshot replaces the current snapshot for the next scrape.
func (p *PrometheusServer) UpdateSnapshot(snap *stats.Snapshot) {
	if p == nil {
		return
	}
	p.mu.Lock()
	p.snap = snap
	p.mu.Unlock()
}

// Start begins serving /metrics. Blocks until ctx is cancelled.
func (p *PrometheusServer) Start(ctx context.Context) error {
	if p == nil {
		return nil
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/metrics", p.handleMetrics)

	listener, err := net.Listen("tcp", p.addr)
	if err != nil {
		return fmt.Errorf("prometheus listen %s: %w", p.addr, err)
	}

	srv := &http.Server{Handler: mux}

	go func() {
		<-ctx.Done()
		srv.Close()
	}()

	if err := srv.Serve(listener); err != http.ErrServerClosed {
		return err
	}
	return nil
}

func (p *PrometheusServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	p.mu.RLock()
	snap := p.snap
	p.mu.RUnlock()

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")

	if snap == nil {
		fmt.Fprintln(w, "# No data available yet")
		return
	}

	var b strings.Builder

	// System metrics (OTEL semantic conventions).
	if snap.System != nil {
		b.WriteString("# HELP system_cpu_utilization System CPU utilization ratio\n")
		b.WriteString("# TYPE system_cpu_utilization gauge\n")
		fmt.Fprintf(&b, "system_cpu_utilization %f\n", snap.System.CPUPercent/100)

		b.WriteString("# HELP system_memory_utilization System memory utilization ratio\n")
		b.WriteString("# TYPE system_memory_utilization gauge\n")
		fmt.Fprintf(&b, "system_memory_utilization %f\n", snap.System.MemUsedPct/100)

		b.WriteString("# HELP system_memory_usage_available Available memory in bytes\n")
		b.WriteString("# TYPE system_memory_usage_available gauge\n")
		fmt.Fprintf(&b, "system_memory_usage_available %d\n", snap.System.MemAvailMB*1024*1024)

		b.WriteString("# HELP system_cpu_load_average_1m 1-minute load average\n")
		b.WriteString("# TYPE system_cpu_load_average_1m gauge\n")
		fmt.Fprintf(&b, "system_cpu_load_average_1m %f\n", snap.System.LoadAvg1)
	}

	// Per-operation metrics.
	b.WriteString("# HELP gpu_cuda_operation_duration_microseconds CUDA operation latency percentiles\n")
	b.WriteString("# TYPE gpu_cuda_operation_duration_microseconds gauge\n")

	for _, op := range snap.Ops {
		source := op.Source.String()
		labels := fmt.Sprintf(`source="%s",operation="%s"`, source, op.Op)

		fmt.Fprintf(&b, "gpu_cuda_operation_duration_microseconds{%s,percentile=\"p50\"} %f\n", labels, float64(op.P50.Microseconds()))
		fmt.Fprintf(&b, "gpu_cuda_operation_duration_microseconds{%s,percentile=\"p95\"} %f\n", labels, float64(op.P95.Microseconds()))
		fmt.Fprintf(&b, "gpu_cuda_operation_duration_microseconds{%s,percentile=\"p99\"} %f\n", labels, float64(op.P99.Microseconds()))
	}

	b.WriteString("# HELP gpu_cuda_operation_count Total event count per operation\n")
	b.WriteString("# TYPE gpu_cuda_operation_count counter\n")
	for _, op := range snap.Ops {
		source := op.Source.String()
		fmt.Fprintf(&b, "gpu_cuda_operation_count{source=\"%s\",operation=\"%s\"} %d\n", source, op.Op, op.Count)
	}

	b.WriteString("# HELP ingero_anomaly_count Total anomaly events\n")
	b.WriteString("# TYPE ingero_anomaly_count counter\n")
	fmt.Fprintf(&b, "ingero_anomaly_count %d\n", snap.AnomalyEvents)

	fmt.Fprint(w, b.String())
}
