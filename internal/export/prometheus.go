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

	// Ring-buffer overflow total across every attached tracer. A
	// fast-climbing value means the kernel is producing events faster
	// than userspace drains them; operators should respond by raising
	// ring sizes or narrowing instrumentation.
	b.WriteString("# HELP ingero_ringbuf_overflows_total Cumulative eBPF ring-buffer / channel drops across all tracers.\n")
	b.WriteString("# TYPE ingero_ringbuf_overflows_total counter\n")
	fmt.Fprintf(&b, "ingero_ringbuf_overflows_total %d\n", snap.RingbufOverflows)

	// Trace DB size + prune counters. Exposed when the snapshot source
	// has a Store handle; absent otherwise (commands that don't open a
	// DB don't publish these). The two together answer "is prune
	// keeping up?" — db_bytes should stay under the operator's --max-db
	// while pruned_rows advances monotonically.
	if snap.TraceDB != nil {
		b.WriteString("# HELP ingero_trace_db_bytes Disk bytes used by the trace DB (main file + WAL + SHM).\n")
		b.WriteString("# TYPE ingero_trace_db_bytes gauge\n")
		fmt.Fprintf(&b, "ingero_trace_db_bytes %d\n", snap.TraceDB.DiskBytes)

		b.WriteString("# HELP ingero_trace_db_pruned_rows_total Cumulative rows deleted by size-based pruning across all tracked tables.\n")
		b.WriteString("# TYPE ingero_trace_db_pruned_rows_total counter\n")
		fmt.Fprintf(&b, "ingero_trace_db_pruned_rows_total %d\n", snap.TraceDB.PrunedRows)
	}

	// libnccl process discovery (v0.14 item A). Mirrors the OTLP emission
	// in otlp.go: one gauge=1 row per discovered NCCL-loaded process, plus
	// a per-node count. Empty slice (not nil) emits the total as 0 so
	// dashboards can plot "no NCCL workloads on this node".
	if snap.NCCLProcessReadings != nil {
		b.WriteString("# HELP gpu_nccl_process_loaded NCCL-loaded process discovered on this node (1=present)\n")
		b.WriteString("# TYPE gpu_nccl_process_loaded gauge\n")
		for _, r := range snap.NCCLProcessReadings {
			fmt.Fprintf(&b, "gpu_nccl_process_loaded{pid=\"%d\",comm=%q,libnccl_path=%q,libnccl_version=%q} 1\n",
				r.PID, r.Comm, r.LibPath, r.LibVersion)
		}
		b.WriteString("# HELP gpu_nccl_processes_total Count of NCCL-loaded processes on this node\n")
		b.WriteString("# TYPE gpu_nccl_processes_total gauge\n")
		fmt.Fprintf(&b, "gpu_nccl_processes_total %d\n", len(snap.NCCLProcessReadings))
	}

	// NVML-poll memory snapshot. Polling-based; the fragmentation
	// gauge is a coarse heuristic over (used, free, total).
	if len(snap.MemFragReadings) > 0 {
		b.WriteString("# HELP gpu_memory_used_bytes GPU memory currently allocated (NVML poll)\n")
		b.WriteString("# TYPE gpu_memory_used_bytes gauge\n")
		for _, r := range snap.MemFragReadings {
			fmt.Fprintf(&b, "gpu_memory_used_bytes{gpu_uuid=%q} %d\n", r.UUID, r.UsedBytes)
		}
		b.WriteString("# HELP gpu_memory_free_bytes GPU memory free (NVML poll)\n")
		b.WriteString("# TYPE gpu_memory_free_bytes gauge\n")
		for _, r := range snap.MemFragReadings {
			fmt.Fprintf(&b, "gpu_memory_free_bytes{gpu_uuid=%q} %d\n", r.UUID, r.FreeBytes)
		}
		b.WriteString("# HELP gpu_memory_total_bytes Total GPU memory (NVML poll)\n")
		b.WriteString("# TYPE gpu_memory_total_bytes gauge\n")
		for _, r := range snap.MemFragReadings {
			fmt.Fprintf(&b, "gpu_memory_total_bytes{gpu_uuid=%q} %d\n", r.UUID, r.TotalBytes)
		}
		b.WriteString("# HELP gpu_memory_fragmentation_estimate Coarse GPU memory fragmentation heuristic from NVML poll [0,1]\n")
		b.WriteString("# TYPE gpu_memory_fragmentation_estimate gauge\n")
		for _, r := range snap.MemFragReadings {
			fmt.Fprintf(&b, "gpu_memory_fragmentation_estimate{gpu_uuid=%q} %f\n", r.UUID, r.FragmentationEstimate)
		}
	}
	if len(snap.MemFragProcessReadings) > 0 {
		b.WriteString("# HELP gpu_memory_process_allocated_bytes Per-process GPU memory allocation (nvidia-smi compute-apps)\n")
		b.WriteString("# TYPE gpu_memory_process_allocated_bytes gauge\n")
		for _, p := range snap.MemFragProcessReadings {
			fmt.Fprintf(&b, "gpu_memory_process_allocated_bytes{gpu_uuid=%q,pid=\"%d\"} %d\n",
				p.UUID, p.PID, p.UsedBytes)
		}
	}

	// Per-direction CUDA memcpy aggregates (v0.14 item C). bytes_total is
	// cumulative; duration_ms is per-window average reset on each drain.
	if len(snap.MemcpyDirReadings) > 0 {
		b.WriteString("# HELP gpu_memcpy_bytes_total Cumulative CUDA memcpy bytes by direction\n")
		b.WriteString("# TYPE gpu_memcpy_bytes_total counter\n")
		for _, m := range snap.MemcpyDirReadings {
			fmt.Fprintf(&b, "gpu_memcpy_bytes_total{direction=%q} %d\n", m.Direction, m.BytesTotal)
		}
		b.WriteString("# HELP gpu_memcpy_duration_ms Average per-event CUDA memcpy duration by direction over the last export window\n")
		b.WriteString("# TYPE gpu_memcpy_duration_ms gauge\n")
		for _, m := range snap.MemcpyDirReadings {
			fmt.Fprintf(&b, "gpu_memcpy_duration_ms{direction=%q} %f\n", m.Direction, m.AverageDurationMs)
		}
	}

	// NCCL collective running counters (v0.15 F2). The OTLP path
	// emits one per-event gauge per collective; the running counters
	// here are the pull-friendly view for vanilla Prometheus
	// scrapers. count + bytes_total are monotonic across the agent
	// process lifetime.
	if len(snap.NCCLCollectiveCounters) > 0 {
		// Emit count + bytes for non-barrier rows in a single pass per
		// metric so the help/type lines stay grouped per Prometheus
		// exposition convention.
		hasCount := false
		hasBytes := false
		hasBarrier := false
		for _, c := range snap.NCCLCollectiveCounters {
			if c.Count > 0 {
				hasCount = true
			}
			if c.BytesTotal > 0 {
				hasBytes = true
			}
			if c.BarrierEvents > 0 {
				hasBarrier = true
			}
		}
		if hasCount {
			b.WriteString("# HELP gpu_nccl_collective_count Total NCCL collective events captured per op_type\n")
			b.WriteString("# TYPE gpu_nccl_collective_count counter\n")
			for _, c := range snap.NCCLCollectiveCounters {
				if c.Count > 0 {
					fmt.Fprintf(&b, "gpu_nccl_collective_count{op_type=%q} %d\n", c.OpType, c.Count)
				}
			}
		}
		if hasBytes {
			b.WriteString("# HELP gpu_nccl_collective_bytes_total Cumulative bytes transferred by NCCL collective per op_type\n")
			b.WriteString("# TYPE gpu_nccl_collective_bytes_total counter\n")
			for _, c := range snap.NCCLCollectiveCounters {
				if c.BytesTotal > 0 {
					fmt.Fprintf(&b, "gpu_nccl_collective_bytes_total{op_type=%q} %d\n", c.OpType, c.BytesTotal)
				}
			}
		}
		if hasBarrier {
			b.WriteString("# HELP gpu_nccl_collective_barrier_events Total NCCL barrier-wait events captured per op_type\n")
			b.WriteString("# TYPE gpu_nccl_collective_barrier_events counter\n")
			for _, c := range snap.NCCLCollectiveCounters {
				if c.BarrierEvents > 0 {
					fmt.Fprintf(&b, "gpu_nccl_collective_barrier_events{op_type=%q} %d\n", c.OpType, c.BarrierEvents)
				}
			}
		}
	}

	// NVML clock-throttle reasons (v0.12.10 W2-poller). Four gauges per
	// GPU (1=active, 0=inactive). Polling-based; events shorter than the
	// poll interval may be missed.
	if len(snap.ThrottleReadings) > 0 {
		for _, kind := range []struct {
			name string
			help string
		}{
			{"gpu_throttle_power_active", "GPU clock throttling for power reasons (1=active)"},
			{"gpu_throttle_thermal_active", "GPU clock throttling for thermal reasons (1=active)"},
			{"gpu_throttle_sw_active", "GPU clock throttling for software-imposed reasons (1=active)"},
			{"gpu_throttle_hw_active", "GPU clock throttling for hardware reasons, umbrella (1=active)"},
		} {
			fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s gauge\n", kind.name, kind.help, kind.name)
			for _, r := range snap.ThrottleReadings {
				var v int
				switch kind.name {
				case "gpu_throttle_power_active":
					if r.PowerActive {
						v = 1
					}
				case "gpu_throttle_thermal_active":
					if r.ThermalActive {
						v = 1
					}
				case "gpu_throttle_sw_active":
					if r.SWActive {
						v = 1
					}
				case "gpu_throttle_hw_active":
					if r.HWActive {
						v = 1
					}
				}
				fmt.Fprintf(&b, "%s{gpu_uuid=%q} %d\n", kind.name, r.UUID, v)
			}
		}
	}

	fmt.Fprint(w, b.String())
}
