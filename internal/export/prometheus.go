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
	"sort"
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

	// Per-direction CUDA memcpy aggregates (v0.14 item C, v0.15 item C).
	// bytes_total: cumulative counter. duration_ms: per-event histogram
	// (replaces v0.14 per-window-average gauge).
	if len(snap.MemcpyDirReadings) > 0 {
		// Sort by direction for deterministic exposition.
		rows := make([]stats.MemcpyDirStats, len(snap.MemcpyDirReadings))
		copy(rows, snap.MemcpyDirReadings)
		sort.SliceStable(rows, func(i, j int) bool { return rows[i].Direction < rows[j].Direction })

		b.WriteString("# HELP gpu_memcpy_bytes_total Cumulative CUDA memcpy bytes by direction\n")
		b.WriteString("# TYPE gpu_memcpy_bytes_total counter\n")
		for _, m := range rows {
			fmt.Fprintf(&b, "gpu_memcpy_bytes_total{direction=%q} %d\n", m.Direction, m.BytesTotal)
		}
		b.WriteString("# HELP gpu_memcpy_duration_ms Per-event CUDA memcpy duration by direction (histogram; replaces v0.14 gauge)\n")
		b.WriteString("# TYPE gpu_memcpy_duration_ms histogram\n")
		for _, m := range rows {
			h := m.DurationHistogram
			cum := uint64(0)
			for i, b1 := range h.ExplicitBounds {
				cum += h.BucketCounts[i]
				fmt.Fprintf(&b, "gpu_memcpy_duration_ms_bucket{direction=%q,le=\"%g\"} %d\n", m.Direction, b1, cum)
			}
			cum += h.BucketCounts[len(h.BucketCounts)-1]
			fmt.Fprintf(&b, "gpu_memcpy_duration_ms_bucket{direction=%q,le=\"+Inf\"} %d\n", m.Direction, cum)
			fmt.Fprintf(&b, "gpu_memcpy_duration_ms_sum{direction=%q} %g\n", m.Direction, h.Sum)
			fmt.Fprintf(&b, "gpu_memcpy_duration_ms_count{direction=%q} %d\n", m.Direction, h.Count)
		}
	}

	// NCCL collective running counters (v0.15 F2). The OTLP path
	// emits one per-event gauge per collective; the running counters
	// here are the pull-friendly view for vanilla Prometheus
	// scrapers. count + bytes_total are monotonic across the agent
	// process lifetime.
	if len(snap.NCCLCollectiveCounters) > 0 {
		// Sort rows by (op_type, kind) for deterministic exposition.
		// snapshotNCCLCollectiveCounters() returns rows in Go map
		// iteration order, which differs across scrapes. Downstream
		// tools that diff /metrics output need stable ordering.
		// v0.15 F2: sort once at emission.
		rows := make([]stats.NCCLCollectiveCounter, len(snap.NCCLCollectiveCounters))
		copy(rows, snap.NCCLCollectiveCounters)
		sort.SliceStable(rows, func(i, j int) bool {
			if rows[i].OpType != rows[j].OpType {
				return rows[i].OpType < rows[j].OpType
			}
			// Within same op_type, collective row (BarrierEvents==0) before barrier row.
			return rows[i].BarrierEvents < rows[j].BarrierEvents
		})

		// Emit count + bytes for non-barrier rows in a single pass per
		// metric so the help/type lines stay grouped per Prometheus
		// exposition convention.
		hasCount := false
		hasBytes := false
		hasBarrier := false
		for _, c := range rows {
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
			for _, c := range rows {
				if c.Count > 0 {
					fmt.Fprintf(&b, "gpu_nccl_collective_count{op_type=%q} %d\n", c.OpType, c.Count)
				}
			}
		}
		if hasBytes {
			b.WriteString("# HELP gpu_nccl_collective_bytes_total Cumulative bytes transferred by NCCL collective per op_type\n")
			b.WriteString("# TYPE gpu_nccl_collective_bytes_total counter\n")
			for _, c := range rows {
				if c.BytesTotal > 0 {
					fmt.Fprintf(&b, "gpu_nccl_collective_bytes_total{op_type=%q} %d\n", c.OpType, c.BytesTotal)
				}
			}
		}
		if hasBarrier {
			b.WriteString("# HELP gpu_nccl_collective_barrier_events Total NCCL barrier-wait events captured per op_type\n")
			b.WriteString("# TYPE gpu_nccl_collective_barrier_events counter\n")
			for _, c := range rows {
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

	// v0.15 item L: throttle event-edge counters. Cumulative across
	// process lifetime; one row per bucket. Always emitted (even
	// when zero) so consumers can detect "agent saw zero throttle
	// events" vs "agent isn't running this metric at all".
	te := snap.ThrottleEvents
	if te.PowerEvents != 0 || te.ThermalEvents != 0 || te.SWEvents != 0 || te.HWEvents != 0 {
		b.WriteString("# HELP gpu_throttle_power_event_total Cumulative throttle rising-edge events for power reasons\n")
		b.WriteString("# TYPE gpu_throttle_power_event_total counter\n")
		fmt.Fprintf(&b, "gpu_throttle_power_event_total %d\n", te.PowerEvents)
		b.WriteString("# HELP gpu_throttle_thermal_event_total Cumulative throttle rising-edge events for thermal reasons\n")
		b.WriteString("# TYPE gpu_throttle_thermal_event_total counter\n")
		fmt.Fprintf(&b, "gpu_throttle_thermal_event_total %d\n", te.ThermalEvents)
		b.WriteString("# HELP gpu_throttle_sw_event_total Cumulative throttle rising-edge events for software-imposed reasons\n")
		b.WriteString("# TYPE gpu_throttle_sw_event_total counter\n")
		fmt.Fprintf(&b, "gpu_throttle_sw_event_total %d\n", te.SWEvents)
		b.WriteString("# HELP gpu_throttle_hw_event_total Cumulative throttle rising-edge events for hardware reasons (umbrella)\n")
		b.WriteString("# TYPE gpu_throttle_hw_event_total counter\n")
		fmt.Fprintf(&b, "gpu_throttle_hw_event_total %d\n", te.HWEvents)
	}

	// v0.16.3 inference exporter surface. Mirrors the OTLP block:
	// per-workload histogram + mean + p95 gauges, engine-level
	// cumulative counters, and sampler observability. Empty when
	// --inference is not engaged.
	if len(snap.InferWorkloads) > 0 {
		// Determinism: sort by (cgroup, pid, stream, phase) so
		// successive scrapes produce stable line ordering.
		rows := make([]stats.InferWorkloadStats, len(snap.InferWorkloads))
		copy(rows, snap.InferWorkloads)
		sort.SliceStable(rows, func(i, j int) bool {
			if rows[i].CGroupHash != rows[j].CGroupHash {
				return rows[i].CGroupHash < rows[j].CGroupHash
			}
			if rows[i].PID != rows[j].PID {
				return rows[i].PID < rows[j].PID
			}
			if rows[i].StreamHandle != rows[j].StreamHandle {
				return rows[i].StreamHandle < rows[j].StreamHandle
			}
			return rows[i].Phase < rows[j].Phase
		})

		b.WriteString("# HELP ingero_infer_step_duration_ns Per-workload inference step duration distribution (cumulative)\n")
		b.WriteString("# TYPE ingero_infer_step_duration_ns histogram\n")
		for _, w := range rows {
			labels := inferWorkloadLabels(w)
			h := w.Histogram
			cum := uint64(0)
			for i, b1 := range h.ExplicitBounds {
				cum += h.BucketCounts[i]
				fmt.Fprintf(&b, "ingero_infer_step_duration_ns_bucket{%s,le=\"%g\"} %d\n", labels, b1, cum)
			}
			if len(h.BucketCounts) > 0 {
				cum += h.BucketCounts[len(h.BucketCounts)-1]
			}
			fmt.Fprintf(&b, "ingero_infer_step_duration_ns_bucket{%s,le=\"+Inf\"} %d\n", labels, cum)
			fmt.Fprintf(&b, "ingero_infer_step_duration_ns_sum{%s} %g\n", labels, h.Sum)
			fmt.Fprintf(&b, "ingero_infer_step_duration_ns_count{%s} %d\n", labels, h.Count)
		}

		b.WriteString("# HELP ingero_infer_baseline_mean_ns Per-workload inference step EMA mean\n")
		b.WriteString("# TYPE ingero_infer_baseline_mean_ns gauge\n")
		for _, w := range rows {
			fmt.Fprintf(&b, "ingero_infer_baseline_mean_ns{%s} %g\n", inferWorkloadLabels(w), w.MeanNs)
		}
		b.WriteString("# HELP ingero_infer_baseline_p95_ns Per-workload inference step P² p95 estimate\n")
		b.WriteString("# TYPE ingero_infer_baseline_p95_ns gauge\n")
		for _, w := range rows {
			fmt.Fprintf(&b, "ingero_infer_baseline_p95_ns{%s} %g\n", inferWorkloadLabels(w), w.P95Ns)
		}
	}
	if len(snap.InferWorkloads) > 0 || snap.InferStats.WorkloadsTracked > 0 ||
		len(snap.InferStats.OutliersTotal) > 0 || snap.InferSampler.DegradationsTotal > 0 {
		b.WriteString("# HELP ingero_infer_workloads_tracked Distinct (cgroup,pid,stream,phase) workloads currently tracked\n")
		b.WriteString("# TYPE ingero_infer_workloads_tracked gauge\n")
		fmt.Fprintf(&b, "ingero_infer_workloads_tracked %d\n", snap.InferStats.WorkloadsTracked)
	}
	if len(snap.InferStats.OutliersTotal) > 0 {
		buckets := make([]string, 0, len(snap.InferStats.OutliersTotal))
		for k := range snap.InferStats.OutliersTotal {
			buckets = append(buckets, k)
		}
		sort.Strings(buckets)
		b.WriteString("# HELP ingero_infer_outlier_total Cumulative inference step-duration outliers per bucket\n")
		b.WriteString("# TYPE ingero_infer_outlier_total counter\n")
		for _, bk := range buckets {
			fmt.Fprintf(&b, "ingero_infer_outlier_total{bucket=%q} %d\n", bk, snap.InferStats.OutliersTotal[bk])
		}
	}
	if len(snap.InferStats.ThrottleAtOutlier) > 0 {
		buckets := make([]string, 0, len(snap.InferStats.ThrottleAtOutlier))
		for k := range snap.InferStats.ThrottleAtOutlier {
			buckets = append(buckets, k)
		}
		sort.Strings(buckets)
		b.WriteString("# HELP ingero_infer_throttle_active_total Cumulative inference outliers observed while NVML throttle reasons were active\n")
		b.WriteString("# TYPE ingero_infer_throttle_active_total counter\n")
		for _, bk := range buckets {
			fmt.Fprintf(&b, "ingero_infer_throttle_active_total{bucket=%q} %d\n", bk, snap.InferStats.ThrottleAtOutlier[bk])
		}
	}
	// KV-cache alloc-age histogram. Engine-wide cumulative.
	if snap.InferStats.KVCacheAllocAgeHist.HasObservation {
		h := snap.InferStats.KVCacheAllocAgeHist
		b.WriteString("# HELP ingero_infer_kvcache_alloc_age_ms Live cudaMalloc allocation ages sampled at decode-phase outliers\n")
		b.WriteString("# TYPE ingero_infer_kvcache_alloc_age_ms histogram\n")
		cum := uint64(0)
		for i, bound := range h.ExplicitBounds {
			cum += h.BucketCounts[i]
			fmt.Fprintf(&b, "ingero_infer_kvcache_alloc_age_ms_bucket{le=\"%g\"} %d\n", bound, cum)
		}
		if len(h.BucketCounts) > 0 {
			cum += h.BucketCounts[len(h.BucketCounts)-1]
		}
		fmt.Fprintf(&b, "ingero_infer_kvcache_alloc_age_ms_bucket{le=\"+Inf\"} %d\n", cum)
		fmt.Fprintf(&b, "ingero_infer_kvcache_alloc_age_ms_sum %g\n", h.Sum)
		fmt.Fprintf(&b, "ingero_infer_kvcache_alloc_age_ms_count %d\n", h.Count)
	}
	if snap.InferSampler.DegradationsTotal > 0 || snap.InferSampler.Degraded {
		causeLabel := ""
		if snap.InferSampler.LastCause != "" {
			causeLabel = fmt.Sprintf(`{cause=%q}`, snap.InferSampler.LastCause)
		}
		b.WriteString("# HELP ingero_infer_sampler_degraded Inference sampler degraded state (1=admitting 100%)\n")
		b.WriteString("# TYPE ingero_infer_sampler_degraded gauge\n")
		degraded := 0
		if snap.InferSampler.Degraded {
			degraded = 1
		}
		fmt.Fprintf(&b, "ingero_infer_sampler_degraded%s %d\n", causeLabel, degraded)
		b.WriteString("# HELP ingero_infer_sampler_degradations_total Cumulative inference sampler flip-to-degraded transitions\n")
		b.WriteString("# TYPE ingero_infer_sampler_degradations_total counter\n")
		fmt.Fprintf(&b, "ingero_infer_sampler_degradations_total%s %d\n", causeLabel, snap.InferSampler.DegradationsTotal)
	}

	// v0.15 item K: per-cmd memfrag IOCTL event counters. cmd is
	// the raw nvidia_unlocked_ioctl cmd field; operators decode
	// against open-gpu-kernel-modules nv-ioctl-numbers.h /
	// nvos.h. Only emitted when the agent ran with
	// --enable-experimental-kprobes AND the host was on the
	// allowlist AND the BPF program saw >= 1 ioctl.
	if len(snap.MemfragIOCTLCounters) > 0 {
		b.WriteString("# HELP gpu_memfrag_ioctl_event_total Cumulative nvidia_unlocked_ioctl invocations by cmd code\n")
		b.WriteString("# TYPE gpu_memfrag_ioctl_event_total counter\n")
		for _, c := range snap.MemfragIOCTLCounters {
			fmt.Fprintf(&b, "gpu_memfrag_ioctl_event_total{cmd=\"0x%X\"} %d\n", c.Cmd, c.Count)
		}
	}

	// v0.15 item M: per-PID kernel-launch aggregates. count is
	// cumulative; threads_per_block / grid_blocks are histograms
	// fed by the cuLaunchKernel uprobe. Only emitted when the
	// agent ran with --enable-experimental-kprobes AND libcuda
	// was discovered.
	if len(snap.KernelLaunches) > 0 {
		b.WriteString("# HELP gpu_kernel_launch_count Cumulative cuLaunchKernel invocations\n")
		b.WriteString("# TYPE gpu_kernel_launch_count counter\n")
		for _, k := range snap.KernelLaunches {
			fmt.Fprintf(&b, "gpu_kernel_launch_count{pid=\"%d\"} %d\n", k.PID, k.Count)
		}
		b.WriteString("# HELP gpu_kernel_launch_threads_per_block Per-launch CUDA block thread count (BlockX*BlockY; BlockZ defaulted to 1)\n")
		b.WriteString("# TYPE gpu_kernel_launch_threads_per_block histogram\n")
		for _, k := range snap.KernelLaunches {
			h := k.ThreadsPerBlockHist
			cum := uint64(0)
			for i, b1 := range h.ExplicitBounds {
				cum += h.BucketCounts[i]
				fmt.Fprintf(&b, "gpu_kernel_launch_threads_per_block_bucket{pid=\"%d\",le=\"%g\"} %d\n", k.PID, b1, cum)
			}
			if len(h.BucketCounts) > 0 {
				cum += h.BucketCounts[len(h.BucketCounts)-1]
			}
			fmt.Fprintf(&b, "gpu_kernel_launch_threads_per_block_bucket{pid=\"%d\",le=\"+Inf\"} %d\n", k.PID, cum)
			fmt.Fprintf(&b, "gpu_kernel_launch_threads_per_block_sum{pid=\"%d\"} %g\n", k.PID, h.Sum)
			fmt.Fprintf(&b, "gpu_kernel_launch_threads_per_block_count{pid=\"%d\"} %d\n", k.PID, h.Count)
		}
		b.WriteString("# HELP gpu_kernel_launch_grid_blocks Per-launch grid block count (GridX*GridY*GridZ)\n")
		b.WriteString("# TYPE gpu_kernel_launch_grid_blocks histogram\n")
		for _, k := range snap.KernelLaunches {
			h := k.GridBlocksHist
			cum := uint64(0)
			for i, b1 := range h.ExplicitBounds {
				cum += h.BucketCounts[i]
				fmt.Fprintf(&b, "gpu_kernel_launch_grid_blocks_bucket{pid=\"%d\",le=\"%g\"} %d\n", k.PID, b1, cum)
			}
			if len(h.BucketCounts) > 0 {
				cum += h.BucketCounts[len(h.BucketCounts)-1]
			}
			fmt.Fprintf(&b, "gpu_kernel_launch_grid_blocks_bucket{pid=\"%d\",le=\"+Inf\"} %d\n", k.PID, cum)
			fmt.Fprintf(&b, "gpu_kernel_launch_grid_blocks_sum{pid=\"%d\"} %g\n", k.PID, h.Sum)
			fmt.Fprintf(&b, "gpu_kernel_launch_grid_blocks_count{pid=\"%d\"} %d\n", k.PID, h.Count)
		}
	}

	fmt.Fprint(w, b.String())
}

// inferWorkloadLabels builds the Prometheus label set for one
// per-workload inference data point. The base set is always
// (cgroup_path_hash, pid, stream_handle, phase); kernel_fingerprint,
// gen_ai_request_model, and gen_ai_system are appended only when the
// workload row carries non-empty values, so a default-mode agent
// (no fingerprint key, no inference scrape detection) keeps a tight
// label set that doesn't churn series cardinality.
func inferWorkloadLabels(w stats.InferWorkloadStats) string {
	labels := fmt.Sprintf(`cgroup_path_hash=%q,pid="%d",stream_handle="%d",phase=%q`,
		w.CGroupHash, w.PID, w.StreamHandle, w.Phase)
	if w.KernelFingerprint != 0 {
		labels = fmt.Sprintf(`%s,kernel_fingerprint="%016x"`, labels, w.KernelFingerprint)
	}
	if w.ModelName != "" {
		labels = fmt.Sprintf(`%s,gen_ai_request_model=%q`, labels, w.ModelName)
	}
	if w.EngineSystem != "" {
		labels = fmt.Sprintf(`%s,gen_ai_system=%q`, labels, w.EngineSystem)
	}
	return labels
}
