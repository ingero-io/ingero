package health

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/internal/sysinfo"
	"github.com/ingero-io/ingero/pkg/events"
)

// SQLiteCollectorConfig configures the SQLite-backed health signal collector.
// The collector reads aggregated CUDA events from a SQLite database written
// by a sibling `ingero trace --record` process.
type SQLiteCollectorConfig struct {
	// DBPath is the path to the SQLite DB populated by `ingero trace --record`.
	// Required.
	DBPath string
	// Window is the rolling window used to compute throughput and compute
	// signals. Defaults to 60s. When Window <= 60s, reads come from the
	// 5-second aggregate table (event_aggregates_5s), giving sub-minute
	// reactivity. Longer windows read the 1-minute table.
	Window time.Duration
	// NumGPUs is the number of GPUs on this host, used to normalize the
	// compute signal. 0 means autodetect (falls back to 1).
	NumGPUs int
	// Log receives info/warn messages; nil uses slog.Default().
	Log *slog.Logger
}

// subMinuteThreshold is the window cutoff at which the collector prefers
// the 5-second aggregate table over the 1-minute table. At exactly 60s both
// tables return equivalent numerators; the 5s table wins because its buckets
// are flushed every 5s, eliminating the up-to-60s latency of the 1m table.
const subMinuteThreshold = 60 * time.Second

// sqliteCollector reads the last Window seconds of event_aggregates from
// the trace SQLite DB and derives the four health signals plus a
// kernel-launch count for state-machine idle detection.
type sqliteCollector struct {
	store   *store.Store
	sys     *sysinfo.Collector
	gpuMem  *gpuMemReader
	window  time.Duration
	numGPUs int
	log     *slog.Logger

	// gpuMemOK tracks the state of the last gpu memory read so that
	// transitions (OK->err, err->OK) are logged exactly once per change.
	// Accessed only from Collect, which runs in a single goroutine.
	gpuMemOK bool
}

// NewSQLiteCollector constructs a Collector backed by a SQLite DB and
// live /proc/{stat,meminfo,loadavg} readings. The DB must already exist
// and be writable by a `trace --record` process; this opens it read-only.
func NewSQLiteCollector(cfg SQLiteCollectorConfig) (Collector, error) {
	if cfg.DBPath == "" {
		return nil, fmt.Errorf("SQLiteCollectorConfig.DBPath is required")
	}
	if cfg.Window <= 0 {
		cfg.Window = 60 * time.Second
	}
	if cfg.NumGPUs <= 0 {
		cfg.NumGPUs = 1
	}
	if cfg.Log == nil {
		cfg.Log = slog.Default()
	}
	s, err := store.NewReadOnly(cfg.DBPath)
	if err != nil {
		return nil, fmt.Errorf("opening signal db: %w", err)
	}
	gpu := newGPUMemReader(cfg.Log)
	if gpu.Available() {
		cfg.Log.Info("gpu memory: nvidia-smi available, using GPU memory for headroom signal")
	} else {
		cfg.Log.Info("gpu memory: nvidia-smi unavailable, falling back to host RAM proxy")
	}
	return &sqliteCollector{
		store:    s,
		sys:      sysinfo.New(),
		gpuMem:   gpu,
		window:   cfg.Window,
		numGPUs:  cfg.NumGPUs,
		log:      cfg.Log,
		gpuMemOK: gpu.Available(),
	}, nil
}

// Collect reads the last Window of CUDA aggregates from the DB plus a
// fresh sysinfo snapshot and derives the four signals:
//
//	Throughput = cudaLaunchKernel_count / window_seconds  (raw, kernels/sec)
//	Compute    = clamp(sum_dur_ns / (window_ns * numGPUs), 0, 1)
//	Memory     = (gpu_total - gpu_used) / gpu_total        (via nvidia-smi; host RAM proxy when unavailable)
//	CPU        = 1 - (CPUPercent / 100)
//
// Kernel-launch count is returned as the second value and feeds the
// state machine's idle detection.
func (c *sqliteCollector) Collect(ctx context.Context, now time.Time) (RawObservation, int, error) {
	// Rolling window. Windows <= 1 minute read the 5-second aggregate table
	// for sub-minute reactivity; longer windows use the 1-minute table which
	// retains deep history.
	q := store.QueryParams{
		From:   now.Add(-c.window),
		To:     now,
		Source: uint8(events.SourceCUDA),
	}
	var aggs []store.AggregateOpStats
	var err error
	if c.window <= subMinuteThreshold {
		aggs, err = c.store.QueryAggregatePerOp5s(q)
	} else {
		aggs, err = c.store.QueryAggregatePerOp(q)
	}
	if err != nil {
		return RawObservation{}, 0, fmt.Errorf("query aggregates: %w", err)
	}

	// Sum kernel launches (Throughput numerator) and total CUDA wall-time
	// (Compute numerator) over the window.
	var kernelLaunches, totalCount, totalSumDur int64
	for _, a := range aggs {
		totalCount += a.Count
		totalSumDur += a.SumDur
		if a.Op == uint8(events.CUDALaunchKernel) {
			kernelLaunches += a.Count
		}
	}

	windowSec := c.window.Seconds()
	windowNs := float64(c.window.Nanoseconds())

	var throughput, compute float64
	if windowSec > 0 {
		throughput = float64(kernelLaunches) / windowSec
	}
	if windowNs > 0 && c.numGPUs > 0 {
		compute = float64(totalSumDur) / (windowNs * float64(c.numGPUs))
		if compute > 1 {
			compute = 1
		}
		if compute < 0 {
			compute = 0
		}
	}

	// Live system stats for CPU + memory fallback.
	snap := c.sys.ReadOnce()

	// Prefer real GPU memory headroom; fall back to host RAM proxy when
	// nvidia-smi is unavailable or erroring. Log each state transition
	// exactly once to avoid flooding the log on every tick.
	memory := c.readMemorySignal(ctx, snap)

	cpu := 1 - (snap.CPUPercent / 100)
	if cpu < 0 {
		cpu = 0
	}
	if cpu > 1 {
		cpu = 1
	}

	obs := RawObservation{
		Throughput: throughput,
		Compute:    compute,
		Memory:     memory,
		CPU:        cpu,
	}

	// Clamp to safe int range. In practice kernel launches/sec rarely
	// exceeds a few hundred thousand, but totalCount may span multiple
	// minutes of aggregation so guard against overflow.
	launches := kernelLaunches
	if launches > int64(^uint32(0)>>1) {
		launches = int64(^uint32(0) >> 1)
	}
	_ = totalCount // kept for future per-op breakdown logging
	return obs, int(launches), nil
}

// readMemorySignal returns the memory headroom signal in [0, 1]. Uses
// GPU memory via nvidia-smi when available; falls back to host RAM.
// State transitions (available->err, err->available) are logged once.
func (c *sqliteCollector) readMemorySignal(ctx context.Context, snap sysinfo.SystemSnapshot) float64 {
	if c.gpuMem.Available() {
		used, total, err := c.gpuMem.Read(ctx)
		if err == nil && total > 0 {
			if !c.gpuMemOK {
				c.log.Info("gpu memory: reads recovered")
				c.gpuMemOK = true
			}
			m := float64(total-used) / float64(total)
			if m < 0 {
				m = 0
			}
			if m > 1 {
				m = 1
			}
			return m
		}
		if c.gpuMemOK {
			c.log.Warn("gpu memory: read failed, falling back to host RAM", "err", err)
			c.gpuMemOK = false
		}
	}
	if snap.MemTotalMB <= 0 {
		return 0
	}
	m := float64(snap.MemAvailMB) / float64(snap.MemTotalMB)
	if m < 0 {
		m = 0
	}
	if m > 1 {
		m = 1
	}
	return m
}

// Close releases the DB handle.
func (c *sqliteCollector) Close() error {
	return c.store.Close()
}
