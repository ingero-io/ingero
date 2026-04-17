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
	// signals. Defaults to 60s. The minute-granularity of event_aggregates
	// means values shorter than 60s produce the same numerator as 60s.
	Window time.Duration
	// NumGPUs is the number of GPUs on this host, used to normalize the
	// compute signal. 0 means autodetect (falls back to 1).
	NumGPUs int
	// Log receives info/warn messages; nil uses slog.Default().
	Log *slog.Logger
}

// sqliteCollector reads the last Window seconds of event_aggregates from
// the trace SQLite DB and derives the four health signals plus a
// kernel-launch count for state-machine idle detection.
type sqliteCollector struct {
	store   *store.Store
	sys     *sysinfo.Collector
	window  time.Duration
	numGPUs int
	log     *slog.Logger
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
	return &sqliteCollector{
		store:   s,
		sys:     sysinfo.New(),
		window:  cfg.Window,
		numGPUs: cfg.NumGPUs,
		log:     cfg.Log,
	}, nil
}

// Collect reads the last Window of CUDA aggregates from the DB plus a
// fresh sysinfo snapshot and derives the four signals:
//
//	Throughput = cudaLaunchKernel_count / window_seconds  (raw, kernels/sec)
//	Compute    = clamp(sum_dur_ns / (window_ns * numGPUs), 0, 1)
//	Memory     = MemAvailMB / MemTotalMB                   (host RAM proxy for GPU-mem headroom)
//	CPU        = 1 - (CPUPercent / 100)
//
// Kernel-launch count is returned as the second value and feeds the
// state machine's idle detection.
func (c *sqliteCollector) Collect(ctx context.Context, now time.Time) (RawObservation, int, error) {
	// Rolling window. event_aggregates buckets are minute-granular so we
	// query a small epsilon past the window to pick up the current bucket.
	q := store.QueryParams{
		From:   now.Add(-c.window),
		To:     now,
		Source: uint8(events.SourceCUDA),
	}
	aggs, err := c.store.QueryAggregatePerOp(q)
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

	// Live system stats for CPU/memory signals.
	snap := c.sys.ReadOnce()

	var memory float64
	if snap.MemTotalMB > 0 {
		memory = float64(snap.MemAvailMB) / float64(snap.MemTotalMB)
		if memory > 1 {
			memory = 1
		}
		if memory < 0 {
			memory = 0
		}
	}

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

// Close releases the DB handle.
func (c *sqliteCollector) Close() error {
	return c.store.Close()
}
