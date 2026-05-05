package cli

// W1 NVML-poll memfrag heuristic poller (v0.14 item D).
//
// Mirrors the v0.12.10 W2-poller pattern (throttle_poller.go):
// background goroutine polls nvidia-smi at a configurable interval;
// the snapshot callback drains a last-value-wins buffer per GPU and
// per (GPU, PID) pair into the snapshot for OTLP emission.
//
// CAVEAT (also in the OTLP metric description): polling-based
// heuristic, NOT the IOCTL-level memfrag tracking that v0.15 W1
// brings via a kprobe on nvidia_unlocked_ioctl. Burst-allocations
// shorter than the poll interval are missed by design.

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/ingero-io/ingero/internal/nvml"
	"github.com/ingero-io/ingero/internal/stats"
)

var (
	memFragBufMu sync.Mutex
	memFragBuf   map[string]stats.MemFragReading
	// per-(uuid,pid) last reading; keyed string "uuid|pid"
	memFragProcBuf map[string]stats.MemFragProcessReading
	memFragReady   bool
)

// addMemFragReading stores the latest per-GPU reading.
func addMemFragReading(r stats.MemFragReading) {
	memFragBufMu.Lock()
	defer memFragBufMu.Unlock()
	if memFragBuf == nil {
		memFragBuf = map[string]stats.MemFragReading{}
	}
	memFragBuf[r.UUID] = r
	memFragReady = true
}

// replaceMemFragProc replaces the per-process map atomically.
// Compute-apps comes back as a complete snapshot each tick (it is
// "every job currently running"), so a partial-merge would leak
// dead jobs forever. Replace.
func replaceMemFragProc(rs []stats.MemFragProcessReading) {
	memFragBufMu.Lock()
	defer memFragBufMu.Unlock()
	memFragProcBuf = make(map[string]stats.MemFragProcessReading, len(rs))
	for _, r := range rs {
		key := procKey(r.UUID, r.PID)
		memFragProcBuf[key] = r
	}
	memFragReady = true
}

func procKey(uuid string, pid uint32) string {
	// no allocation-free option without strconv import; this is the
	// hot path of one drain per snapshot, low frequency.
	return uuid + "|" + uintToStr(pid)
}

func uintToStr(v uint32) string {
	if v == 0 {
		return "0"
	}
	var buf [10]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

// drainMemFragBuf returns the latest per-GPU + per-process readings.
// Last-value-wins (gauge semantics); buffers are NOT cleared so
// snapshot ticks faster than the poll interval still see the last
// reading. Returns (nil, nil) before the first scan completes.
func drainMemFragBuf() ([]stats.MemFragReading, []stats.MemFragProcessReading) {
	memFragBufMu.Lock()
	defer memFragBufMu.Unlock()
	if !memFragReady {
		return nil, nil
	}
	gpus := make([]stats.MemFragReading, 0, len(memFragBuf))
	for _, r := range memFragBuf {
		gpus = append(gpus, r)
	}
	procs := make([]stats.MemFragProcessReading, 0, len(memFragProcBuf))
	for _, r := range memFragProcBuf {
		procs = append(procs, r)
	}
	return gpus, procs
}

// resetMemFragState clears module-level state. Test-only helper.
func resetMemFragState() {
	memFragBufMu.Lock()
	defer memFragBufMu.Unlock()
	memFragBuf = nil
	memFragProcBuf = nil
	memFragReady = false
}

// startMemFragPoller spawns the background polling goroutine.
// When either runner is nil (nvidia-smi missing), the poller logs
// a debug message and exits without spawning. interval <= 0 also
// disables.
func startMemFragPoller(ctx context.Context, interval time.Duration, memRun, appsRun nvml.Runner, log *slog.Logger) {
	if memRun == nil {
		if log != nil {
			log.Debug("memfrag poller: nvidia-smi not on PATH, NVML metrics disabled")
		}
		return
	}
	if interval <= 0 {
		return
	}
	if log == nil {
		log = slog.Default()
	}
	go func() {
		pollMemFragOnce(ctx, memRun, appsRun, log)
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				pollMemFragOnce(ctx, memRun, appsRun, log)
			}
		}
	}()
}

func pollMemFragOnce(ctx context.Context, memRun, appsRun nvml.Runner, log *slog.Logger) {
	memReadings, err := nvml.GetMemoryUsage(ctx, memRun)
	if err != nil {
		log.Debug("memfrag poller: memory query failed", "err", err)
		return
	}
	for _, r := range memReadings {
		addMemFragReading(stats.MemFragReading{
			UUID:                  r.UUID,
			UsedBytes:             r.UsedBytes,
			FreeBytes:             r.FreeBytes,
			TotalBytes:            r.TotalBytes,
			FragmentationEstimate: nvml.FragmentationEstimate(r.UsedBytes, r.FreeBytes, r.TotalBytes),
		})
	}
	// Compute-apps query is best-effort: a missing runner (rare -
	// would mean nvidia-smi vanished between memRun init and now) or
	// a parse failure should not kill the gpu-memory readings above.
	if appsRun != nil {
		apps, err := nvml.GetComputeApps(ctx, appsRun)
		if err != nil {
			log.Debug("memfrag poller: compute-apps query failed", "err", err)
			// Don't poison the per-process map on transient failure;
			// keep the previous snapshot until next success.
			return
		}
		converted := make([]stats.MemFragProcessReading, 0, len(apps))
		for _, a := range apps {
			converted = append(converted, stats.MemFragProcessReading{
				UUID:      a.UUID,
				PID:       a.PID,
				UsedBytes: a.UsedBytes,
			})
		}
		replaceMemFragProc(converted)
	}
}
