package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"log/slog"
	"math/bits"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/cilium/ebpf"
	"github.com/spf13/cobra"

	"github.com/ingero-io/ingero/internal/cgroup"
	"github.com/ingero-io/ingero/internal/config"
	"github.com/ingero-io/ingero/internal/correlate"
	"github.com/ingero-io/ingero/internal/discover"
	"github.com/ingero-io/ingero/internal/health"
	"github.com/ingero-io/ingero/internal/infer"
	"github.com/ingero-io/ingero/internal/infer/enginedetect"
	"github.com/ingero-io/ingero/internal/infer/kvcache"
	"github.com/ingero-io/ingero/internal/infer/scrape"
	"github.com/ingero-io/ingero/internal/kprobe"
	"github.com/ingero-io/ingero/internal/sampling"
	"github.com/ingero-io/ingero/internal/ebpf/blockio"
	"github.com/ingero-io/ingero/internal/ebpf/cuda"
	"github.com/ingero-io/ingero/internal/ebpf/cudagraph"
	"github.com/ingero-io/ingero/internal/ebpf/driver"
	"github.com/ingero-io/ingero/internal/ebpf/host"
	"github.com/ingero-io/ingero/internal/ebpf/memfrag"
	"github.com/ingero-io/ingero/internal/ebpf/ncclprobe"
	nettracer "github.com/ingero-io/ingero/internal/ebpf/net"
	"github.com/ingero-io/ingero/internal/ebpf/pytrace"
	"github.com/ingero-io/ingero/internal/ebpf/tcp"
	"github.com/ingero-io/ingero/internal/export"
	"github.com/ingero-io/ingero/internal/filter"
	"github.com/ingero-io/ingero/internal/memtrack"
	"github.com/ingero-io/ingero/internal/nvml"
	"github.com/ingero-io/ingero/internal/procpath"
	"github.com/ingero-io/ingero/internal/remediate"
	"github.com/ingero-io/ingero/internal/straggler"
	"github.com/ingero-io/ingero/internal/k8s"
	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/internal/symtab"
	"github.com/ingero-io/ingero/internal/sysinfo"
	"github.com/ingero-io/ingero/internal/version"
	"github.com/ingero-io/ingero/pkg/events"
)

// Flag variables for the trace command.
var (
	tracePIDs         []int
	traceUser         string
	traceDuration     time.Duration
	traceJSON         bool
	traceVerbose      bool
	traceOTLP         string // OTLP endpoint (e.g., "localhost:4317"). Empty = disabled.
	traceOTLPInsecure bool   // Send OTLP over plaintext HTTP. Default false (TLS).
	traceOTLPAllowNonLoopback bool // Permit --otlp-insecure with a non-loopback endpoint.
	traceProm         string // Prometheus listen addr (e.g., ":9090"). Empty = disabled.
	traceRecord       bool   // Record events to SQLite (default true).
	traceRecordAll    bool   // Store every event (disable selective storage).
	traceStack        bool   // Capture userspace stack traces per event.
	traceDBPath       string // Custom DB path (default: ~/.ingero/ingero.db).
	traceMaxDB        string // Max DB size (e.g., "10g"). 0 = unlimited.
	traceStackSamples int    // Max events stored per unique call stack (0 = unlimited).
	traceLogPath      string        // Log output file path (debug, no rotation).
	traceDeadbandPct  float64       // Deadband threshold % (0 = disabled).
	traceHeartbeat    time.Duration // Heartbeat interval (0 = no heartbeat).
	traceNoIO         bool          // Disable block I/O tracing.
	traceNoTCP        bool          // Disable TCP retransmit tracing.
	traceNoNet        bool          // Disable network socket tracing.
	traceNCCL         bool          // Opt-in: attach NCCL collective uprobes (v0.12.0).
	traceNCCLLib      string        // Explicit libnccl.so path (skip discovery).

	// v0.15 Tier 2: experimental closed-driver kprobes (W1 memfrag,
	// W2 throttle, kernel grid/block dims). Default off; when set,
	// the agent additionally checks the running driver + kernel
	// pair against an allowlist and only loads probes on tested
	// configurations.
	traceEnableExperimentalKprobes bool

	// ncclBufMu guards ncclBuf, the per-snapshot drain buffer for NCCL
	// data points. Producer: NCCL ringbuf goroutine. Consumer: onSnapshot
	// callback. Capped at ncclBufMax to bound memory if the consumer
	// stalls; overflow events are dropped silently and surfaced via the
	// "NCCL tracing: ... dropped" stderr line at shutdown.
	ncclBufMu  sync.Mutex
	ncclBuf    []stats.NCCLDataPoint
	ncclBufMax = 4096
	traceRemediate    bool          // Enable VRAM tracking and UDS remediation endpoint.
	traceRemediateGid int           // Numeric GID for remediation socket group access. Mirrors fleet-push.
	traceNode         string        // Node identity for multi-node correlation.
	traceCluster      string        // Cluster identity (operator-supplied, shared across pods of one job).
	traceCUDALib      string        // Explicit libcudart.so path (skip discovery).
	traceRingBufSize  string        // Ring buffer size override (e.g., "32m", "8m").
	traceSamplingRate uint32        // Fixed sampling rate (0 = adaptive).
	tracePyWalker     string        // Python frame walker: auto, ebpf, userspace.
	traceWorkloadType string        // Mirrors fleet.workload_type from configs/ingero.yaml; gates correlator window.
	traceThrottlePoll time.Duration // NVML clock-throttle reason poll interval (default 5s; 0 = disabled).

	// v0.14 item A: libnccl process-discovery scanner interval.
	// Default 10s; 0 disables. Scanner is independent of --nccl;
	// useful by itself for "which nodes/processes have NCCL loaded".
	traceLibNCCLDiscoveryInterval time.Duration

	// v0.14 item D: NVML-poll memfrag heuristic interval. Default 10s;
	// 0 disables. Polling-based; v0.15 W1 will replace with IOCTL-level
	// tracing.
	traceMemFragPollInterval time.Duration

	// v0.16 inference-umbrella flags. The umbrella (`--inference`) is
	// a meta-flag that flips defaults on a coordinated set: workload
	// type, sampler attachment, output mode, DB rollover, and the
	// per-workload step-duration baseliner. Operators can still
	// override any individual flag explicitly.
	traceInference                  bool
	traceDBRolloverSize             string
	traceDBRolloverKeep             int
	traceInferenceWarmup            int
	traceInferenceOutlierRatio      float64
	traceInferencePauseSeverity     string
	traceInferenceSamplerDegradeOn  string

	// v0.16.1 phase-classifier flags. Phase-aware baselines
	// eliminate false negatives on heterogeneous-task streams
	// (vLLM continuous batching, single-stream PyTorch with mixed
	// prompt sizes). Default-on when --inference is engaged.
	traceInferencePhaseClassifier      string // "rule" | "off"
	traceInferencePhaseDecodeMaxLaunch int
	traceInferencePhaseDecodeMaxMemcpy string // human size
	traceInferencePhasePrefillMinLaunch int
	traceInferencePhasePrefillMinAvgKern time.Duration
	traceInferencePhaseMixedMemcpy       string // human size
	traceInferencePhaseMixedLaunchLow    int
	traceInferencePhaseMixedLaunchHigh   int
	traceInferencePhaseMemfragDecodeMin  int

	// v0.16.5b: kernel-fingerprint workload key.
	traceInferenceFingerprintKey         bool

	// KV-cache lineage tracking knobs.
	traceInferenceKVCacheLineage  bool
	traceInferenceKVCacheTopN     int
	traceInferenceKVCacheMaxPerPID int

	// inferEngine is the per-workload step-duration baseline +
	// classifier. Constructed only when --inference is set.
	inferEngine *infer.Engine
	// inferSampler is the store-side sampler attached to admissions.
	// Captured here so the infer engine can degrade it on outlier.
	inferSampler *sampling.Sampler
	// inferCgroupCache resolves event PIDs to cgroup_path_hash so the
	// per-workload key on the infer engine matches the emitter's
	// emission attribute. Constructed only when --inference is set.
	inferCgroupCache *health.PIDCGroupHashCache

	// v0.16.2 engine /metrics scrape flags.
	traceInferenceScrape         string // auto | off
	traceInferenceScrapeInterval time.Duration
	traceInferenceScrapeHost     string // override of localhost for non-typical pod topologies

	// v0.16.4 #10: continuous engine re-detection cadence. The
	// scraper polls this interval (when no engine known) to walk
	// /proc and pick up engines that started after the agent;
	// once at least one engine is registered the cadence drops to
	// the internal slow constant (5m) automatically.
	traceInferenceScrapeRedetectInterval time.Duration

	// inferScraper is the periodic engine /metrics scraper. Active
	// only when --inference is set AND scrape mode is "auto" AND
	// at least one engine PID is detected on the host.
	inferScraper *scrape.Scraper
)

// ncclBufferAdd appends a data point to the snapshot drain buffer. Drops
// silently when the buffer is full (snapshot consumer is too slow OR
// no exporter is configured to drain it).
//
// v0.15 F2: also tally the running counter so the Prometheus exporter
// has something to emit. Buffer-full does NOT skip the counter
// update; the counter survives back-pressure (we still want the
// total to be honest even if per-event records get dropped).
func ncclBufferAdd(p stats.NCCLDataPoint) {
	recordNCCLCollective(p)
	ncclBufMu.Lock()
	defer ncclBufMu.Unlock()
	if len(ncclBuf) >= ncclBufMax {
		return
	}
	ncclBuf = append(ncclBuf, p)
}

// ncclBufferDrain returns and clears the buffer. Called from onSnapshot.
func ncclBufferDrain() []stats.NCCLDataPoint {
	ncclBufMu.Lock()
	defer ncclBufMu.Unlock()
	if len(ncclBuf) == 0 {
		return nil
	}
	out := ncclBuf
	ncclBuf = nil
	return out
}

var traceCmd = &cobra.Command{
	Use:   "trace",
	Short: "Live CUDA event stream with stats and anomaly detection",
	Long: `Attach eBPF uprobes to CUDA runtime (libcudart.so) and stream events
in real time. Shows per-operation latency percentiles (p50/p95/p99),
time-fraction breakdown (% of wall clock per operation), and flags
periodic spikes and statistical anomalies.

Auto-detects running CUDA processes. Use --pid to target a specific process.

Requires root privileges (sudo) for eBPF probe attachment.`,

	RunE: traceRunE,
}

func init() {
	traceCmd.Flags().IntSliceVarP(&tracePIDs, "pid", "p", nil, "target process ID(s), comma-separated (default: all CUDA processes for current user)")
	traceCmd.Flags().StringVarP(&traceUser, "user", "u", "", "trace all CUDA processes owned by this user")
	traceCmd.Flags().DurationVarP(&traceDuration, "duration", "d", 0, "stop after duration (e.g., 30s, 5m). 0 = run until Ctrl+C")
	traceCmd.Flags().BoolVar(&traceJSON, "json", false, "output events as JSON lines (for piping to jq, scripts, MCP)")
	traceCmd.Flags().BoolVarP(&traceVerbose, "verbose", "v", false, "show verbose table output (extra columns in TUI mode)")
	traceCmd.Flags().StringVar(&traceOTLP, "otlp", "", "OTLP endpoint for metric export (HTTP JSON, POST /v1/metrics; default port 4318, e.g., localhost:4318). Disabled by default. Compatible with OpenTelemetry Collector, Grafana Alloy/Cloud, Datadog Agent, New Relic, and any OTLP-HTTP receiver. Metrics: gpu.cuda.operation.{duration,count}, system.{cpu,memory}.utilization, ingero.anomaly.count. Defaults to HTTPS; pass --otlp-insecure for plaintext (loopback only unless --otlp-insecure-allow-non-loopback). See docs/otlp.md.")
	traceCmd.Flags().BoolVar(&traceOTLPInsecure, "otlp-insecure", false, "Send OTLP metrics + spans over plaintext HTTP instead of HTTPS. Refused for non-loopback endpoints unless --otlp-insecure-allow-non-loopback is also passed; the payload includes per-PID workload fingerprints (model names, kernel-launch sequences, cgroup hashes) that an in-path observer can scrape.")
	traceCmd.Flags().BoolVar(&traceOTLPAllowNonLoopback, "otlp-insecure-allow-non-loopback", false, "Override the non-loopback refusal that pairs with --otlp-insecure. Required when intentionally exporting plaintext OTLP to a remote collector (e.g. a sidecar pod on a service-mesh-mTLS network).")
	traceCmd.Flags().StringVar(&traceProm, "prometheus", "", "Prometheus /metrics listen address (e.g., :9090). Disabled by default. Same metric names as --otlp. The listener has no authentication; non-loopback binds log a startup warning. See docs/otlp.md.")
	traceCmd.Flags().BoolVar(&traceRecord, "record", true, "record events to SQLite (default true, use --record=false to disable)")
	traceCmd.Flags().BoolVar(&traceRecordAll, "record-all", false, "store every event individually (disables selective storage, larger DB)")
	traceCmd.Flags().BoolVar(&traceStack, "stack", true, "capture userspace stack traces (0.4-0.6% overhead, use --stack=false to disable)")
	traceCmd.Flags().StringVar(&traceDBPath, "db", "", "database path (default: ~/.ingero/ingero.db)")
	traceCmd.Flags().StringVar(&traceMaxDB, "max-db", "10g", "max database size (e.g., 10g, 500m, 1t). 0 = unlimited")
	traceCmd.Flags().IntVar(&traceStackSamples, "stack-samples", 100, "max events stored per unique call stack (0 = unlimited)")
	traceCmd.Flags().StringVar(&traceLogPath, "log", "", "write log output to file (append, no rotation)")
	traceCmd.Flags().Float64Var(&traceDeadbandPct, "deadband", 0, "suppress snapshot writes when all metrics change < this % (0 = disabled)")
	traceCmd.Flags().DurationVar(&traceHeartbeat, "heartbeat", 0, "force a snapshot write at least this often even if within deadband (0 = no heartbeat)")
	traceCmd.Flags().BoolVar(&traceNoIO, "no-io", false, "disable block I/O tracing")
	traceCmd.Flags().BoolVar(&traceNoTCP, "no-tcp", false, "disable TCP retransmit tracing")
	traceCmd.Flags().BoolVar(&traceNoNet, "no-net", false, "disable network socket tracing")
	traceCmd.Flags().BoolVar(&traceNCCL, "nccl", false, "experimental: attach NCCL collective uprobes (v0.12.0; opt-in until v0.13)")
	traceCmd.Flags().StringVar(&traceNCCLLib, "nccl-lib", "", "explicit libnccl.so path (skip /proc/<pid>/maps discovery)")
	traceCmd.Flags().BoolVar(&traceRemediate, "remediate", false, "enable VRAM tracking and UDS remediation endpoint (requires an external consumer; see docs/remediation-protocol_fleet.md)")
	traceCmd.Flags().IntVar(&traceRemediateGid, "remediate-gid", 65532,
		"Numeric GID granted group access to the remediation socket (chown -1:gid + chmod 0770). Default 65532 matches distroless 'nonroot'. Set < 0 to keep the socket owner-only (0700).")
	traceCmd.Flags().StringVar(&traceNode, "node", "", "node identity for multi-node correlation (default: os.Hostname())")
	traceCmd.Flags().StringVar(&traceCluster, "cluster", "", "cluster identity shared across pods of one training/serving job. Emitted on every OTLP push as the ingero.cluster.id resource attribute so the Fleet processor can group cross-pod metrics by (cluster, model, phase) for peer-relative outlier detection. Empty leaves the attribute off the wire.")
	traceCmd.Flags().StringVar(&traceCUDALib, "cuda-lib", "", "explicit path to libcudart.so (skip auto-discovery)")
	traceCmd.Flags().StringVar(&traceRingBufSize, "ringbuf-size", "", "override ring buffer size for high-throughput probes (cuda, driver, host). Low-throughput probes (tcp, net, blockio, graph) keep their compiled defaults. Must be power of 2, min 4096.")
	traceCmd.Flags().Uint32Var(&traceSamplingRate, "sampling-rate", 0, "event sampling rate (0 = adaptive, 1 = emit all, N > 1 = emit 1 in N). Adaptive mode auto-increases rate under sustained drop pressure.")
	traceCmd.Flags().StringVar(&tracePyWalker, "py-walker", "auto",
		"Python frame walker: auto (userspace, default), ebpf (kernel-side, requires Python 3.12), userspace (force userspace)")
	traceCmd.Flags().StringVar(&traceWorkloadType, "fleet-workload-type", "unknown",
		"Workload type from fleet.workload_type (training | inference | unknown). Selects correlator window: training/unknown=10s, inference=500ms.")
	traceCmd.Flags().DurationVar(&traceThrottlePoll, "throttle-poll-interval", 5*time.Second,
		"interval between NVML clock-throttle reason polls (gpu.throttle.*_active metrics). 0 = disable. Floor: bursts shorter than this are missed by design.")
	traceCmd.Flags().DurationVar(&traceLibNCCLDiscoveryInterval, "libnccl-discovery-interval", 10*time.Second,
		"interval between libnccl process-discovery scans (gpu.nccl.process_loaded, gpu.nccl.processes_total metrics). 0 = disable. Independent of --nccl.")
	traceCmd.Flags().DurationVar(&traceMemFragPollInterval, "memfrag-poll-interval", 10*time.Second,
		"interval between NVML memory polls for the memfrag heuristic (gpu.memory.{used,free,total,fragmentation_estimate,process.allocated_bytes}). 0 = disable. Polling-based; v0.15 ships an event-driven IOCTL kprobe behind --enable-experimental-kprobes.")
	traceCmd.Flags().BoolVar(&traceEnableExperimentalKprobes, "enable-experimental-kprobes", false,
		"EXPERIMENTAL: load v0.15 closed-driver kprobes (memfrag IOCTL, throttle, kernel grid/block dims). Probes only attach when the running NVIDIA driver + Linux kernel pair is on a tested allowlist (DefaultAllowlist in internal/kprobe). Outside the allowlist: warning at startup, no probe load.")

	// v0.16 inference-umbrella flags. --inference is a meta-flag that
	// engages a coordinated set of defaults for production daemon use:
	// workload_type=inference (sub-second causal window), JSON-only
	// output, sampler attached to the SQLite store, --remediate=true
	// (UDS socket exposed), DB rollover instead of in-place pruning,
	// and per-workload step-duration outlier detection. Operators can
	// still override any individual flag explicitly.
	traceCmd.Flags().BoolVar(&traceInference, "inference", false,
		"v0.16 umbrella: production daemon for inference. Sets workload_type=inference, attaches the event sampler, switches to JSON output, enables --remediate, swaps --max-db for DB rollover, and turns on per-workload step-duration outlier detection. Individual flags still override.")
	traceCmd.Flags().StringVar(&traceDBRolloverSize, "db-rollover-size", "",
		"rotate the SQLite trace DB when its size crosses this value (e.g. 1g, 500m). Empty = disabled. Mutually exclusive with --max-db. Default 1g when --inference is set.")
	traceCmd.Flags().IntVar(&traceDBRolloverKeep, "db-rollover-keep", 6,
		"number of rolled-over DB files to retain on disk (oldest deleted first).")
	traceCmd.Flags().IntVar(&traceInferenceWarmup, "inference-warmup", 30,
		"healthy steps required before per-workload outlier classification activates.")
	traceCmd.Flags().Float64Var(&traceInferenceOutlierRatio, "inference-outlier-ratio", 3.0,
		"step duration must exceed baseline p95 by this multiplier to land in the largest outlier bucket. Smaller buckets fire at 1.5x and 2x of p95.")
	traceCmd.Flags().StringVar(&traceInferencePauseSeverity, "inference-pause-on-severity", "HIGH",
		"pause baseline updates while a causal chain at this severity or higher is active for the PID (HIGH | MEDIUM | LOW | empty to disable).")
	traceCmd.Flags().StringVar(&traceInferenceSamplerDegradeOn, "inference-sampler-degrade-on", "3x",
		"smallest outlier bucket that bumps the store sampler to admit 100%% of events (1.5x | 2x | 3x | off).")

	// v0.16.1 phase classifier flags. Phase-aware baselines split the
	// per-(cgroup, pid, stream) baseline by phase (prefill / decode /
	// mixed / unknown), so a 10x slow decode is compared against the
	// decode baseline (not the mixed-bucket p95 absorbed by prefill).
	traceCmd.Flags().StringVar(&traceInferencePhaseClassifier, "inference-phase-classifier", "rule",
		"per-step phase classifier mode: rule (default) | off. When off, all steps land in a single per-(cgroup,pid,stream) baseline, restoring v0.16.0 behavior.")
	traceCmd.Flags().IntVar(&traceInferencePhaseDecodeMaxLaunch, "inference-phase-decode-max-launches", 50,
		"a step is decode if launches < this AND memcpy < threshold AND no NCCL. Tighten for embedding/vision workloads.")
	traceCmd.Flags().StringVar(&traceInferencePhaseDecodeMaxMemcpy, "inference-phase-decode-max-memcpy", "1m",
		"max memcpy bytes for a decode step (e.g., 1m, 512k). Above this, the step lands in mixed.")
	traceCmd.Flags().IntVar(&traceInferencePhasePrefillMinLaunch, "inference-phase-prefill-min-launches", 200,
		"a step is prefill if launches > this OR avg-kernel > prefill-min-avg-kernel.")
	traceCmd.Flags().DurationVar(&traceInferencePhasePrefillMinAvgKern, "inference-phase-prefill-min-avg-kernel", 500*time.Microsecond,
		"a step is prefill if launches > prefill-min-launches OR avg kernel duration > this.")
	traceCmd.Flags().StringVar(&traceInferencePhaseMixedMemcpy, "inference-phase-mixed-memcpy", "10m",
		"a step is mixed if memcpy >= this (e.g., 10m, 50m).")
	traceCmd.Flags().IntVar(&traceInferencePhaseMixedLaunchLow, "inference-phase-mixed-launch-low", 50,
		"low end of the mixed-phase launch range (inclusive).")
	traceCmd.Flags().IntVar(&traceInferencePhaseMixedLaunchHigh, "inference-phase-mixed-launch-high", 200,
		"high end of the mixed-phase launch range (inclusive).")
	traceCmd.Flags().IntVar(&traceInferencePhaseMemfragDecodeMin, "inference-phase-memfrag-decode-min", 1,
		"minimum NVIDIA memfrag IOCTL events between syncs to classify a low-launch step as decode (KV-cache pressure rule). Set to a large value to effectively disable.")
	traceCmd.Flags().BoolVar(&traceInferenceFingerprintKey, "inference-fingerprint-key", false,
		"v0.16.5b: include a per-step kernel-launch-sequence fingerprint in the inference WorkloadKey. Engage when a single (pid, stream) hosts multiple models / model versions and you want independent baselines per kernel sequence. Off by default to keep the LRU footprint at v0.16.4 levels.")
	traceCmd.Flags().BoolVar(&traceInferenceKVCacheLineage, "inference-kvcache-lineage", false,
		"track per-PID cudaMalloc / cudaFree lineage so decode-phase outliers carry top-N alloc ages (KV-cache fragmentation context). Adds the ingero.infer.kvcache.alloc_age_ms histogram and a kv_cache_top_alloc_ages_ms field on the UDS outlier envelope. Off by default - bounded memory cost (~64 KB per inference PID).")
	traceCmd.Flags().IntVar(&traceInferenceKVCacheTopN, "inference-kvcache-top-n", 5,
		"how many oldest live allocations to attach to a decode-phase outlier event. Caps the UDS / OTLP attribute set under fragmentation; 5 is enough to identify the stale-cache pattern without inflating the wire envelope.")
	traceCmd.Flags().IntVar(&traceInferenceKVCacheMaxPerPID, "inference-kvcache-max-per-pid", 0,
		"per-PID live-allocation tracking cap. Older entries LRU-evict on insert. 0 uses the kvcache package default (8192).")

	// v0.16.2 engine /metrics scrape flags. When --inference is set
	// and an engine (vLLM/TGI/SGLang/Triton) is detected on the host,
	// the agent periodically pulls /metrics and translates engine-
	// specific names to OTel GenAI semantic conventions
	// (gen_ai.client.operation.time_to_first_token, etc). Downstream
	// dashboards that already speak OTel GenAI light up automatically.
	traceCmd.Flags().StringVar(&traceInferenceScrape, "inference-scrape", "auto",
		"engine /metrics scraping: auto (default; auto-detect vLLM/TGI/SGLang/Triton from cmdline) | off.")
	traceCmd.Flags().DurationVar(&traceInferenceScrapeInterval, "inference-scrape-interval", 10*time.Second,
		"how often to scrape an engine's /metrics endpoint.")
	traceCmd.Flags().StringVar(&traceInferenceScrapeHost, "inference-scrape-host", "127.0.0.1",
		"host to scrape engine /metrics from. Defaults to 127.0.0.1 (engine in same pod). Set to a remote host only for sidecar-on-different-host topologies.")
	traceCmd.Flags().DurationVar(&traceInferenceScrapeRedetectInterval, "inference-scrape-redetect-interval", 30*time.Second,
		"how often the scraper re-walks /proc to discover newly-booted engines and re-confirm registered targets. Drops to a slower internal cadence (5m) once at least one engine is found.")

	rootCmd.AddCommand(traceCmd)
}

// validatePyWalker returns nil if s is one of the supported --py-walker
// values, or a descriptive error otherwise. Exported (package-private)
// so the selection logic is unit-testable without invoking traceRunE.
func validatePyWalker(s string) error {
	switch s {
	case "auto", "ebpf", "userspace":
		return nil
	default:
		return fmt.Errorf("invalid --py-walker %q (must be auto, ebpf, or userspace)", s)
	}
}

// parseRingBufSize parses a human-readable size string (e.g., "32m", "16m",
// "8388608") into a uint32 byte count. Validates that the result is a power
// of 2 and at least 4096 (one page). Returns 0 if the input is empty.
func parseRingBufSize(s string) (uint32, error) {
	if s == "" {
		return 0, nil
	}

	s = strings.TrimSpace(strings.ToLower(s))
	var multiplier uint64 = 1
	numStr := s

	switch {
	case strings.HasSuffix(s, "k"):
		multiplier = 1024
		numStr = s[:len(s)-1]
	case strings.HasSuffix(s, "m"):
		multiplier = 1024 * 1024
		numStr = s[:len(s)-1]
	case strings.HasSuffix(s, "g"):
		multiplier = 1024 * 1024 * 1024
		numStr = s[:len(s)-1]
	}

	n, err := strconv.ParseUint(numStr, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid ringbuf-size %q: %w", s, err)
	}

	total := n * multiplier
	if total == 0 {
		return 0, fmt.Errorf("ringbuf-size %q resolves to 0 bytes", s)
	}
	if total > 1<<32-1 {
		return 0, fmt.Errorf("ringbuf-size %q exceeds uint32 max (4GB)", s)
	}
	if total < 4096 {
		return 0, fmt.Errorf("ringbuf-size %q too small: minimum is 4096 (4k)", s)
	}
	if bits.OnesCount64(total) != 1 {
		return 0, fmt.Errorf("ringbuf-size %q (%d bytes) is not a power of 2", s, total)
	}

	return uint32(total), nil
}

// ncclSetupParams collects the inputs setupNCCLTracer needs. Function
// fields make it injectable for tests (QA audit ★2 #10).
type ncclSetupParams struct {
	explicitLib       string
	targetPIDs        []int
	explicitPIDs      bool
	geteuid           func() int
	hasCapBPF         func() bool
	findLibForPID     func(pid int) string
	findLibSystemwide func() string
	debugf            func(format string, args ...any)
	stderr            io.Writer
}

// setupNCCLTracer resolves libnccl, attaches the NCCL uprobe set, and
// installs the per-tenant PID filter when --pid is explicit. Returns
// (nil, 0) on every soft-fail path (no libnccl, attach error). Warnings
// go to p.stderr; the only hard error path is no caller — callers swallow
// failures and proceed without NCCL tracing.
func setupNCCLTracer(p ncclSetupParams) (*ncclprobe.Tracer, int) {
	// L7: surface a friendly error when running unprivileged. Uprobe
	// attach needs CAP_BPF + CAP_PERFMON on Linux >= 5.8 or root on
	// older kernels. Without this check users get a confusing libbpf
	// "operation not permitted" deep in the attach path.
	if p.geteuid() != 0 && !p.hasCapBPF() {
		fmt.Fprintln(p.stderr, "  Warning: --nccl requires root or CAP_BPF + CAP_PERFMON; attach will likely fail")
		p.debugf("nccl: euid=%d, no CAP_BPF; continuing but expect failure", p.geteuid())
	}
	libPath := p.explicitLib
	if libPath == "" {
		for _, pid := range p.targetPIDs {
			if pid > 0 {
				if lib := p.findLibForPID(pid); lib != "" {
					libPath = lib
					break
				}
			}
		}
	}
	if libPath == "" {
		libPath = p.findLibSystemwide()
	}
	if libPath == "" {
		// v0.15 F1: no eager libnccl found, but the discovery scanner
		// may still find a venv-installed libnccl after the workload
		// boots. Stand up a lazy tracer (BPF spec loaded, ringbuf
		// reader open, no uprobes yet) so the scanner sink can call
		// AttachAt later. Without --libnccl-discovery-interval set,
		// this lazy tracer never gets used and is harmless.
		fmt.Fprintln(p.stderr, "  No eager libnccl found; lazy-attach armed (uprobes wire up when discovery scanner finds a workload)")
		p.debugf("nccl: lazy tracer; AttachAt will fire from the discovery scanner sink")
		nt := ncclprobe.New("")
		var filterPIDs []uint32
		if p.explicitPIDs {
			for _, pid := range p.targetPIDs {
				if pid > 0 {
					filterPIDs = append(filterPIDs, uint32(pid))
				}
			}
		}
		if err := nt.Prepare(filterPIDs); err != nil {
			fmt.Fprintf(p.stderr, "  Warning: NCCL lazy-attach unavailable: %v\n", err)
			p.debugf("nccl: lazy Prepare failed: %v", err)
			return nil, 0
		}
		return nt, 0
	}
	// v0.12.2 (LHF #7 + Arch ★3 attach race): when --pid is set, scope
	// the NCCL probe to just those PIDs. The filter is passed *into*
	// Attach so that the BPF map is populated BEFORE the first uprobe
	// goes live; otherwise there is a window where a sibling tenant's
	// NCCL events would leak into our ringbuf. Without explicit --pid
	// the probe traces system-wide (matching the other tracers).
	var filterPIDs []uint32
	if p.explicitPIDs {
		for _, pid := range p.targetPIDs {
			if pid > 0 {
				filterPIDs = append(filterPIDs, uint32(pid))
			}
		}
	}
	nt := ncclprobe.New(libPath)
	if err := nt.Attach(filterPIDs); err != nil {
		fmt.Fprintf(p.stderr, "  Warning: NCCL tracing unavailable: %v\n", err)
		p.debugf("nccl tracer: attach failed: %v", err)
		return nil, 0
	}
	// v0.12.3 (Sys Arch ★1): real attached count, not a hardcoded
	// constant. Older NCCL builds without ncclCommInitAll attach 16,
	// not 18; the banner must reflect what actually happened.
	probeCount := nt.AttachedProbeCount()
	fmt.Fprintf(p.stderr, "  NCCL tracing: attached %d probes to %s\n", probeCount, libPath)
	p.debugf("nccl tracer: %d probes attached at %s (filter pids=%v)", probeCount, libPath, filterPIDs)
	return nt, probeCount
}

// ---------------------------------------------------------------------------
// Main trace logic
// ---------------------------------------------------------------------------

func traceRunE(cmd *cobra.Command, args []string) error {
	// Single-instance enforcement: refuse to start if another ingero trace
	// is already running on this host. Prevents silent watchdog-pin
	// overwrite (Bug 8) and doubles overhead from concurrent uprobes.
	unlock, err := acquireTraceLock()
	if err != nil {
		return err
	}
	defer unlock()

	// Parse --ringbuf-size early — fail fast on invalid input.
	ringBufBytes, err := parseRingBufSize(traceRingBufSize)
	if err != nil {
		return err
	}
	if ringBufBytes > 0 {
		debugf("ring buffer override applied to cuda/driver/host only (high-throughput probes): %d bytes", ringBufBytes)
	}

	// v0.16 inference umbrella. Loads YAML, layers CLI flags, validates
	// mutex constraints, and applies the umbrella defaults to the
	// trace-command flag vars. Returns the resolved struct so the
	// engine + rollover wiring below can read the post-merge values
	// directly (instead of re-checking each flag).
	cfgPath, _ := cmd.Flags().GetString("config")
	agentCfg, err := config.Load(cfgPath)
	if err != nil {
		return fmt.Errorf("load config %s: %w", cfgPath, err)
	}
	resolvedInfer, err := resolveInferenceConfig(agentCfg, cmd, cfgPath)
	if err != nil {
		return err
	}
	if err := applyInferenceDefaults(cmd, resolvedInfer); err != nil {
		return err
	}

	// Validate --py-walker early — fail fast on invalid input.
	if err := validatePyWalker(tracePyWalker); err != nil {
		return err
	}

	// Warn on the known edge case: at ptrace_scope=3, system-wide
	// (pid=-1) uprobes may not fire. Trace-all mode without --pid
	// relies on those. See Bug 7 for the adversarial repro.
	if tracePyWalker == "ebpf" && len(tracePIDs) == 0 {
		if scope, err := discover.CheckPtraceScope(); err == nil && scope == 3 {
			slog.Warn("ptrace_scope=3 + trace-all + --py-walker=ebpf may not emit cuda events",
				"hint", "pass --pid X or lower kernel.yama.ptrace_scope")
		}
	}

	// Resolve node identity early; fail fast if name contains colon.
	nodeIdentity, err := ResolveNodeIdentity(traceNode)
	if err != nil {
		return err
	}
	debugf("node identity: %s", nodeIdentity)

	// v0.15 Tier 2: experimental closed-driver kprobe gate. Detect
	// the running NVIDIA driver + Linux kernel and check the
	// allowlist. The result drives K/L/M probe-load decisions
	// further down. Logged at startup so operators see the gate
	// outcome.
	expKprobesAllowed := false
	if traceEnableExperimentalKprobes {
		v := kprobe.DetectVersions()
		expKprobesAllowed = kprobe.IsAllowed(v, kprobe.DefaultAllowlist)
		slog.Info(kprobe.DescribeStatus(v, expKprobesAllowed, kprobe.DefaultAllowlist))
	}

	// Rank cache for distributed training rank detection (reads /proc/[pid]/environ).
	rankCache := discover.NewRankCache()

	// --log: redirect log output (stderr-style debug messages) to a file.
	// Append mode, no rotation — intended for debugging, not production logging.
	// Production deployments should use systemd journal or kubectl logs.
	if traceLogPath != "" {
		f, err := os.OpenFile(traceLogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("opening log file %s: %w", traceLogPath, err)
		}
		defer f.Close()
		log.SetOutput(f)
		log.Printf("ingero trace: log output started (debug=%v)", debugMode)
		fmt.Fprintf(os.Stderr, "  Logging to %s\n", traceLogPath)
	}

	// Step 0: Set up graceful shutdown context early — PodCache needs it.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Step 0b: K8s pod metadata enrichment (no-op on bare metal).
	// Initialize early so GPU pod auto-discovery is available for resolveTargets.
	// PodCache polls the K8s API every 30s for pods on this node.
	// Single Run() goroutine with the signal-aware context — no double-start.
	var podCache *k8s.PodCache
	if k8s.IsInCluster() {
		kClient, kErr := k8s.NewInCluster()
		if kErr != nil {
			fmt.Fprintf(os.Stderr, "  K8s API: %v (pod metadata disabled)\n", kErr)
		} else {
			podCache = k8s.NewPodCache(kClient)
			if debugMode {
				podCache.SetDebugLog(debugf)
			}
			go podCache.Run(ctx)
			if err := podCache.WaitReady(ctx, 5*time.Second); err != nil {
				debugf("K8s pod cache not ready: %v (continuing without pod metadata)", err)
			} else {
				fmt.Fprintf(os.Stderr, "  K8s: pod metadata enrichment enabled\n")
			}
		}
	}

	// Step 1: Resolve targets — find libcudart.so and target PIDs.
	//
	// Three modes:
	//   --pid 123,456   → trace exactly those PIDs
	//   --user bob      → trace all CUDA PIDs owned by bob
	//   (default)       → trace all CUDA PIDs owned by SUDO_USER (invoking user)
	libPaths, targetPIDs, processNames, err := resolveTargets(tracePIDs)
	if err != nil {
		return err
	}
	if len(libPaths) == 0 {
		return fmt.Errorf("no CUDA libraries found")
	}
	// Primary library path (first element) — used for display and backward compat.
	libPath := libPaths[0]
	debugf("targets resolved: libs=%v pids=%v names=%v", libPaths, targetPIDs, processNames)

	// If --user not explicitly set and no --pid given, default to SUDO_USER
	// so that "sudo ingero trace" traces the invoking user's processes, not root's.
	if traceUser == "" && len(tracePIDs) == 0 {
		if sudoUser := os.Getenv("SUDO_USER"); sudoUser != "" {
			traceUser = sudoUser
		}
	}

	// If --user specified (or defaulted from SUDO_USER), resolve ALL PIDs
	// for that user. Skip when:
	//   - --pid was explicitly set (explicit PIDs take precedence over --user)
	//   - no CUDA processes running (pre-attach mode: probes fire for any process)
	if traceUser != "" && len(tracePIDs) == 0 && len(targetPIDs) > 0 {
		pids, err := resolvePIDsForUser(traceUser)
		if err != nil {
			return fmt.Errorf("resolving user %q: %w", traceUser, err)
		}
		if len(pids) == 0 {
			return fmt.Errorf("no CUDA processes found for user %q", traceUser)
		}
		targetPIDs = pids
		processNames = resolveProcessNames(pids)
		fmt.Fprintf(os.Stderr, "  User %q: tracing %d CUDA process(es)\n", traceUser, len(pids))
	} else if traceUser != "" && len(tracePIDs) == 0 {
		pids, _ := resolvePIDsForUser(traceUser)
		if len(pids) > 0 {
			targetPIDs = pids
			processNames = resolveProcessNames(pids)
			fmt.Fprintf(os.Stderr, "  User %q: tracing %d CUDA process(es)\n", traceUser, len(pids))
		} else {
			fmt.Fprintf(os.Stderr, "  User %q: no CUDA processes yet — probes will trace all processes\n", traceUser)
		}
	}

	// K8s GPU pod auto-discovery: when running in K8s with no --pid, find
	// PIDs in GPU-requesting pods. Complements CUDA library scanning — catches
	// processes that haven't loaded CUDA yet (e.g., during initialization).
	if podCache != nil && len(tracePIDs) == 0 && len(targetPIDs) == 0 {
		gpuPIDs, gpuErr := k8s.FindGPUPodPIDs(podCache)
		if gpuErr != nil {
			debugf("K8s GPU pod discovery: %v", gpuErr)
		} else if len(gpuPIDs) > 0 {
			targetPIDs = gpuPIDs
			processNames = resolveProcessNames(gpuPIDs)
			fmt.Fprintf(os.Stderr, "  K8s: discovered %d PIDs in GPU pods\n", len(gpuPIDs))
		}
	}

	// Build PID filter for event loop (nil = accept all).
	pidFilter := pidSetFromInts(targetPIDs)

	// Step 2: Create CUDA tracer(s) and attach eBPF uprobes.
	// When multiple libcudart.so copies are discovered (e.g., system + venv),
	// create a tracer per library so probes fire regardless of which copy
	// the workload loads.
	var cudaOpts []cuda.Option
	if traceStack {
		cudaOpts = append(cudaOpts, cuda.WithStackCapture(true))
	}
	if ringBufBytes > 0 {
		cudaOpts = append(cudaOpts, cuda.WithRingBufSize(ringBufBytes))
	}
	// PID-specific uprobe attach when tracing a single target. Reduces
	// kernel-wide uprobe overhead and may work around ptrace_scope=3
	// gates on system-wide perf_event_open (Bug 7).
	if len(tracePIDs) == 1 && tracePIDs[0] > 0 {
		cudaOpts = append(cudaOpts, cuda.WithUprobePID(tracePIDs[0]))
	}

	var cudaTracers []*cuda.Tracer
	var graphTracers []*cudagraph.Tracer
	graphProbeCount := 0

	// Track all attached library paths (resolved) for runtime mismatch detection.
	attachedLibs := make(map[string]bool)

	for _, lp := range libPaths {
		ct := cuda.New(lp, cudaOpts...)
		if err := ct.Attach(); err != nil {
			if len(libPaths) == 1 {
				// Single library — hard failure (original behavior).
				return fmt.Errorf("attaching CUDA probes: %w", err)
			}
			// Multiple libraries — log warning and continue with others.
			fmt.Fprintf(os.Stderr, "  Warning: CUDA probes failed for %s: %v\n", lp, err)
			debugf("CUDA tracer: attach failed for %s: %v", lp, err)
			continue
		}
		cudaTracers = append(cudaTracers, ct)
		attachedLibs[lp] = true
		debugf("CUDA tracer: %d probes attached to %s", ct.ProbeCount(), lp)

		// Step 2b: Create CUDA Graph tracer per library (non-fatal).
		// ringbuf override not applied — graph is low-throughput; uses compiled default.
		gt := cudagraph.New(lp)
		if err := gt.Attach(); err != nil {
			debugf("graph tracer: attach failed for %s: %v", lp, err)
		} else {
			graphTracers = append(graphTracers, gt)
			graphProbeCount += gt.ProbeCount()
			debugf("graph tracer: %d probes attached to %s", gt.ProbeCount(), lp)
		}
	}
	if len(cudaTracers) == 0 {
		return fmt.Errorf("attaching CUDA probes: no libraries could be attached")
	}
	// Defers stack — all tracers close when traceRunE returns.
	for _, ct := range cudaTracers {
		defer ct.Close()
	}
	for _, gt := range graphTracers {
		defer gt.Close()
	}
	if len(graphTracers) == 0 {
		fmt.Fprintf(os.Stderr, "  graph probes: skipped (symbols not found)\n")
	}

	// Use first CUDA tracer as the "primary" for probe count display.
	cudaTracer := cudaTracers[0]

	// --py-walker=ebpf: obtain the py_runtime_map from EVERY cuda tracer.
	// The map is part of cuda_trace.bpf.c via the included python_walker.bpf.h
	// header. Each cuda tracer loads its own BPF objects (one per libcudart.so
	// discovered), so each has its OWN per-instance py_runtime_map. Per-PID
	// walker state must be written to ALL of them — the workload's cuda
	// events are processed by whichever tracer is attached to the libcudart
	// the workload loaded, and that tracer's BPF program only sees its own
	// map. A nil map means bpf2go hasn't yet regenerated the bindings with
	// the walker header — fail loudly so operators run `make generate`.
	var pyMaps []*ebpf.Map
	if tracePyWalker == "ebpf" {
		for _, ct := range cudaTracers {
			m := ct.PyRuntimeMap()
			if m == nil {
				return fmt.Errorf("--py-walker=ebpf requires py_runtime_map " +
					"(run 'make generate' after the walker header was added to cuda_trace.bpf.c)")
			}
			pyMaps = append(pyMaps, m)
		}
		slog.Info("Python walker: using kernel-side eBPF walker (--py-walker=ebpf)",
			"cuda_tracers", len(cudaTracers))
	}

	// Step 3: Create host tracer (non-fatal — graceful degradation).
	// Always attach host tracepoints. When no PIDs are targeted, the BPF
	// target_pids map starts empty — PIDs are added dynamically as CUDA
	// events arrive. This ensures host correlation works even without --pid.
	var hostTracer *host.Tracer
	hostProbeCount := 0
	{
		// Seed with first PID (host.New takes one initial PID, 0 = none).
		initialPID := singlePIDOrZero(targetPIDs)
		var hostOpts []host.Option
		if ringBufBytes > 0 {
			hostOpts = append(hostOpts, host.WithRingBufSize(ringBufBytes))
		}
		ht := host.New(uint32(initialPID), hostOpts...)
		if err := ht.Attach(); err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: host tracepoints unavailable: %v\n", err)
			fmt.Fprintf(os.Stderr, "  Continuing with CUDA-only mode.\n\n")
			debugf("host tracer: attach failed: %v", err)
		} else {
			hostTracer = ht
			hostProbeCount = 4 // sched_switch, sched_wakeup, mm_page_alloc, oom_kill
			// Seed all target PIDs into the BPF map.
			for _, pid := range targetPIDs {
				if pid > 0 {
					ht.SetTargetPID(uint32(pid))
				}
			}
			if len(targetPIDs) > 0 {
				debugf("host tracer: %d probes attached (%d target PIDs)", hostProbeCount, len(targetPIDs))
			} else {
				debugf("host tracer: %d probes attached (dynamic PID tracking)", hostProbeCount)
			}
		}
	}
	if hostTracer != nil {
		defer hostTracer.Close()
	}

	// Step 3b: Create driver API tracer (non-fatal — not all systems have libcuda.so).
	var driverTracer *driver.Tracer
	driverProbeCount := 0
	if libcudaPath, err := discover.FindLibCUDA(); err == nil {
		debugf("libcuda.so found at %s", libcudaPath)
		var driverOpts []driver.Option
		if traceStack {
			driverOpts = append(driverOpts, driver.WithStackCapture(true))
		}
		if ringBufBytes > 0 {
			driverOpts = append(driverOpts, driver.WithRingBufSize(ringBufBytes))
		}
		if len(tracePIDs) == 1 && tracePIDs[0] > 0 {
			driverOpts = append(driverOpts, driver.WithUprobePID(tracePIDs[0]))
		}
		dt := driver.New(libcudaPath, driverOpts...)
		if err := dt.Attach(); err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: driver API tracing unavailable: %v\n", err)
			debugf("driver tracer: attach failed: %v", err)
		} else {
			driverTracer = dt
			driverProbeCount = dt.ProbeCount()
			debugf("driver tracer: %d probes attached to %s", driverProbeCount, libcudaPath)
		}
	} else {
		debugf("libcuda.so not found: %v", err)
	}
	if driverTracer != nil {
		defer driverTracer.Close()
	}

	// Step 3c: Create block I/O tracer (non-fatal).
	// ringbuf override not applied — block I/O is low-throughput; uses compiled default.
	var ioTracer *blockio.Tracer
	ioProbeCount := 0
	if !traceNoIO {
		iot := blockio.New()
		if err := iot.Attach(); err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: block I/O tracing unavailable: %v\n", err)
			debugf("I/O tracer: attach failed: %v", err)
		} else {
			ioTracer = iot
			ioProbeCount = 2
			debugf("I/O tracer: %d tracepoints attached", ioProbeCount)
		}
	}
	if ioTracer != nil {
		defer ioTracer.Close()
	}

	// Step 3d: Create TCP retransmit tracer (non-fatal).
	// ringbuf override not applied — TCP is low-throughput; uses compiled default.
	var tcpTracer *tcp.Tracer
	tcpProbeCount := 0
	if !traceNoTCP {
		tt := tcp.New()
		if err := tt.Attach(); err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: TCP tracing unavailable: %v\n", err)
			debugf("TCP tracer: attach failed: %v", err)
		} else {
			tcpTracer = tt
			tcpProbeCount = 1
			debugf("TCP tracer: %d tracepoint attached", tcpProbeCount)
		}
	}
	if tcpTracer != nil {
		defer tcpTracer.Close()
	}

	// Step 3e: Create network socket tracer (non-fatal).
	// ringbuf override not applied — net is low-throughput; uses compiled default.
	var netTracer *nettracer.Tracer
	netProbeCount := 0
	if !traceNoNet {
		nt := nettracer.New()
		if err := nt.Attach(); err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: network tracing unavailable: %v\n", err)
			debugf("net tracer: attach failed: %v", err)
		} else {
			netTracer = nt
			netProbeCount = 4
			debugf("net tracer: %d tracepoints attached", netProbeCount)
			// Seed net PID filter ONLY when --pid is explicit. Auto-discovered
			// PIDs are CUDA processes — seeding them blocks all non-CUDA network
			// traffic, producing zero net events in practice. With an empty map
			// the BPF filter traces all PIDs (net_pid_map_empty() → true).
			if len(tracePIDs) > 0 {
				for _, pid := range targetPIDs {
					if pid > 0 {
						nt.SetTargetPID(uint32(pid))
					}
				}
			}
		}
	}
	if netTracer != nil {
		defer netTracer.Close()
	}

	// Step 3f: Optional NCCL tracer (v0.12.0 opt-in). Discover libnccl.so
	// via /proc/<pid>/maps if --nccl is set and --nccl-lib wasn't given.
	// Falls back to libtorch_cuda.so / libtorch_global_deps.so for PyTorch
	// builds that statically link NCCL.
	var ncclTracer *ncclprobe.Tracer
	if traceNCCL {
		ncclTracer, _ = setupNCCLTracer(ncclSetupParams{
			explicitLib:       traceNCCLLib,
			targetPIDs:        targetPIDs,
			explicitPIDs:      len(tracePIDs) > 0,
			geteuid:           os.Geteuid,
			hasCapBPF:         ncclprobe.HasCapBPF,
			findLibForPID:     ncclprobe.FindLibNCCL,
			findLibSystemwide: ncclprobe.FindLibNCCLSystemwide,
			debugf:            debugf,
			stderr:            os.Stderr,
		})
	}
	if ncclTracer != nil {
		defer ncclTracer.Close()
	}

	// Step 4: Create stats collector and correlation engine.
	collector := stats.New()
	corr := correlate.New(correlate.WithWindowMode(traceWorkloadType))
	corr.SetNode(nodeIdentity)

	// Step 4b: Create OTEL exporters (disabled by default).
	var otlpExporter *export.OTLPExporter
	if traceOTLP != "" {
		// Refuse plaintext OTLP on non-loopback endpoints unless the operator
		// explicitly opts in. The payload contains per-PID workload identity;
		// any in-path observer reads it without authentication.
		if traceOTLPInsecure && !IsLoopback(traceOTLP) && !traceOTLPAllowNonLoopback {
			return fmt.Errorf("--otlp-insecure with non-loopback endpoint %q refused; payload includes workload fingerprints. Pass --otlp-insecure-allow-non-loopback to override or drop --otlp-insecure to use HTTPS", traceOTLP)
		}
		otlpExporter = export.NewOTLP(export.OTLPConfig{
			Endpoint:  traceOTLP,
			Insecure:  traceOTLPInsecure,
			NodeID:    nodeIdentity,
			ClusterID: traceCluster,
			DebugLog:  debugf,
		})
		debugf("OTLP: configured endpoint=%s node=%s cluster=%s insecure=%v", traceOTLP, nodeIdentity, traceCluster, traceOTLPInsecure)
	}

	var promSrv *export.PrometheusServer
	if traceProm != "" {
		// The /metrics body emits per-PID workload identity (model names,
		// kernel-launch fingerprints, cgroup hashes, NCCL comm metadata).
		// On a non-loopback bind, every host on the network can scrape it
		// without authentication. The listener stays plaintext-no-auth by
		// design (Prometheus scrape convention); the caller is responsible
		// for fronting it with TLS+auth or restricting reachability.
		if !IsLoopback(traceProm) {
			fmt.Fprintf(os.Stderr, "  WARNING: --prometheus %s binds to a non-loopback interface and exposes per-PID workload identity unauthenticated.\n", traceProm)
			fmt.Fprintf(os.Stderr, "  Restrict to loopback (e.g. --prometheus 127.0.0.1:9090), front with TLS+auth (Prometheus Agent / OTLP collector), or firewall the port.\n")
		}
		promSrv = export.NewPrometheus(traceProm)
	}

	// Step 4c: Open SQLite store (recording is on by default).
	if !traceRecord && traceRecordAll {
		fmt.Fprintf(os.Stderr, "  Warning: --record-all ignored because --record=false\n")
	}
	var eventStore *store.Store
	var sessionID int64
	var procNames *pidNameCache
	if traceRecord {
		dbPath := traceDBPath
		if dbPath == "" {
			dbPath = store.DefaultDBPath()
		}
		s, err := store.New(dbPath)
		if err != nil {
			return fmt.Errorf("opening database for recording: %w", err)
		}
		eventStore = s
		eventStore.SetNode(nodeIdentity)
		debugf("recording to %s (node=%s)", dbPath, nodeIdentity)

		// Set size-based DB limit if --max-db is specified.
		if traceMaxDB != "" && traceMaxDB != "0" {
			maxBytes, err := store.ParseSize(traceMaxDB)
			if err != nil {
				s.Close() // close DB before returning — defer isn't registered yet
				return fmt.Errorf("parsing --max-db %q: %w", traceMaxDB, err)
			}
			eventStore.SetMaxDBSize(maxBytes)
		}

		// v0.16: file-level DB rollover (mutually exclusive with
		// --max-db, enforced by resolveInferenceConfig). When the
		// umbrella is engaged, applyInferenceDefaults has already
		// flipped --max-db off and set --db-rollover-size=1g; here we
		// just plumb the resolved values into the store.
		if rerr := configureRollover(eventStore); rerr != nil {
			s.Close()
			return rerr
		}

		// v0.16: per-workload step-duration baseliner + sampler. No-op
		// when --inference is not engaged (returns nil engine and nil
		// sampler). Captured into package-level vars so the event
		// handler and snapshot loop below can route events into them
		// without additional plumbing.
		inferEngine, inferSampler = configureInferenceEngine(eventStore)
		_ = inferSampler // captured via Engine config; reference here keeps compiler happy when the symbol is otherwise unused

		// v0.16.2: engine /metrics scraper. Auto-detects vLLM/TGI/
		// SGLang/Triton from the target PIDs' cmdlines and pulls
		// /metrics every --inference-scrape-interval. Maps
		// engine-specific Prometheus names to OTel GenAI semantic
		// conventions (TTFT, TPOT, prefill/decode, token usage).
		// No-op when --inference is off or --inference-scrape=off.
		inferScraper = configureInferenceScraper(targetPIDs)
		if inferScraper != nil {
			go func() {
				if err := inferScraper.Run(ctx); err != nil &&
					err != context.Canceled {
					debugf("infer scrape Run: %v", err)
				}
			}()
		}

		// When KV-cache lineage is engaged the agent must keep the
		// cudaMalloc / cudaFree BPF probes' watchdog gate open. The
		// gate exists so memtrack-style remediation can stop tracking
		// allocations when the orchestrator dies; here the agent
		// itself is the consumer, so we self-heartbeat. 25 ms cadence
		// stays well below the 50 ms staleness floor in
		// common.bpf.h's watchdog_is_stale.
		if traceInferenceKVCacheLineage && cudaTracer != nil {
			go func() {
				ticker := time.NewTicker(25 * time.Millisecond)
				defer ticker.Stop()
				// Fire once immediately so the first cudaMalloc /
				// cudaFree after agent attach already passes the gate.
				if err := cudaTracer.WriteWatchdogHeartbeat(); err != nil {
					debugf("infer kvcache: initial watchdog heartbeat failed: %v", err)
				}
				for {
					select {
					case <-ctx.Done():
						return
					case <-ticker.C:
						if err := cudaTracer.WriteWatchdogHeartbeat(); err != nil {
							debugf("infer kvcache: watchdog heartbeat: %v", err)
						}
					}
				}
			}()
		}

		// Record session metadata (GPU model, driver, CPU, OS, etc.).
		// These are ~1ms reads at startup — no overhead concern.
		kernelVer, _ := discover.KernelVersion()
		session := store.Session{
			StartedAt: time.Now(),
			GPUModel:  discover.CheckGPUModel().Value,
			GPUDriver: discover.CheckNVIDIA().Value,
			CPUModel:  discover.CPUModel(),
			CPUCores:  discover.CPUCores(),
			MemTotal:  sysinfo.MemTotalMB(),
			Kernel:    kernelVer,
			OSRelease: discover.OSRelease(),
			CUDAVer:   discover.CUDAVersion(),
			PythonVer: discover.PythonVersion(),
			IngeroVer: version.String(),
			PIDFilter: formatPIDFilter(targetPIDs),
			Flags:     formatTraceFlags(),
			Node:      nodeIdentity,
		}
		// Detect distributed training rank from the first target PID.
		if len(targetPIDs) > 0 && targetPIDs[0] > 0 {
			ri := rankCache.Lookup(uint32(targetPIDs[0]))
			session.Rank = ri.Rank
			session.LocalRank = ri.LocalRank
			session.WorldSize = ri.WorldSize
		}
		sessionID, err = eventStore.StartSession(session)
		if err != nil {
			debugf("failed to record session: %v", err)
		} else {
			debugf("session %d started", sessionID)
		}

		// Record PID→name mappings for query-time enrichment.
		for i, pid := range targetPIDs {
			name := ""
			if i < len(processNames) {
				name = processNames[i]
			}
			if name != "" {
				eventStore.RecordProcessName(uint32(pid), name)
			}
		}

		defer func() {
			// Wait for the batch writer goroutine to finish flushing
			// before writing session metadata or closing the DB.
			eventStore.WaitDone()
			// Flush discovered PID→name mappings to SQLite.
			if procNames != nil {
				eventStore.RecordProcessNames(procNames.Names())
			}
			if sessionID > 0 {
				if err := eventStore.StopSession(sessionID, time.Now()); err != nil {
					debugf("failed to stop session: %v", err)
				}
			}
			eventStore.Compact()
			eventStore.Close()
			fmt.Fprintf(os.Stderr, "  Recorded events to %s\n", dbPath)
		}()
	}

	// Step 4d: Create symbol resolver if --stack is set.
	var resolver *symtab.Resolver
	if traceStack {
		if debugMode {
			symtab.SetDebugLog(debugf)
		}
		resolver = symtab.NewResolver()
	}

	// Step 5: Apply --duration timeout to the signal-aware context.
	if traceDuration > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, traceDuration)
		defer cancel()
	}

	// Deadband filter for system snapshots (nil = disabled / no-op).
	snapFilter := filter.Config{
		DeadbandPct:       traceDeadbandPct,
		HeartbeatInterval: traceHeartbeat,
	}.NewSnapshotFilter()

	// Sum CUDA probe counts across all tracers.
	totalCUDAProbes := 0
	for _, ct := range cudaTracers {
		totalCUDAProbes += ct.ProbeCount()
	}

	// Step 6: Print header.
	printTraceHeader(libPath, targetPIDs, processNames, totalCUDAProbes, graphProbeCount, hostProbeCount, driverProbeCount, ioProbeCount, tcpProbeCount, netProbeCount, snapFilter)

	// Step 7: Launch tracers and merge event channels.
	for _, ct := range cudaTracers {
		go ct.Run(ctx)
	}
	if hostTracer != nil {
		go hostTracer.Run(ctx)
	}
	if driverTracer != nil {
		go driverTracer.Run(ctx)
	}

	// Launch new tracers — each feeds into the merged channel.
	var extraChs [](<-chan events.Event)
	// Additional CUDA tracers (beyond the primary) feed into extraChs.
	for _, ct := range cudaTracers[1:] {
		extraChs = append(extraChs, ct.Events())
	}
	for _, gt := range graphTracers {
		go gt.Run(ctx)
		extraChs = append(extraChs, gt.Events())
	}
	if ioTracer != nil {
		go ioTracer.Run(ctx)
		extraChs = append(extraChs, ioTracer.Events())
	}
	if tcpTracer != nil {
		go tcpTracer.Run(ctx)
		extraChs = append(extraChs, tcpTracer.Events())
	}
	if netTracer != nil {
		go netTracer.Run(ctx)
		extraChs = append(extraChs, netTracer.Events())
	}
	// NCCL tracer: drains ringbuf into a snapshot-attached buffer that the
	// onSnapshot callback drains and feeds to OTLP / Prometheus exporters.
	// v0.12.0 keeps NCCL events out of the main events.Event pipeline
	// (correlator + SQLite store don't yet have a NCCL row shape); v0.12.1
	// will fold nccl.collective.* into the unified flow + the SQLite store.
	if ncclTracer != nil {
		go func() {
			if err := ncclTracer.Run(ctx); err != nil {
				debugf("nccl tracer Run: %v", err)
			}
		}()
		go func() {
			n := uint64(0)
			for ev := range ncclTracer.Events() {
				n++
				if traceVerbose {
					fmt.Fprintln(os.Stderr, ev.String())
				}
				if n%1000 == 0 {
					debugf("nccl events seen: %d", n)
				}
				// Convert ncclprobe.Event into a stats.NCCLDataPoint and
				// queue for the next snapshot tick. Skip the header-only
				// CommInitRank / CommDestroy events (op codes 1/2): those
				// are lifecycle events with no duration data dashboards
				// can plot.
				if ev.Op == 1 || ev.Op == 2 {
					continue
				}
				ncclBufferAdd(stats.NCCLDataPoint{
					TimestampUnixNano: int64(ev.TimestampNs),
					OpType:            ev.OpName(),
					CommIDHash:        fmt.Sprintf("%016x", ev.CommIDHash),
					Rank:              ev.Rank,
					NRanks:            ev.NRanks,
					Datatype:          ev.Datatype,
					ReduceOp:          ev.ReduceOp,
					DurationMs:        float64(ev.DurationNs) / 1e6,
					CountBytes:        ev.CountBytes,
					ReturnCode:        ev.ReturnCode,
					PeerRank:          ev.PeerRank,
				})
				// v0.12.1 (LHF #1 follow-on): record for later
				// barrier-wait correlation against the next
				// cudaStreamSynchronize on the same (pid, stream).
				recordNCCLForBarrier(ev)

				// v0.16.1: feed NCCL participation into the infer
				// engine's phase classifier. NCCL on a stream is
				// the strongest signal for prefill (tensor-parallel
				// allreduce); the classifier fires rule 1 on any
				// non-zero NCCL count. Engine method is no-op when
				// the phase classifier is disabled.
				if inferEngine != nil {
					cgroupHash := ""
					if inferCgroupCache != nil {
						cgroupHash = inferCgroupCache.Resolve(ev.PID)
					}
					inferEngine.OnNCCLEvent(ev.PID, cgroupHash, ev.StreamHandle, time.Unix(0, int64(ev.TimestampNs)))
				}
			}
			fmt.Fprintf(os.Stderr, "  NCCL tracing: %d events captured (%d dropped)\n", n, ncclTracer.Dropped())
		}()
	}

	// Start SQLite writer if recording.
	if eventStore != nil {
		go eventStore.Run(ctx)
	}

	// v0.12.1 (LHF #1): tap the CUDA tracer channel for stream-sync
	// events when --nccl is on, so the barrier-wait correlator gets
	// every cudaStreamSynchronize. The forked channel still carries
	// every CUDA event into the merged stream so existing
	// downstream consumers (correlator, stats, store) are unaffected.
	cudaCh := cudaTracer.Events()
	if traceNCCL {
		cudaCh = forkCUDAForBarrier(ctx, cudaCh)
	}

	// Fan-in: merge all event channels into one.
	merged := mergeAllEventChannels(ctx, cudaCh, hostTracer, driverTracer, extraChs...)

	// --- Begin remediate wiring ---
	var tracker *memtrack.Tracker
	var remediateSrv *remediate.Server
	if traceRemediate {
		log.Printf("INFO: remediate: starting -- connect an external consumer to /tmp/ingero-remediate.sock (see docs/remediation-protocol_fleet.md)")
		gpuVRAM, err := memtrack.DetectGPUVRAM()
		if err != nil {
			log.Printf("ERROR: remediate: vram_detection_failed error=%v", err)
			// Fall through — degrade to OSS mode (tracker stays nil).
		} else {
			for gpuID, vram := range gpuVRAM {
				log.Printf("INFO: remediate: vram_detected gpu_id=%d total_vram_mib=%d total_vram_bytes=%d", gpuID, vram/(1024*1024), vram)
			}
			srv := remediate.NewServer("")
			srv.SetSocketGid(traceRemediateGid)
			if err := srv.Start(); err != nil {
				log.Printf("ERROR: remediate: uds_bind_failed path=/tmp/ingero-remediate.sock error=%v", err)
				// Fall through — degrade to OSS mode (tracker stays nil).
			} else {
				remediateSrv = srv
				tracker = memtrack.NewTracker(gpuVRAM, srv.Send)
				defer srv.Close()
				log.Printf("INFO: remediate: enabled gpu_count=%d socket=/tmp/ingero-remediate.sock", len(gpuVRAM))
			}
		}
	}
	// --- End remediate wiring ---

	// --- Begin straggler detector wiring ---
	var stragglerDetector *straggler.Detector
	if traceRemediate {
		var sink straggler.Sink
		if remediateSrv != nil {
			sink = remediateSrv
		}
		stragglerDetector = straggler.NewDetector(straggler.DefaultConfig(), sink)
		go stragglerDetector.Run(ctx)
		log.Printf("INFO: straggler: detector_started config=%s", straggler.DefaultConfig())
	}
	// --- End straggler detector wiring ---

	// Start Prometheus server if configured.
	if promSrv != nil {
		go promSrv.Start(ctx)
	}

	// Start OTLP exporter if configured.
	if otlpExporter != nil {
		go otlpExporter.Start(ctx)
	}

	// Start NVML clock-throttle reason poller (v0.12.10 W2-poller). Only
	// useful when an exporter is configured; otherwise the buffer would
	// fill and never be drained. The poller is no-op when nvidia-smi is
	// not on PATH (e.g. in dev containers without GPU drivers).
	if (otlpExporter != nil || promSrv != nil) && traceThrottlePoll > 0 {
		startThrottlePoller(ctx, traceThrottlePoll, nvml.NewSubprocessRunner(), slog.Default())
	}

	// Start libnccl process-discovery scanner (v0.14 item A). Same gate
	// as throttle: only useful when an exporter is wired up.
	if (otlpExporter != nil || promSrv != nil) && traceLibNCCLDiscoveryInterval > 0 {
		// v0.15 F1: pass ncclTracer so the scanner sink attaches
		// uprobes against runtime-discovered libnccl paths
		// (PyTorch+pip workloads that ship libnccl in a venv).
		startNCCLDiscoveryScanner(ctx, traceLibNCCLDiscoveryInterval, slog.Default(), ncclTracer)
	}

	// Start NVML memfrag poller (v0.14 item D, W1 baseline).
	if (otlpExporter != nil || promSrv != nil) && traceMemFragPollInterval > 0 {
		startMemFragPoller(ctx, traceMemFragPollInterval, nvml.NewMemoryRunner(), nvml.NewComputeAppsRunner(), slog.Default())
	}

	// v0.15 item K + M: experimental closed-driver kprobes /
	// uprobes. Only attempted when (a) the operator passed
	// --enable-experimental-kprobes AND (b) the driver/kernel pair
	// is on internal/kprobe.DefaultAllowlist. Failure to attach
	// (kprobe target absent, libcuda missing) is logged and the
	// agent continues; the gauges/counters simply stay empty.
	if expKprobesAllowed && (otlpExporter != nil || promSrv != nil) {
		startMemfragTracer(ctx, slog.Default())
		// kernel-launch uprobe needs the libcuda path. Reuse the
		// driver tracer's discovery: if libcuda was found above,
		// use the same path; otherwise skip the kernel-launch probe
		// silently (the same behavior the driver tracer has).
		if libcudaPath, err := discover.FindLibCUDA(); err == nil {
			startKernelLaunchTracer(ctx, libcudaPath, slog.Default())
		} else {
			slog.Info("kernel-launch tracer: libcuda not found; skipping", "err", err)
		}
	}

	// Snapshot callback for exporters (OTLP, Prometheus).
	// Called every 1s from the table/JSON mode tickers.
	// OTLP push is rate-limited: only every ExportInterval seconds (default 10s).
	// Only created when at least one exporter is configured, to avoid spinning
	// a sysinfo.Collector goroutine (reading /proc every second) for nothing.
	var onSnapshot func(*stats.Snapshot)
	// droppedFn is the sum of ring-buffer + channel drops across every
	// tracer, used by both the TUI and the snapshot callback. Forward-
	// declared here so the onSnapshot closure below can reference it;
	// assigned further down once all tracers are set up.
	var droppedFn func() uint64
	if promSrv != nil || otlpExporter != nil {
		var otlpPushCount int
		onSnapshot = func(snap *stats.Snapshot) {
			// Attach trace-DB stats whenever a Store is wired in. Cheap
			// (os.Stat + atomic load) and published via the Prometheus
			// /metrics endpoint so operators can monitor prune health.
			if eventStore != nil {
				st := eventStore.ReadStats()
				snap.TraceDB = &stats.TraceDBSnapshot{
					DiskBytes:  st.DiskBytes,
					PrunedRows: st.PrunedRows,
				}
			}
			// Ring-buffer overflow total. Summed across every tracer so
			// operators can alert on "burst larger than ring buffer"
			// without subscribing to per-tracer series. droppedFn is
			// defined further below once tracers are set up; the nil
			// guard keeps this closure safe for snapshots that fire
			// before tracer attachment completes.
			if droppedFn != nil {
				snap.RingbufOverflows = droppedFn()
			}
			// Drain NCCL events captured between snapshots into the
			// snapshot so the OTLP exporter emits them as
			// nccl.collective.* metrics. Empty when --nccl is off.
			snap.NCCLDataPoints = ncclBufferDrain()
			// Drain the latest NVML clock-throttle reading per GPU
			// (v0.12.10 W2-poller). Last-value-wins semantics: a gauge
			// is "current state", not a time series of distinct events,
			// so the poller's per-UUID map flushes here. Nil when
			// nvidia-smi is missing or the poller is disabled.
			snap.ThrottleReadings = drainThrottleBuf()
			// v0.14 item A: latest libnccl discovery batch (gauge
			// semantics, last-batch-wins). Persists across snapshot
			// ticks until the scanner pushes a new batch.
			snap.NCCLProcessReadings = drainNCCLDiscoveryBuf()
			// v0.14 item D: latest NVML memfrag poll snapshot.
			snap.MemFragReadings, snap.MemFragProcessReadings = drainMemFragBuf()
			// v0.14 item C: per-direction memcpy aggregates.
			snap.MemcpyDirReadings = drainMemcpyStats()
			// v0.15 F2: NCCL collective running counters (Prometheus
			// pull-friendly view of nccl.collective.* events).
			snap.NCCLCollectiveCounters = snapshotNCCLCollectiveCounters()
			// v0.15 item K: per-cmd memfrag IOCTL counters.
			snap.MemfragIOCTLCounters = snapshotMemfragCounters()
			// v0.15 item L: throttle event-edge counters.
			ec := throttleEdgeDetector.Snapshot()
			snap.ThrottleEvents = stats.ThrottleEventCounters{
				PowerEvents:   ec.PowerEvents,
				ThermalEvents: ec.ThermalEvents,
				SWEvents:      ec.SWEvents,
				HWEvents:      ec.HWEvents,
			}
			// v0.15 item M: per-PID kernel-launch aggregates.
			snap.KernelLaunches = snapshotKernelLaunchCounters()

			// v0.16 inference outliers. Drain the engine's queue and
			// publish each outlier on the FOSS UDS socket (when
			// --remediate is enabled). The slog INFO line in
			// internal/infer/engine.go's maybeLogOutlier handles the
			// rate-limited operator log.
			//
			// v0.16.3 expands the on-wire surface: OTLP histogram +
			// per-bucket counter + sampler-degraded gauge + thermal
			// context. The per-workload + engine-stats + sampler-state
			// triple is captured here so the OTLP/Prometheus exporters
			// see a fresh view on every push.
			if inferEngine != nil {
				outliers := inferEngine.Drain()
				// Use the resolved nodeIdentity (post --node defaulting +
				// hostname fallback) and the operator-supplied --cluster
				// flag rather than the raw flags. This keeps the UDS
				// envelope's NodeID + ClusterID consistent with the OTLP
				// resource attributes set above; without the resolved
				// values, the orchestrator's per-node correlation map
				// silently keys on empty strings.
				for _, oe := range outliers {
					emitInferOutlier(remediateSrv, nodeIdentity, traceCluster, oe)
				}
				for _, sd := range inferEngine.DrainSampler() {
					emitInferSamplerDegraded(remediateSrv, nodeIdentity, traceCluster, sd)
				}
				snap.InferWorkloads, snap.InferStats, snap.InferSampler = inferEngine.SnapshotForExport()
				// Enrich each workload row with model + engine
				// identity from the scraper. When the PID is a known
				// inference engine (vLLM / TGI / SGLang / Triton) the
				// scraper has the cmdline-extracted model id. The
				// Fleet-side groupBy uses these labels to aggregate
				// peer baselines across pods of the same job.
				if inferScraper != nil {
					for i := range snap.InferWorkloads {
						pid := snap.InferWorkloads[i].PID
						snap.InferWorkloads[i].ModelName = inferScraper.LookupModel(pid)
						snap.InferWorkloads[i].EngineSystem = inferScraper.LookupEngine(pid)
					}
				}
				// Per-outlier OTLP spans. Push on every snapshot tick
				// where at least one outlier was drained so operators
				// can jump from a slow request in Tempo / Honeycomb
				// straight to the workload-key context. Each outlier
				// becomes a self-contained span (own trace_id) with
				// status=ERROR; cross-outlier correlation lives on the
				// workload-key attribute set, not on parent_span_id.
				if otlpExporter != nil && len(outliers) > 0 {
					spans := buildOutlierSpans(outliers, inferScraper)
					if err := otlpExporter.PushSpans(ctx, spans); err != nil {
						debugf("OTLP traces: push error: %v", err)
					}
				}
			}
			if promSrv != nil {
				promSrv.UpdateSnapshot(snap)
			}
			if otlpExporter != nil {
				otlpPushCount++
				if otlpPushCount%otlpExporter.Interval() == 0 {
					if err := otlpExporter.Push(ctx, snap); err != nil {
						debugf("OTLP: push error: %v", err)
					}
				}
			}
		}
	}

	// droppedDetailFn captures tracer slices by reference. These slices are
	// immutable after this point — do not append to cudaTracers/graphTracers.

	// Combined dropped count from all tracers.
	droppedFn = func() uint64 {
		var d uint64
		for _, ct := range cudaTracers {
			d += ct.Dropped()
		}
		if hostTracer != nil {
			d += hostTracer.Dropped()
			d += hostTracer.CriticalDropped()
		}
		if driverTracer != nil {
			d += driverTracer.Dropped()
		}
		if ioTracer != nil {
			d += ioTracer.Dropped()
		}
		if tcpTracer != nil {
			d += tcpTracer.Dropped()
		}
		if netTracer != nil {
			d += netTracer.Dropped()
		}
		return d
	}

	// Per-tracer drop breakdown for display.
	droppedDetailFn := func() string {
		var cudaD, graphD uint64
		for _, ct := range cudaTracers {
			cudaD += ct.Dropped()
		}
		for _, gt := range graphTracers {
			graphD += gt.Dropped()
		}
		var hostD, hostCritD, driverD, ioD, tcpD, netD uint64
		if hostTracer != nil {
			hostD = hostTracer.Dropped()
			hostCritD = hostTracer.CriticalDropped()
		}
		if driverTracer != nil {
			driverD = driverTracer.Dropped()
		}
		if ioTracer != nil {
			ioD = ioTracer.Dropped()
		}
		if tcpTracer != nil {
			tcpD = tcpTracer.Dropped()
		}
		if netTracer != nil {
			netD = netTracer.Dropped()
		}
		return fmt.Sprintf("cuda=%d driver=%d host=%d host_crit=%d io=%d tcp=%d net=%d graph=%d",
			cudaD, driverD, hostD, hostCritD, ioD, tcpD, netD, graphD)
	}

	// Dynamic PID tracking: when tracing all processes (no --pid), we add
	// PIDs to the host tracer's BPF map as we discover them from CUDA events.
	// This ensures host correlation works without --pid.
	trackedPIDs := make(map[uint32]bool)
	for _, pid := range targetPIDs {
		if pid > 0 {
			trackedPIDs[uint32(pid)] = true
		}
	}

	// cudaPIDs tracks PIDs with CUDA Runtime or Driver API activity only.
	// Used by shouldStore() to filter sched_switch storage — IO/TCP/Net PIDs
	// are system-wide tracepoints that would widen the set to include system
	// daemons, defeating the storage filter. trackedPIDs (above) is broader
	// and includes IO/TCP/Net PIDs for eBPF target_pids enrollment.
	cudaPIDs := make(map[uint32]bool)

	// Correlator PID: single PID for single-process, 0 for multi/all (aggregate).
	corrPID := singlePIDOrZero(targetPIDs)
	trackPID := func(pid uint32) {
		if pid == 0 {
			return
		}
		if !trackedPIDs[pid] {
			trackedPIDs[pid] = true
			if hostTracer != nil {
				hostTracer.SetTargetPID(pid)
				debugf("host tracer: dynamically added PID %d", pid)
			}
			if netTracer != nil && len(tracePIDs) > 0 {
				netTracer.SetTargetPID(pid)
				debugf("net tracer: dynamically added PID %d", pid)
			}
			// Seed py_runtime_map for dynamically-discovered PIDs. Without
			// this, the eBPF walker's state push relies on HostProcessExec
			// events — which the host tracer suppresses for PIDs not yet in
			// target_pids, so post-trace-start workloads never get their
			// state pushed. Dedup via pyPushedPIDs prevents duplicate
			// harvester runs when HostProcessExec fires later for this PID.
			tryPushPyRuntimeStateOnce(pid, pyMaps)
		}
	}

	// Apply initial sampling rate (fixed or adaptive baseline).
	// Rate 0 from the flag means "adaptive" — start at 1 (emit all) and
	// let runAdaptiveSamplingMonitor adjust based on drop pressure.
	initialRate := traceSamplingRate
	if initialRate == 0 {
		initialRate = 1
	}
	for _, ct := range cudaTracers {
		if err := ct.SetSamplingRate(initialRate); err != nil {
			debugf("setting cuda sampling rate: %v", err)
		}
	}
	if driverTracer != nil {
		if err := driverTracer.SetSamplingRate(initialRate); err != nil {
			debugf("setting driver sampling rate: %v", err)
		}
	}
	for _, gt := range graphTracers {
		if err := gt.SetSamplingRate(initialRate); err != nil {
			debugf("setting graph sampling rate: %v", err)
		}
	}

	// Launch adaptive sampling monitor if --sampling-rate was not set (or was 0).
	if traceSamplingRate == 0 {
		go runAdaptiveSamplingMonitor(ctx, cudaTracers, driverTracer, graphTracers, droppedFn)
	}

	// Step 8: Build PID→name cache for JSON output enrichment.
	procNames = newPIDNameCache(targetPIDs, processNames)

	// Library mismatch checker: on first CUDA event per PID, verify the
	// process's loaded libcudart.so matches one of our attached libraries.
	// Warns once per PID if there's a mismatch (e.g., venv library vs system).
	mismatchCheck := newLibMismatchChecker(attachedLibs)

	// Seed py_runtime_map for already-running target Python processes.
	// Three seed points feed into tryPushPyRuntimeStateOnce, which dedups
	// per-PID via pyPushedPIDs:
	//   1. Startup loop (here) — for PIDs known via --pid X.
	//   2. trackPID closure (above) — fires on first non-host event for a
	//      dynamically-discovered PID. Covers workloads that started AFTER
	//      the trace and whose sched_process_exec was suppressed by the
	//      host tracer's target_pids gate.
	//   3. HostProcessExec handler in the event loop — re-pushes after an
	//      exec (binary may have changed).
	// HostProcessExit clears the dedup mark alongside pytrace.ClearPID.
	if len(pyMaps) > 0 {
		for _, pid := range targetPIDs {
			if pid > 0 {
				tryPushPyRuntimeStateOnce(uint32(pid), pyMaps)
			}
		}
	}

	// Step 9: Run the event loop.
	// trackPID is passed as onFork — called for both fork children and
	// newly-discovered CUDA process PIDs (dynamic host tracer enrollment).
	var loopErr error
	// Collect py_debug_stats maps for periodic reserve-failure warning
	// (Bug 9) and end-of-run counter dump. Do this before the event loop
	// so the periodic ticker can read counters during the trace, not just
	// at the end.
	var pyStatsMaps []*ebpf.Map
	if tracePyWalker == "ebpf" {
		for _, ct := range cudaTracers {
			if m := ct.PyDebugStatsMap(); m != nil {
				pyStatsMaps = append(pyStatsMaps, m)
			}
		}
	}

	loopCfg := &eventLoopConfig{
		Collector:     collector,
		PIDFilter:     pidFilter,
		OnSnapshot:    onSnapshot,
		EventStore:    eventStore,
		Resolver:      resolver,
		PodCache:      podCache,
		SnapFilter:    snapFilter,
		ProcNames:     procNames,
		CUDAPIDs:      cudaPIDs,
		MemTracker:    tracker,
		StragglerDet:  stragglerDetector,
		NodeIdentity:  nodeIdentity,
		RankCache:     rankCache,
		MismatchCheck: mismatchCheck,
		PyMaps:        pyMaps,
		PyStatsMaps:   pyStatsMaps,
	}

	if traceJSON {
		loopErr = runJSONMode(ctx, merged, loopCfg, corrPID, corr, trackPID)
	} else {
		loopErr = runTableMode(ctx, merged, loopCfg, corrPID, droppedFn, droppedDetailFn, corr, trackPID)
	}

	// Debug: dump Python walker per-CPU counters when eBPF walker was active.
	if debugMode && len(pyStatsMaps) > 0 {
		dumpPyDebugStats(pyStatsMaps)
	}
	return loopErr
}

// dumpPyDebugStats reads the per-CPU py_debug_stats counters and logs them.
// The map has 16 uint64 slots; slots we currently use are defined in
// bpf/python_walker.bpf.h. We sum across all CPUs for each slot, then
// aggregate across maps (each cuda tracer has its own stats map).
func dumpPyDebugStats(maps []*ebpf.Map) {
	labels := []string{
		"entered_dispatcher",
		"state_lookup_ok",
		"entered_312",
		"read_interp_ok",
		"read_threads_head_ok",
		"thread_loop_iterations",
		"thread_match_found",
		"frame_loop_first_iteration",
		"depth_gt_zero",
		"read_first_native_tid",
		"entered_311",
		"311_thread_found",
		"311_cframe_read_ok",
		"311_cframe_nonzero",
		"311_interp_frame_read_ok",
		"311_frame_nonzero",
		"311_loop_code_read_ok",
		"311_loop_code_nonzero",
		"single_thread_fallback",
		"read_any_native_tid",
		"walker_310_entered",
		"walker_310_frame_nonzero",
		"walker_310_code_read_ok",
		"walker_310_code_nonzero",
		"walker_310_emitted_frames",
		"scratch_lookup_ok",
		"have_py_set",
		"entered_pyextended_branch",
		"reserved_pyextended",
		"unicode_non_compact_skipped",
	}
	numCPU := runtime.NumCPU()
	fmt.Fprintf(os.Stderr, "  py-walker debug counters (aggregated across %d cuda tracer(s)):\n", len(maps))
	for i := 0; i < len(labels); i++ {
		key := uint32(i)
		var sum uint64
		for _, m := range maps {
			perCPU := make([]uint64, numCPU)
			if err := m.Lookup(&key, &perCPU); err != nil {
				continue
			}
			for _, v := range perCPU {
				sum += v
			}
		}
		fmt.Fprintf(os.Stderr, "    [%d] %-28s %d\n", i, labels[i], sum)
	}
}

// readPyDebugCounter sums a single py_debug_stats slot across all per-CPU
// slices across all cuda tracer stats maps. Returns 0 on any read error.
func readPyDebugCounter(statsMaps []*ebpf.Map, slot uint32) uint64 {
	numCPU := runtime.NumCPU()
	var sum uint64
	for _, m := range statsMaps {
		perCPU := make([]uint64, numCPU)
		if err := m.Lookup(&slot, &perCPU); err != nil {
			continue
		}
		for _, v := range perCPU {
			sum += v
		}
	}
	return sum
}

// checkPyReserveFailures reads the walker's reserve-attempt vs reserve-success
// counters and emits a WARN if >5% of Python extended-record reservations
// failed (indicating ringbuf pressure is silently dropping Python frames).
// Throttled: only fires once per 5s. Must be called from the event-loop
// ticker goroutine (not from BPF callbacks).
func checkPyReserveFailures(statsMaps []*ebpf.Map, lastWarn *time.Time) {
	if len(statsMaps) == 0 {
		return
	}
	now := time.Now()
	if now.Sub(*lastWarn) < 5*time.Second {
		return
	}
	attempted := readPyDebugCounter(statsMaps, 27) // entered_pyextended_branch
	succeeded := readPyDebugCounter(statsMaps, 28) // reserved_pyextended
	if attempted == 0 {
		return
	}
	// Counters are read non-atomically across per-CPU slices, so a CPU
	// incrementing [28] between our two Lookup calls can yield
	// succeeded > attempted. Clamp to avoid nonsense percentages.
	if succeeded > attempted {
		succeeded = attempted
	}
	failed := attempted - succeeded
	failPct := float64(failed) / float64(attempted) * 100
	if failPct > 5 {
		slog.Warn("Python frame records dropped due to ringbuf pressure",
			"reserved", succeeded, "attempted", attempted,
			"failure_pct", fmt.Sprintf("%.0f%%", failPct),
			"hint", "raise --ringbuf-size")
		*lastWarn = now
	}
}

// ---------------------------------------------------------------------------
// Single-instance lock
// ---------------------------------------------------------------------------

const watchdogPinPath = "/sys/fs/bpf/ingero_watchdog"

// acquireTraceLock enforces single-instance per host. Returns an unlock
// function that must be deferred. If another ingero trace is running,
// returns a user-facing error. If a stale lock exists from a SIGKILL'd
// ingero, cleans up orphaned BPF state and proceeds.
func acquireTraceLock() (func(), error) {
	lockPath := "/var/run/ingero-trace.lock"
	if _, err := os.Stat("/var/run"); err != nil {
		lockPath = "/tmp/ingero-trace.lock"
	}

	if data, err := os.ReadFile(lockPath); err == nil {
		pidStr := strings.TrimSpace(string(data))
		if pid, err := strconv.Atoi(pidStr); err == nil && pid > 0 {
			comm, _ := os.ReadFile(fmt.Sprintf("/proc/%d/comm", pid))
			commStr := strings.TrimSpace(string(comm))
			if commStr == "ingero" || strings.HasPrefix(commStr, "ingero") {
				return nil, fmt.Errorf("another ingero trace is running (PID %d) — only one instance per host", pid)
			}
		}
		slog.Info("cleaning up BPF state from previous ingero invocation",
			"lock_path", lockPath,
			"hint", "previous ingero was likely SIGKILL'd or crashed")
		os.Remove(watchdogPinPath)
	}

	if err := os.WriteFile(lockPath, []byte(strconv.Itoa(os.Getpid())), 0o600); err != nil {
		debugf("lock file write failed (non-fatal): %v", err)
	}

	return func() {
		os.Remove(lockPath)
	}, nil
}

// ---------------------------------------------------------------------------
// Adaptive sampling monitor
// ---------------------------------------------------------------------------

// adaptiveSamplingWindowDropsThreshold is the per-5s-window drop count that
// counts as "high pressure". At 5s windows, 1000 drops ≈ 200/sec sustained.
const adaptiveSamplingWindowDropsThreshold = 1000

// adaptiveSamplingMaxRate caps the sampling divisor. Beyond 1-in-100 the
// loss of fidelity outweighs the throughput win for investigations.
const adaptiveSamplingMaxRate uint32 = 100

// nextSamplingRate returns the next sampling rate and updated pressure/quiet
// counters given the current rate and window drop count. Extracted as a pure
// function so the rate-selection logic is unit-testable without a goroutine.
//
// Strategy:
//   - windowDrops > threshold for 2 consecutive windows (≥10s sustained):
//     bump rate 1 → 10 → 100 (capped at adaptiveSamplingMaxRate).
//   - windowDrops == 0 for 6 consecutive windows (≥30s quiet): reset to 1.
//   - Otherwise: hold rate, reset both counters.
//
// Returns (newRate, newHighPressureCount, newQuietCount).
func nextSamplingRate(currentRate uint32, windowDrops uint64, highPressureCount, quietCount int) (uint32, int, int) {
	if windowDrops > adaptiveSamplingWindowDropsThreshold {
		quietCount = 0
		// Already at cap — no point tracking pressure we can't act on.
		if currentRate >= adaptiveSamplingMaxRate {
			return currentRate, highPressureCount, quietCount
		}
		highPressureCount++
		if highPressureCount >= 2 {
			newRate := currentRate * 10
			if newRate < currentRate { // overflow guard
				newRate = adaptiveSamplingMaxRate
			}
			if newRate > adaptiveSamplingMaxRate {
				newRate = adaptiveSamplingMaxRate
			}
			return newRate, 0, 0
		}
		return currentRate, highPressureCount, quietCount
	}
	if windowDrops == 0 {
		quietCount++
		highPressureCount = 0
		if quietCount >= 6 && currentRate > 1 {
			return 1, 0, 0
		}
		return currentRate, highPressureCount, quietCount
	}
	// Mixed: some drops but below threshold. Hold rate, reset counters.
	return currentRate, 0, 0
}

// runAdaptiveSamplingMonitor watches the drop rate and adjusts BPF sampling
// to reduce pressure. Checks every 5 seconds; see nextSamplingRate for the
// rate-change strategy.
func runAdaptiveSamplingMonitor(ctx context.Context, cudaTracers []*cuda.Tracer, driverTracer *driver.Tracer, graphTracers []*cudagraph.Tracer, droppedFn func() uint64) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	var rate uint32 = 1
	var highPressureCount, quietCount int
	var lastDropCount uint64

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			currentDrops := droppedFn()
			windowDrops := currentDrops - lastDropCount
			lastDropCount = currentDrops

			newRate, hp, q := nextSamplingRate(rate, windowDrops, highPressureCount, quietCount)
			highPressureCount = hp
			quietCount = q
			if newRate != rate {
				applyRate(cudaTracers, driverTracer, graphTracers, newRate)
				if newRate > rate {
					slog.Info("adaptive sampling: increased rate due to sustained drops", "old_rate", rate, "new_rate", newRate, "window_drops", windowDrops)
				} else {
					slog.Info("adaptive sampling: reset to 1 (no drops for 30s)", "old_rate", rate)
				}
				rate = newRate
			}
		}
	}
}

// applyRate writes the sampling rate to every attached tracer's BPF
// config_map. Errors are logged via debugf (best-effort — tracers that
// fail to update stay at the previous rate until the next attempt).
func applyRate(cudaTracers []*cuda.Tracer, driverTracer *driver.Tracer, graphTracers []*cudagraph.Tracer, rate uint32) {
	for _, ct := range cudaTracers {
		if err := ct.SetSamplingRate(rate); err != nil {
			debugf("adaptive sampling: cuda rate update failed: %v", err)
		}
	}
	if driverTracer != nil {
		if err := driverTracer.SetSamplingRate(rate); err != nil {
			debugf("adaptive sampling: driver rate update failed: %v", err)
		}
	}
	for _, gt := range graphTracers {
		if err := gt.SetSamplingRate(rate); err != nil {
			debugf("adaptive sampling: graph rate update failed: %v", err)
		}
	}
}

// ---------------------------------------------------------------------------
// Selective storage — store only investigation-valuable events, aggregate rest
// ---------------------------------------------------------------------------

// flushEveryN triggers an inline aggregate flush every N events in the event
// loop. This prevents Go select starvation during high-throughput periods
// (400K+ events/min) where the 1-second ticker may be starved by the event
// channel, causing missed minute-bucket flushes. flushAggregates() is cheap
// (map scan, only writes completed minute-buckets), so inline calls are safe.
const flushEveryN = 10_000

// aggKey identifies a minute-bucket for aggregation.
// Events that shouldStore() rejects are counted here instead of stored individually.
type aggKey struct {
	Bucket int64  // minute-truncated unix nanos
	Source uint8
	Op     uint8
	PID    uint32
}

// aggValue accumulates stats for one aggKey.
type aggValue struct {
	Count   int64
	Stored  int64
	SumDur  int64
	MinDur  int64
	MaxDur  int64
	SumArg0 int64 // sum of arg0 values (e.g., mm_page_alloc bytes)
}

// shouldStore decides whether an individual event should be persisted to SQLite.
// Events that return false are only counted in the aggregate table.
//
// The filtering happens AFTER stats/correlator see the event — only SQLite
// persistence is filtered. This preserves full accuracy for live anomaly
// detection and causal chain analysis.
//
// Decision hierarchy (first match wins):
//  0. mm_page_alloc → aggregate only (always, even during bootstrap/record-all)
//  1. Stack sampling limit reached → skip (unless anomaly)
//  2. --record-all mode → always store
//  3. Bootstrap (first 10s of session) → store (builds baseline in DB)
//  4. Process lifecycle (exec/exit/fork/OOM) → store (rare, high value)
//  5. sched_switch → store if CUDA-active PID (aggregate-only for others)
//  6. Sync ops (StreamSync/DeviceSync/CtxSync) → store (latency symptoms)
//  7. Anomalous (duration > 3x p50) → store (the interesting stuff)
//  8. Everything else → aggregate only (cuLaunchKernel, sched_wakeup, etc.)
func shouldStore(evt events.Event, sessionStart time.Time, recordAll bool,
	collector *stats.Collector, maxStackSamples int, stackSamples map[uint64]int,
	activePIDs map[uint32]bool) bool {

	// mm_page_alloc: always aggregate, never store individually.
	// Must be checked BEFORE bootstrap and --record-all gates.
	// Each event has duration=0 and no stack — zero per-event investigation
	// value. The causal chain engine needs COUNT + SUM(arg0) > 1GB, which it
	// gets from the live event stream (correlator sees ALL events). Stored
	// chains capture the result. The aggregate table's sum_arg0 preserves
	// total bytes for historical queries via run_sql.
	if evt.Source == events.SourceHost && events.HostOp(evt.Op) == events.HostPageAlloc {
		return false
	}

	// Stack sampling: limit events per unique call stack.
	// Even with --record-all, 10K copies of the same stack are redundant.
	// Anomalies bypass the limit (different timing = real investigation value).
	// Host events (sched_switch) have no stacks → unaffected.
	//
	// Uses HashStackSymbols (not HashStackIPs) so that ASLR doesn't defeat
	// dedup: each process maps libraries at different addresses, but the
	// resolved symbol names are identical across PIDs.
	if maxStackSamples > 0 && len(evt.Stack) > 0 {
		h := events.HashStackSymbols(evt.Stack)
		if stackSamples[h] >= maxStackSamples {
			if collector == nil || !collector.IsAnomaly(evt) {
				return false
			}
		}
	}

	if recordAll {
		return true
	}

	// Bootstrap: first 10 seconds — store everything to build DB baseline.
	if time.Since(sessionStart) < 10*time.Second {
		return true
	}

	// Process lifecycle events are always stored (rare, high investigation value).
	if evt.Source == events.SourceHost {
		switch events.HostOp(evt.Op) {
		case events.HostProcessExec, events.HostProcessExit, events.HostProcessFork, events.HostOOMKill:
			return true
		case events.HostSchedSwitch:
			// Store only for tracked PIDs (CUDA/driver activity or fork
			// children). Non-tracked sched_switch (stress-ng, system
			// daemons) is aggregate-only — the chain engine uses
			// counts/durations, not individual rows.
			return len(activePIDs) == 0 || activePIDs[evt.PID]
		case events.HostPodRestart, events.HostPodEviction, events.HostPodOOMKill:
			// K8s lifecycle events are always stored (rare, high value).
			return true
		}
	}

	// Sync ops are always stored (latency symptoms, investigation targets).
	if evt.Source == events.SourceCUDA {
		switch events.CUDAOp(evt.Op) {
		case events.CUDAStreamSync, events.CUDADeviceSync:
			return true
		}
	}
	if evt.Source == events.SourceDriver && events.DriverOp(evt.Op) == events.DriverCtxSync {
		return true
	}

	// TCP retransmits: always store (rare, high investigation value).
	if evt.Source == events.SourceTCP {
		return true
	}

	// Block I/O: store slow operations (>10ms latency, indicates contention).
	if evt.Source == events.SourceIO && evt.Duration > 10*time.Millisecond {
		return true
	}

	// Network: store only anomalous events (high throughput = lots of events).
	// Falls through to the anomaly check below.

	// Anomalous events (duration > 3x p50) are always stored.
	if collector.IsAnomaly(evt) {
		return true
	}

	// Everything else: aggregate only.
	return false
}

// isStackResolved returns true if at least one frame has resolved symbol info.
// Stacks where every frame is unresolved (no symbol, no file, no Python info)
// are garbage from bpf_get_stack() bottom-of-stack artifacts — not worth storing.
func isStackResolved(stack []events.StackFrame) bool {
	for _, f := range stack {
		if f.SymbolName != "" || f.File != "" || f.PyFile != "" || f.PyFunc != "" {
			return true
		}
	}
	return false
}

// truncateMinute truncates a time to the start of its minute as unix nanos.
func truncateMinute(t time.Time) int64 {
	return t.Truncate(time.Minute).UnixNano()
}

// truncateFiveSeconds truncates a time to the start of its 5-second bucket
// as unix nanos. Used for the event_aggregates_5s table that backs the
// health signal collector's sub-minute derivation windows.
func truncateFiveSeconds(t time.Time) int64 {
	return t.Truncate(5 * time.Second).UnixNano()
}

// flushAggregates converts the in-memory aggregate maps into store.Aggregate
// slices and writes them to SQLite. Only flushes buckets older than the
// current bucket for each granularity (completed buckets).
//
// Two tables, two cadences:
//   - event_aggregates (1 min) keeps deep history, pruned by size.
//   - event_aggregates_5s (5 s) feeds the sub-minute health signal window,
//     retained for FiveSecondAggregateRetention.
func flushAggregates(aggs map[aggKey]*aggValue, aggs5s map[aggKey]*aggValue, eventStore *store.Store, now time.Time) {
	if eventStore == nil {
		return
	}

	if len(aggs) > 0 {
		currentBucket := truncateMinute(now)
		batch := drainCompleted(aggs, currentBucket)
		if len(batch) > 0 {
			eventStore.RecordAggregates(batch)
		}
	}
	if len(aggs5s) > 0 {
		currentBucket5s := truncateFiveSeconds(now)
		batch := drainCompleted(aggs5s, currentBucket5s)
		if len(batch) > 0 {
			eventStore.RecordAggregates5s(batch)
		}
	}
}

// drainCompleted pulls out every entry whose Bucket is strictly less than
// currentBucket, returning them as a store.Aggregate batch and removing them
// from the source map.
func drainCompleted(aggs map[aggKey]*aggValue, currentBucket int64) []store.Aggregate {
	var batch []store.Aggregate
	for k, v := range aggs {
		if k.Bucket >= currentBucket {
			continue
		}
		batch = append(batch, store.Aggregate{
			Bucket:  k.Bucket,
			Source:  k.Source,
			Op:      k.Op,
			PID:     k.PID,
			Count:   v.Count,
			Stored:  v.Stored,
			SumDur:  v.SumDur,
			MinDur:  v.MinDur,
			MaxDur:  v.MaxDur,
			SumArg0: v.SumArg0,
		})
		delete(aggs, k)
	}
	return batch
}

// flushAllAggregates flushes ALL buckets from both maps (including in-flight
// ones). Called at shutdown to ensure no data is lost.
func flushAllAggregates(aggs map[aggKey]*aggValue, aggs5s map[aggKey]*aggValue, eventStore *store.Store) {
	if eventStore == nil {
		return
	}

	if len(aggs) > 0 {
		batch := make([]store.Aggregate, 0, len(aggs))
		for k, v := range aggs {
			batch = append(batch, store.Aggregate{
				Bucket: k.Bucket, Source: k.Source, Op: k.Op, PID: k.PID,
				Count: v.Count, Stored: v.Stored, SumDur: v.SumDur,
				MinDur: v.MinDur, MaxDur: v.MaxDur, SumArg0: v.SumArg0,
			})
		}
		eventStore.RecordAggregates(batch)
	}
	if len(aggs5s) > 0 {
		batch := make([]store.Aggregate, 0, len(aggs5s))
		for k, v := range aggs5s {
			batch = append(batch, store.Aggregate{
				Bucket: k.Bucket, Source: k.Source, Op: k.Op, PID: k.PID,
				Count: v.Count, Stored: v.Stored, SumDur: v.SumDur,
				MinDur: v.MinDur, MaxDur: v.MaxDur, SumArg0: v.SumArg0,
			})
		}
		eventStore.RecordAggregates5s(batch)
	}
}

// isArgBytes returns true if arg0 for this (source, op) represents a byte count
// that is meaningful to sum (e.g., allocation size, copy count). Returns false
// for pointer-valued ops where summing would overflow int64 (e.g., cudaFree
// receives a device pointer, cuLaunchKernel receives a function pointer).
func isArgBytes(source events.Source, op uint8) bool {
	switch source {
	case events.SourceCUDA:
		switch events.CUDAOp(op) {
		case events.CUDAMalloc, events.CUDAMallocManaged, events.CUDAMemcpy, events.CUDAMemcpyAsync:
			return true // arg0 = size or count
		}
	case events.SourceDriver:
		switch events.DriverOp(op) {
		case events.DriverMemAlloc, events.DriverMemAllocManaged, events.DriverMemcpy, events.DriverMemcpyAsync:
			return true // arg0 = size
		}
	case events.SourceHost:
		switch events.HostOp(op) {
		case events.HostPageAlloc:
			return true // arg0 = alloc bytes (4096 << order)
		}
	case events.SourceIO:
		return true // arg0 = sector count (block I/O size)
	}
	return false
}

// recordAggregate updates the in-memory aggregates for an event, writing
// both the 1-minute map (aggs) and the 5-second map (aggs5s) in parallel.
// The 'stored' flag indicates whether the event was also written to the
// events table.
func recordAggregate(aggs map[aggKey]*aggValue, aggs5s map[aggKey]*aggValue, evt events.Event, stored bool) {
	durNanos := int64(evt.Duration)
	src := uint8(evt.Source)
	arg0Bytes := isArgBytes(evt.Source, evt.Op)

	updateOne := func(m map[aggKey]*aggValue, bucket int64) {
		key := aggKey{Bucket: bucket, Source: src, Op: evt.Op, PID: evt.PID}
		v, ok := m[key]
		if !ok {
			v = &aggValue{MinDur: durNanos, MaxDur: durNanos}
			m[key] = v
		}
		v.Count++
		v.SumDur += durNanos
		if durNanos < v.MinDur {
			v.MinDur = durNanos
		}
		if durNanos > v.MaxDur {
			v.MaxDur = durNanos
		}
		// Only accumulate arg0 for byte-count ops (malloc size, memcpy
		// count, page alloc bytes). Skip pointer-valued ops to avoid
		// int64 overflow.
		if arg0Bytes {
			v.SumArg0 += int64(evt.Args[0])
		}
		if stored {
			v.Stored++
		}
	}

	updateOne(aggs, truncateMinute(evt.Timestamp))
	updateOne(aggs5s, truncateFiveSeconds(evt.Timestamp))
}

// mergeAllEventChannels creates a single channel that receives events from
// all active tracers (CUDA runtime, host, driver, I/O, TCP, net).
func mergeAllEventChannels(ctx context.Context, cudaCh <-chan events.Event, hostTracer *host.Tracer, driverTracer *driver.Tracer, extraChs ...(<-chan events.Event)) <-chan events.Event {
	// Collect all active channels.
	channels := []<-chan events.Event{cudaCh}
	if hostTracer != nil {
		channels = append(channels, hostTracer.Events())
	}
	if driverTracer != nil {
		channels = append(channels, driverTracer.Events())
	}
	channels = append(channels, extraChs...)

	if len(channels) == 1 {
		return cudaCh
	}

	merged := make(chan events.Event, 8192)

	// Launch one goroutine per source channel that forwards events to merged.
	// Use sync.WaitGroup to safely close merged when all sources complete.
	var wg sync.WaitGroup
	for _, ch := range channels {
		ch := ch // capture loop variable
		wg.Add(1)
		go func() {
			defer wg.Done()
			for evt := range ch {
				select {
				case merged <- evt:
				case <-ctx.Done():
					return
				}
			}
		}()
	}

	// Close merged when all source goroutines complete.
	go func() {
		wg.Wait()
		close(merged)
	}()

	return merged
}

// ---------------------------------------------------------------------------
// Target resolution
// ---------------------------------------------------------------------------

// resolveTargets finds the libcudart.so path(s) and target processes.
//
// Resolution order:
//  1. If --cuda-lib is set: use that path exclusively, skip all discovery
//  2. If --pid is specified: validate each PID in FindCUDAProcesses(), return lib from first
//  3. If auto-detect: return ALL found CUDA processes (not just the first)
//  4. Fallback (no processes): call FindAllLibCUDART() to discover ALL copies
//     of the library (system + venv). Probes fire for ANY process that loads it.
//
// Returns (libPaths, pids, processNames, error).
// libPaths[0] is the "primary" library (used for header display, etc.).
// Empty pids means "all processes".
func resolveTargets(pids []int) ([]string, []int, []string, error) {
	// --cuda-lib: explicit path, skip all discovery.
	if traceCUDALib != "" {
		if _, err := os.Stat(traceCUDALib); err != nil {
			return nil, nil, nil, fmt.Errorf("--cuda-lib path not accessible: %w", err)
		}
		return []string{traceCUDALib}, nil, nil, nil
	}

	if len(pids) > 0 {
		// User specified PID(s) — find their libcudart.so.
		procs, err := discover.FindCUDAProcesses()
		if err != nil {
			return nil, nil, nil, fmt.Errorf("scanning for CUDA processes: %w", err)
		}

		procMap := make(map[int]discover.CUDAProcess)
		for _, p := range procs {
			procMap[p.PID] = p
		}

		var libPath string
		var resolvedPIDs []int
		var names []string
		for _, pid := range pids {
			p, ok := procMap[pid]
			if !ok {
				return nil, nil, nil, fmt.Errorf("PID %d not found or not using CUDA — is it running?", pid)
			}
			if libPath == "" {
				libPath = p.LibCUDAPath
			}
			resolvedPIDs = append(resolvedPIDs, p.PID)
			names = append(names, p.Name)
		}
		return []string{libPath}, resolvedPIDs, names, nil
	}

	// Auto-detect: find ALL CUDA processes.
	procs, err := discover.FindCUDAProcesses()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("scanning for CUDA processes: %w", err)
	}

	if len(procs) > 0 {
		var resolvedPIDs []int
		var names []string
		for _, p := range procs {
			resolvedPIDs = append(resolvedPIDs, p.PID)
			names = append(names, p.Name)
		}
		return []string{procs[0].LibCUDAPath}, resolvedPIDs, names, nil
	}

	// No running CUDA processes — discover ALL copies of libcudart.so.
	// Attach probes to every copy so that venv-bundled libraries are covered.
	libPaths := discover.FindAllLibCUDART()
	if len(libPaths) == 0 {
		return nil, nil, nil, fmt.Errorf(
			"no CUDA processes found and libcudart.so not found.\n"+
				"  Start a GPU workload first, or install CUDA toolkit.\n"+
				"  Run 'ingero check' for detailed diagnostics")
	}

	if len(libPaths) == 1 {
		fmt.Fprintf(os.Stderr, "  No CUDA processes running — attaching to %s\n", libPaths[0])
	} else {
		fmt.Fprintf(os.Stderr, "  No CUDA processes running — attaching to %d libraries:\n", len(libPaths))
		for _, p := range libPaths {
			fmt.Fprintf(os.Stderr, "    %s\n", p)
		}
	}
	fmt.Fprintf(os.Stderr, "  Probes will fire when a CUDA workload starts.\n\n")
	return libPaths, nil, nil, nil
}

// resolveProcessNames looks up names for a list of PIDs via FindCUDAProcesses.
func resolveProcessNames(pids []int) []string {
	procs, err := discover.FindCUDAProcesses()
	if err != nil {
		return make([]string, len(pids))
	}
	procMap := make(map[int]string)
	for _, p := range procs {
		procMap[p.PID] = p.Name
	}
	names := make([]string, len(pids))
	for i, pid := range pids {
		names[i] = procMap[pid]
	}
	return names
}

// ---------------------------------------------------------------------------
// Library mismatch checker
// ---------------------------------------------------------------------------

// libMismatchChecker detects when a CUDA process has loaded a libcudart.so
// that doesn't match any of the libraries ingero attached probes to. This
// happens when a venv bundles its own libcudart.so but ingero only probed the
// system copy (or vice versa). Warns once per PID via slog.Warn.
//
// Safe for concurrent use — Check() is guarded by a mutex.
type libMismatchChecker struct {
	mu           sync.Mutex
	attachedLibs map[string]bool // resolved paths of libraries with probes
	checked      map[uint32]bool // PIDs already checked (warn-once)
}

func newLibMismatchChecker(attachedLibs map[string]bool) *libMismatchChecker {
	return &libMismatchChecker{
		attachedLibs: attachedLibs,
		checked:      make(map[uint32]bool),
	}
}

// Check reads /proc/<pid>/maps on first CUDA event for this PID and logs
// a warning if the loaded library doesn't match any attached library.
func (c *libMismatchChecker) Check(pid uint32) {
	if c == nil {
		return
	}
	c.mu.Lock()
	if c.checked[pid] {
		c.mu.Unlock()
		return
	}
	c.checked[pid] = true
	c.mu.Unlock()

	loadedLib, err := discover.FindCUDAInMaps(int(pid))
	if err != nil || loadedLib == "" {
		return // process gone or no CUDA mapping — skip silently
	}

	// Check if the loaded library matches any attached library.
	if c.attachedLibs[loadedLib] {
		return // exact match
	}

	// Try resolved path comparison for symlink differences.
	resolved, err := filepath.EvalSymlinks(loadedLib)
	if err == nil && c.attachedLibs[resolved] {
		return
	}

	slog.Warn("PID loaded libcudart.so not matching any attached library", "pid", pid, "loaded_lib", loadedLib)
}

// ---------------------------------------------------------------------------
// Python runtime state lifecycle (--py-walker=ebpf)
// ---------------------------------------------------------------------------

// tryPushPyRuntimeState detects whether the given PID is a CPython 3.10,
// 3.11, or 3.12 process and, if so, pushes its _PyRuntime address and
// struct offsets into the BPF py_runtime_map so the in-kernel walker can
// unwind Python frames. The walker dispatches per-version based on the
// PythonMinor field in the pushed state.
//
// Best-effort: any failure (not Python, unsupported version, libpython
// missing, offset out of uint16 range, map write error) is logged at debug
// level and returns nil — the trace continues without per-kernel frame
// walking for this PID. Callers should not treat failures as fatal.
//
// pyMaps empty is a no-op — simplifies callers that conditionally enable the
// ebpf walker without threading an "enabled" bool through every caller. The
// state is written to EVERY map in the slice (one per cuda tracer instance).
func tryPushPyRuntimeState(pid uint32, pyMaps []*ebpf.Map) {
	if len(pyMaps) == 0 || pid == 0 {
		return
	}

	info := symtab.DetectPython(pid)
	if info == nil {
		debugf("py-walker: PID %d not a Python process — skipping", pid)
		return
	}
	// Per-version dispatch lives in bpf/python_walker.bpf.h. Supported
	// minors: 3.9 (legacy PyFrameObject via walker_310 path),
	// 3.10 and 3.11 (dedicated walkers), 3.12/3.13/3.14 (direct
	// _PyInterpreterFrame via walker_312 path). Anything else falls
	// through to the userspace walker.
	if info.Minor < 9 || info.Minor > 14 {
		debugf("py-walker: PID %d is Python %s (python_minor %d) not supported by BPF walker (only 3.9-3.14); userspace walker will handle", pid, info.Version, info.Minor)
		return
	}
	// Free-threaded (PEP 703, Py_GIL_DISABLED) builds add PyMutex fields
	// to PyThreadState that the GIL-build offset tables don't account
	// for, so the walker would emit garbage. Skip these processes
	// entirely. The userspace walker currently only supports GIL builds
	// of 3.10/3.11/3.12 (see PyFrameWalker.IsSupportedVersion), so
	// Python frames will be missing from cuda events on free-threaded
	// 3.13/3.14 processes until either the BPF walker grows a
	// free-threaded offset table or the userspace walker extends
	// support. Marking the PID as "tried" in pyPushedPIDs (via the
	// caller) keeps this a one-shot per-PID decision, not per-event.
	if info.FreeThreaded {
		slog.Info("py-walker: free-threaded Python build detected — BPF walker skipped; Python frames will not be attached (userspace walker does not yet support free-threaded builds)",
			"pid", pid, "python_version", info.Version, "lib_path", info.LibPath)
		return
	}

	runtimeAddr, err := symtab.FindPyRuntimeAddr(pid, info)
	if err != nil || runtimeAddr == 0 {
		// Transient failure: DetectPython succeeded (we know it's Python)
		// but _PyRuntime did not resolve. The most common cause is a
		// race where DetectPython's /proc/<pid>/exe fallback fires
		// before the libpython PT_LOAD is mapped. Clear the dedup
		// entry so a later event retries the push once /proc/maps
		// stabilizes; otherwise the PID stays permanently marked
		// false and the walker never engages.
		debugf("py-walker: _PyRuntime not found for PID %d (transient — retrying on next event): %v", pid, err)
		pyPushedPIDs.Delete(pid)
		return
	}

	// Resolve struct offsets via a layered fallback chain (highest -> lowest
	// confidence). All sources can fail silently on Ubuntu's distro-patched
	// CPython 3.12 builds, so we combine the best signal from each:
	//
	//   1. _Py_DebugOffsets read from the running process memory (3.13+ only;
	//      3.12 has no such struct — the field at _PyRuntime+0 is _initialized,
	//      not a debug-offsets header).
	//   2. Runtime ctypes harvester subprocess. Spawns the SAME python binary
	//      with a tiny script that uses ctypes + known runtime values
	//      (os.gettid, sys._getframe, id()) to scan struct memory and discover
	//      field offsets empirically. Authoritative for the offsets it finds —
	//      same binary as the workload, no debug symbols, no DWARF, immune to
	//      distro patches. Partial coverage; fields it can't discover are
	//      filled by the next source.
	//   3. GetPyOffsetsBest fallback chain (build-id DB → DWARF → hardcoded).
	//
	// Each source overlays the previous: we start with hardcoded/DWARF as a
	// base table, then overlay harvester values where present, then overlay
	// _Py_DebugOffsets where present. The final table is what gets pushed to
	// the BPF map.
	// When ingero runs in a container, info.LibPath (from /proc/PID/maps) is
	// in the target's mount namespace. GetPyOffsetsBest opens the ELF for
	// build-id + DWARF reads, so it needs a path accessible from our own
	// namespace. HarvestOffsets passes info.LibPath as-is and chroots into
	// /proc/<pid>/root/ so the target-namespace path is correct there.
	elfPath := procpath.ResolveContainerPath(int(pid), info.LibPath)

	var offsets *symtab.PyOffsets
	offsets = symtab.GetPyOffsetsBest(elfPath, info.Minor)
	if offsets == nil {
		debugf("py-walker: no fallback offsets available for Python %s (PID %d)", info.Version, pid)
		return
	}

	// Overlay runtime harvester, but only for 3.11. The harvester's
	// frame-walking heuristics were written against 3.12's
	// _PyInterpreterFrame layout; on uv-distributed CPython 3.11.15 /
	// 3.12.13 the pointer-chasing scans match plausibly-valid but wrong
	// offsets (e.g., TstateFrame=240 on 3.12 where the real value is
	// 56, InterpTstateHead=1048 where the real value is 16). The bad
	// overlay then overwrites now-correct hardcoded tables and the
	// walker emits empty py_func strings. 3.10 has distro-invariant
	// offsets and does not need the harvester. 3.12 is covered by the
	// hardcoded table. 3.13+ uses _Py_DebugOffsets. Only 3.11 still
	// benefits from the harvester.
	if info.Minor == 11 {
		if harvested, hErr := symtab.HarvestOffsets(info.LibPath, int(pid)); hErr != nil {
			debugf("py-walker: harvester subprocess failed for PID %d (%s): %v (using fallback offsets)", pid, info.LibPath, hErr)
		} else if harvested != nil {
			offsets = harvested.Overlay(offsets)
			debugf("py-walker: overlaid runtime-harvested offsets onto %s table for PID %d", offsets.Version, pid)
		}
	}

	// CPython 3.12 ALWAYS uses tstate.current_frame directly (no cframe
	// indirection). The harvester's empirical scan can falsely conclude
	// indirection because parent_frame.previous points to its child, mimicking
	// the cframe-current_frame relationship. Force direct access for 3.12 so
	// the dispatcher routes to walker_312 (which validates code_ptr is a heap
	// pointer and skips C-call entry-frame stubs).
	if info.Minor == 12 {
		offsets.CframeCurrentFrame = 0
	}

	// Overlay _Py_DebugOffsets if present (3.13+).
	if info.Minor >= 13 {
		if pyDebugOff, doErr := symtab.ReadDebugOffsetsFromPID(pid, runtimeAddr, info.Minor); doErr == nil && pyDebugOff != nil {
			offsets = pyDebugOff
			debugf("py-walker: using _Py_DebugOffsets from process memory for PID %d", pid)
		}
	}

	state, err := pyRuntimeStateFromOffsets(runtimeAddr, offsets, info.Minor)
	if err != nil {
		debugf("py-walker: converting offsets for PID %d: %v", pid, err)
		return
	}

	// Broadcast to every cuda tracer's py_runtime_map. Each tracer's BPF
	// program only reads from its own map; without broadcast, workload
	// events going through tracer[N] (N>0) would see an empty map.
	var firstErr error
	for i, pyMap := range pyMaps {
		if err := pytrace.SetPyRuntimeState(pyMap, pid, state); err != nil {
			debugf("py-walker: writing py_runtime_map[%d] for PID %d: %v", i, pid, err)
			if firstErr == nil {
				firstErr = err
			}
		}
	}
	if firstErr != nil {
		return
	}
	// Mark this PID as having walker state (bool=true). The dedup map
	// is keyed on PID; value distinguishes "has state" (true) from
	// "tried but not Python / failed" (false). Fork inheritance uses
	// this distinction — only inherit from a parent with true.
	pyPushedPIDs.Store(pid, true)
	if info.Minor != 12 {
		// INFO-level log (once per PID) to surface multi-version support
		// when a non-3.12 Python gets the BPF walker attached.
		slog.Info("py-walker: pushed runtime state for non-3.12 Python process",
			"pid", pid, "python_version", info.Version, "python_minor", info.Minor,
			"runtime_addr", fmt.Sprintf("0x%x", runtimeAddr))
	}
	debugf("py-walker: pushed state for PID %d (_PyRuntime=0x%x, python_minor=%d, offsets=%s, maps=%d) values: RIH=%d ITH=%d TF=%d FB=%d FC=%d CF=%d CN=%d CFL=%d US=%d UD=%d",
		pid, runtimeAddr, info.Minor, offsets.Version, len(pyMaps),
		offsets.RuntimeInterpretersHead, offsets.InterpTstateHead, offsets.TstateFrame,
		offsets.FrameBack, offsets.FrameCode, offsets.CodeFilename, offsets.CodeName,
		offsets.CodeFirstLineNo, offsets.UnicodeState, offsets.UnicodeData)
}

// pyPushedPIDs is a per-PID dedup cache for the walker push pipeline.
//   - Value bool=false: detection was attempted but the PID is not a
//     supported Python process (or push failed). Future callers skip
//     re-running the 150ms harvester for this PID.
//   - Value bool=true: walker state was successfully written to at least
//     one py_runtime_map. Fork inheritance copies state from this parent
//     to its children.
// Cleared on HostProcessExit and HostProcessExec (exec swaps the binary
// so detection must re-run).
var pyPushedPIDs sync.Map

// tryPushPyRuntimeStateOnce wraps tryPushPyRuntimeState with per-PID dedup.
// First call for a PID runs the full detect+harvest+push pipeline
// (~150ms); repeat calls are a cheap map lookup and return.
func tryPushPyRuntimeStateOnce(pid uint32, pyMaps []*ebpf.Map) {
	if len(pyMaps) == 0 || pid == 0 {
		return
	}
	// LoadOrStore marks the PID as "tried" (false). If it was already
	// present (either true or false), we've attempted this PID before
	// and skip. On success, tryPushPyRuntimeState upgrades the value
	// to true via Store.
	if _, loaded := pyPushedPIDs.LoadOrStore(pid, false); loaded {
		return
	}
	tryPushPyRuntimeState(pid, pyMaps)
}

// handlePyLifecycle encapsulates the per-event py_runtime_map lifecycle
// hooks shared between runTableMode and runJSONMode:
//   - HostProcessFork: inherit parent's state into child (no harvester run)
//   - HostProcessExec: clear dedup + re-push (binary may have changed)
//   - HostProcessExit: clear dedup + delete map entry
//
// No-op when pyMaps is empty (walker disabled) or evt is not a host event.
func handlePyLifecycle(evt events.Event, pyMaps []*ebpf.Map) {
	if len(pyMaps) == 0 || evt.Source != events.SourceHost {
		return
	}
	switch events.HostOp(evt.Op) {
	case events.HostProcessFork:
		childPID := uint32(evt.Args[1])
		if childPID == 0 {
			return
		}
		// Only inherit from parents that have actual walker state
		// (bool=true). A parent with false is a non-Python process we
		// already tried — inheriting from it would copy an empty entry
		// AND poison the child's dedup so its own detection never runs.
		v, ok := pyPushedPIDs.Load(evt.PID)
		if !ok {
			return
		}
		parentHasState, _ := v.(bool)
		if !parentHasState {
			return
		}
		for i, pyMap := range pyMaps {
			if err := pytrace.CopyPID(pyMap, evt.PID, childPID); err != nil {
				debugf("py-walker: CopyPID[%d] parent=%d child=%d: %v", i, evt.PID, childPID, err)
			}
		}
		debugf("py-walker: fork-inherited state parent=%d child=%d (%d maps)", evt.PID, childPID, len(pyMaps))
		pyPushedPIDs.Store(childPID, true)
	case events.HostProcessExec:
		clearPyPushedMark(evt.PID)
		tryPushPyRuntimeStateOnce(evt.PID, pyMaps)
	case events.HostProcessExit:
		clearPyPushedMark(evt.PID)
		for i, pyMap := range pyMaps {
			if err := pytrace.ClearPID(pyMap, evt.PID); err != nil {
				debugf("py-walker: ClearPID[%d](%d) failed: %v", i, evt.PID, err)
			}
		}
		// Clear KV-cache lineage state for the dead PID so the
		// per-PID alloc-LRU doesn't leak across long-running agent
		// sessions where many short-lived inference processes have
		// come and gone. Nil-safe when --inference-kvcache-lineage is
		// off.
		if inferEngine != nil {
			inferEngine.OnProcessExit(evt.PID)
		}
	}
}

// clearPyPushedMark removes a PID from the dedup set. Call on process exit
// (PID is gone) and on exec (the PID may now be running a different binary,
// so the next push attempt should re-detect Python version + offsets).
func clearPyPushedMark(pid uint32) {
	pyPushedPIDs.Delete(pid)
}

// pyRuntimeStateFromOffsets converts a symtab.PyOffsets (uint64 fields —
// CPython offsets are always small but typed wide) into the uint16 fields
// required by the BPF py_runtime_state struct. Returns an error if any
// offset exceeds uint16 — this would indicate a corrupt offset table or a
// future CPython layout change that needs wider fields on the BPF side.
//
// `minor` is the CPython minor version (10, 11, or 12). It is stored in
// the PythonMinor field so the BPF dispatcher can select the right
// per-version walker variant.
func pyRuntimeStateFromOffsets(runtimeAddr uint64, o *symtab.PyOffsets, minor int) (pytrace.PyRuntimeState, error) {
	fields := []struct {
		name string
		val  uint64
	}{
		{"RuntimeInterpretersHead", o.RuntimeInterpretersHead},
		{"InterpTstateHead", o.InterpTstateHead},
		{"TstateNext", o.TstateNext},
		{"TstateNativeThreadID", o.TstateNativeThreadID},
		{"TstateFrame", o.TstateFrame},
		{"FrameBack", o.FrameBack},
		{"FrameCode", o.FrameCode},
		{"CodeFilename", o.CodeFilename},
		{"CodeName", o.CodeName},
		{"CodeFirstLineNo", o.CodeFirstLineNo},
		{"UnicodeState", o.UnicodeState},
		{"UnicodeData", o.UnicodeData},
	}
	for _, f := range fields {
		if f.val > 0xFFFF {
			return pytrace.PyRuntimeState{}, fmt.Errorf("offset %s=%d exceeds uint16 range", f.name, f.val)
		}
	}
	if o.CframeCurrentFrame > 0xFFFF {
		return pytrace.PyRuntimeState{}, fmt.Errorf("offset CframeCurrentFrame=%d exceeds uint16 range", o.CframeCurrentFrame)
	}
	if o.InterpNext > 0xFFFF {
		return pytrace.PyRuntimeState{}, fmt.Errorf("offset InterpNext=%d exceeds uint16 range", o.InterpNext)
	}
	return pytrace.PyRuntimeState{
		RuntimeAddr:                runtimeAddr,
		OffRuntimeInterpretersHead: uint16(o.RuntimeInterpretersHead),
		OffTstateHead:              uint16(o.InterpTstateHead),
		OffTstateNext:              uint16(o.TstateNext),
		OffTstateNativeTid:         uint16(o.TstateNativeThreadID),
		OffTstateFrame:             uint16(o.TstateFrame),
		OffFrameBack:               uint16(o.FrameBack),
		OffFrameCode:               uint16(o.FrameCode),
		OffCodeFilename:            uint16(o.CodeFilename),
		OffCodeName:                uint16(o.CodeName),
		OffCodeFirstLineNo:         uint16(o.CodeFirstLineNo),
		OffUnicodeState:            uint16(o.UnicodeState),
		OffUnicodeData:             uint16(o.UnicodeData),
		PythonMinor:                uint8(minor),
		// CframeCurrentFrame is populated only for 3.11 by
		// symtab.GetPyOffsets; 0 for 3.10/3.12 by design.
		OffCframeCurrentFrame: uint16(o.CframeCurrentFrame),
		// InterpNext is PyInterpreterState.next — first struct field
		// on every supported CPython version, so 0 is the correct
		// runtime value, not a sentinel for "disabled".
		OffInterpNext: uint16(o.InterpNext),
	}, nil
}

// ---------------------------------------------------------------------------
// PID → process name cache
// ---------------------------------------------------------------------------

// pidNameCache is a thread-safe in-memory cache of PID → process name.
// Populated from initial discovery and lazily from /proc/[pid]/comm.
type pidNameCache struct {
	mu    sync.RWMutex
	names map[uint32]string
}

func newPIDNameCache(pids []int, processNames []string) *pidNameCache {
	c := &pidNameCache{names: make(map[uint32]string)}
	for i, pid := range pids {
		if i < len(processNames) && processNames[i] != "" {
			c.names[uint32(pid)] = processNames[i]
		}
	}
	return c
}

// Lookup returns the process name for a PID.
// If not cached, reads /proc/[pid]/comm and caches the result.
func (c *pidNameCache) Lookup(pid uint32) string {
	if c == nil {
		return ""
	}
	c.mu.RLock()
	name, ok := c.names[pid]
	c.mu.RUnlock()
	if ok {
		return name
	}

	// Lazy resolve from /proc.
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/comm", pid))
	if err != nil {
		return ""
	}
	name = strings.TrimSpace(string(data))

	c.mu.Lock()
	c.names[pid] = name
	c.mu.Unlock()
	return name
}

// Names returns a snapshot copy of all cached PID→name mappings.
// Used at shutdown to flush discovered names to SQLite.
func (c *pidNameCache) Names() map[uint32]string {
	if c == nil {
		return nil
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make(map[uint32]string, len(c.names))
	for k, v := range c.names {
		out[k] = v
	}
	return out
}

// ---------------------------------------------------------------------------
// Event loop shared config
// ---------------------------------------------------------------------------

// eventLoopConfig bundles the ~16 dependencies shared between runTableMode
// and runJSONMode. Reduces their signatures from 19–23 params each to ~4–8,
// and makes it a one-line change to add another dependency. Callers fill
// the struct once; callees alias fields to local names to keep existing
// body code unchanged.
type eventLoopConfig struct {
	Collector     *stats.Collector
	PIDFilter     map[uint32]bool
	OnSnapshot    func(*stats.Snapshot)
	EventStore    *store.Store
	Resolver      *symtab.Resolver
	PodCache      *k8s.PodCache
	SnapFilter    *filter.SnapshotFilter
	ProcNames     *pidNameCache
	CUDAPIDs      map[uint32]bool
	MemTracker    *memtrack.Tracker
	StragglerDet  *straggler.Detector
	NodeIdentity  string
	RankCache     *discover.RankCache
	MismatchCheck *libMismatchChecker
	PyMaps        []*ebpf.Map
	PyStatsMaps   []*ebpf.Map
}

// ---------------------------------------------------------------------------
// Per-event routing helpers (shared by table + JSON modes)
// ---------------------------------------------------------------------------

// routeInferEvent feeds a single event into the inference engine. Called
// from both runTableMode and runJSONMode after the cgroup / node-identity
// enrichment so the inference engine sees the same input shape regardless
// of the agent's output mode. No-op when inferEngine is nil
// (i.e. --inference is not engaged).
//
// Note on memcpy: events.CUDAMemcpy / CUDAMemcpyAsync carry byte count in
// Args[0], NOT a stream handle, per pkg/events/types.go:550. The infer
// engine's per-stream observable keying needs the stream handle, which the
// BPF probe does not yet emit for memcpy. Until the probe gains that
// field, the classifier runs without memcpy_bytes input - launches +
// NCCL + avg-kernel are sufficient to distinguish prefill / decode in
// practice.
func routeInferEvent(evt events.Event) {
	if inferEngine == nil {
		return
	}
	cgroupHash := ""
	if inferCgroupCache != nil {
		cgroupHash = inferCgroupCache.Resolve(evt.PID)
	}
	if evt.Source == events.SourceCUDA {
		switch events.CUDAOp(evt.Op) {
		case events.CUDAStreamSync, events.CUDADeviceSync:
			inferEngine.OnSyncEvent(evt, cgroupHash)
		case events.CUDALaunchKernel:
			inferEngine.OnLaunchEvent(evt, cgroupHash, evt.Duration)
		case events.CUDAMalloc, events.CUDAMallocManaged:
			// Args[0] is allocation size, Args[1] is the resolved
			// devPtr (the BPF uretprobe reads it from the void**
			// parameter). Older probes that left arg1 as the
			// parameter address rather than the resolved pointer
			// make every alloc key unique to that void**, which is
			// fine - the tracker just gets less useful pairing.
			inferEngine.OnMallocEvent(evt.PID, evt.Args[1], evt.Args[0], evt.Timestamp)
		case events.CUDAFree:
			inferEngine.OnFreeEvent(evt.PID, evt.Args[0])
		}
	} else if evt.Source == events.SourceDriver &&
		events.DriverOp(evt.Op) == events.DriverCtxSync {
		inferEngine.OnSyncEvent(evt, cgroupHash)
	}
}

// feedCorrelatorEvent feeds a single event into the causal-chain
// correlator. Called from both runTableMode and runJSONMode so the
// correlator sees the same input regardless of output mode. No-op when
// corr is nil (correlator disabled).
//
// pidFilter follows the same nil-means-all convention as the trace event
// loops; when non-nil, only PIDs in the map auto-register their cgroup
// for noisy-neighbor detection.
func feedCorrelatorEvent(corr *correlate.Engine, pidFilter map[uint32]bool, evt events.Event) {
	if corr == nil {
		return
	}
	switch evt.Source {
	case events.SourceHost:
		corr.RecordHost(evt)
	case events.SourceIO, events.SourceTCP, events.SourceNet, events.SourceCUDAGraph:
		corr.RecordEvent(evt)
	}
	if evt.CGroupID > 1 && pidFilter != nil && pidFilter[evt.PID] {
		corr.SetTargetCGroup(evt.CGroupID)
	}
}

// drainCorrelatorChains drains the per-tick correlator outputs for a
// PID: AdvanceClock, SnapshotCorrelations, SnapshotCausalChains, the
// inference-engine severity gate update, and the eventStore chain
// persistence. Returns the same correlations + chains the table-mode
// renderer wants; JSON mode discards them. Mirrors the per-tick block
// in runTableMode so JSON daemon mode produces identical chain rows.
//
// Empty / no-op when corr is nil.
func drainCorrelatorChains(corr *correlate.Engine, eventStore *store.Store, ops []stats.OpStats, corrPID uint32, now time.Time) ([]correlate.Correlation, []correlate.CausalChain) {
	if corr == nil {
		return nil, nil
	}
	corr.AdvanceClock(now)
	corrs := corr.SnapshotCorrelations(ops, corrPID)
	chains := corr.SnapshotCausalChains(ops, corrPID)
	if inferEngine != nil {
		inferEngine.OnChainSnapshot(chains, corrPID, now)
	}
	if eventStore != nil {
		if len(chains) > 0 {
			eventStore.RecordChains(chainsToStored(chains))
		} else {
			eventStore.ExpireChains(2 * time.Minute)
		}
	}
	return corrs, chains
}

// drainK8sLifecycleEvents drains the K8s pod-lifecycle queue and
// injects the events as synthetic host events into the collector,
// correlator, and event store. Mirrors the table-mode pod-lifecycle
// drain so JSON daemon mode emits identical synthetic events. No-op
// when podCache is nil.
func drainK8sLifecycleEvents(podCache *k8s.PodCache, collector *stats.Collector, corr *correlate.Engine, eventStore *store.Store) {
	if podCache == nil {
		return
	}
	for _, lce := range podCache.DrainLifecycleEvents() {
		var op events.HostOp
		switch lce.EventType {
		case "restart":
			op = events.HostPodRestart
		case "eviction":
			op = events.HostPodEviction
		case "oom_kill":
			op = events.HostPodOOMKill
		default:
			continue
		}
		syntheticEvt := events.Event{
			Timestamp: lce.DetectedAt,
			Source:    events.SourceHost,
			Op:        uint8(op),
		}
		collector.Record(syntheticEvt)
		if corr != nil {
			corr.RecordHost(syntheticEvt)
		}
		if eventStore != nil {
			eventStore.Record(syntheticEvt)
		}
		debugf("K8s lifecycle: %s/%s %s: %s", lce.Namespace, lce.PodName, lce.EventType, lce.Detail)
	}
}

// drainCGroupSchedStats drains the per-cgroup off-CPU scheduling stats
// and persists them via the event store. Mirrors the table-mode drain
// so JSON daemon mode populates the same noisy-neighbor table.
func drainCGroupSchedStats(corr *correlate.Engine, eventStore *store.Store, now time.Time) {
	if corr == nil || eventStore == nil {
		return
	}
	cgStats := corr.SnapshotCGroupSchedStats()
	if len(cgStats) == 0 {
		return
	}
	storeStats := make([]store.CGroupSchedStat, len(cgStats))
	for i, cs := range cgStats {
		storeStats[i] = store.CGroupSchedStat{
			CGroupID:    cs.CGroupID,
			P99OffCPU:   int64(cs.P99OffCPU),
			TotalOffCPU: int64(cs.TotalOffCPU),
			EventCount:  cs.EventCount,
			WindowStart: cs.WindowStart.UnixNano(),
			WindowEnd:   now.UnixNano(),
		}
	}
	eventStore.RecordCGroupSchedStats(storeStats)
}

// ---------------------------------------------------------------------------
// Table mode — live-updating stats display
// ---------------------------------------------------------------------------

// runTableMode consumes events and refreshes a stats table every second.
// cfg carries the ~16 shared dependencies; corrPID, droppedFn, droppedDetailFn,
// and corr are table-mode-specific.
func runTableMode(ctx context.Context, eventCh <-chan events.Event, cfg *eventLoopConfig, corrPID uint32, droppedFn func() uint64, droppedDetailFn func() string, corr *correlate.Engine, onFork ...func(uint32)) error {
	// Alias config fields to local names — keeps the (large) body unchanged
	// from the pre-config-struct era. Go compiler elides these; no cost.
	collector := cfg.Collector
	pidFilter := cfg.PIDFilter
	onSnapshot := cfg.OnSnapshot
	eventStore := cfg.EventStore
	resolver := cfg.Resolver
	podCache := cfg.PodCache
	snapFilter := cfg.SnapFilter
	procNames := cfg.ProcNames
	cudaPIDs := cfg.CUDAPIDs
	memTracker := cfg.MemTracker
	stragglerDet := cfg.StragglerDet
	nodeIdentity := cfg.NodeIdentity
	rankCache := cfg.RankCache
	mismatchCheck := cfg.MismatchCheck
	pyMaps := cfg.PyMaps
	pyStatsMaps := cfg.PyStatsMaps

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	var lastPyWarn time.Time

	// Track how many lines we printed for cursor-up overwriting.
	linesDrawn := 0

	// Periodic debug throughput counter.
	var debugEventCount uint64
	var storedEventCount uint64
	var debugTickerCh <-chan time.Time
	if debugMode {
		dt := time.NewTicker(10 * time.Second)
		defer dt.Stop()
		debugTickerCh = dt.C
	}

	// Selective storage: aggregate maps for events not individually stored.
	// Two granularities: 1-minute (deep history) and 5-second (sub-minute
	// health signal window).
	sessionStart := time.Now()
	aggs := make(map[aggKey]*aggValue)
	aggs5s := make(map[aggKey]*aggValue)
	var aggFlushCount int // inline flush counter (see flushEveryN)

	// Stack sampling: per-stack event counter to limit DB redundancy.
	stackSamples := make(map[uint64]int)

	// Cgroup cache: cgroup_id → container_id (lazy, populated on first event).
	cgroupCache := make(map[uint64]string)

	// System context collector (CPU/mem/load from /proc).
	sysColl := sysinfo.New()
	sysColl.Start()
	defer sysColl.Stop()

	// Helper to attach system context and get causal chains.
	updateSysCtx := func() {
		if corr == nil {
			return
		}
		sys := sysColl.Snapshot()
		corr.SetSystemSnapshot(&correlate.SystemContext{
			CPUPercent: sys.CPUPercent,
			MemUsedPct: sys.MemUsedPct,
			MemAvailMB: sys.MemAvailMB,
			SwapUsedMB: sys.SwapUsedMB,
			LoadAvg1:   sys.LoadAvg1,
			Timestamp:  sys.Timestamp,
		})
	}

	for {
		select {
		case <-ctx.Done():
			// Flush remaining aggregates at shutdown.
			flushAllAggregates(aggs, aggs5s, eventStore)
			if eventStore != nil {
				totalEvents := collector.Snapshot().TotalEvents
				if totalEvents > 0 {
					debugf("selective storage: %d/%d events stored individually (%.0f%% reduction)",
						storedEventCount, totalEvents,
						float64(uint64(totalEvents)-storedEventCount)/float64(totalEvents)*100)
				}
			}
			updateSysCtx()
			snap := collector.Snapshot()
			attachSysSnapshot(snap, sysColl)
			var corrs []correlate.Correlation
			var chains []correlate.CausalChain
			if corr != nil {
				corr.AdvanceClock(time.Now())
				corrs = corr.SnapshotCorrelations(snap.Ops, corrPID)
				chains = corr.SnapshotCausalChains(snap.Ops, corrPID)
			}
			// v0.16: feed chain severity into the per-workload
			// baseliner so the severity gate pauses updates while a
			// HIGH chain is active for this PID. No-op when
			// --inference is not engaged.
			if inferEngine != nil {
				inferEngine.OnChainSnapshot(chains, uint32(corrPID), time.Now())
			}
			// Store final chains before rendering (ticker stores intermediate
			// chains, but the last snapshot may detect new/upgraded chains).
			if eventStore != nil && len(chains) > 0 {
				eventStore.RecordChains(chainsToStored(chains))
			}
			renderTable(snap, droppedFn(), droppedDetailFn(), &linesDrawn, true, corrs, chains)
			return nil

		case evt, ok := <-eventCh:
			if !ok {
				flushAllAggregates(aggs, aggs5s, eventStore)
				updateSysCtx()
				snap := collector.Snapshot()
				attachSysSnapshot(snap, sysColl)
				var corrs []correlate.Correlation
				var chains []correlate.CausalChain
				if corr != nil {
					corr.AdvanceClock(time.Now())
					corrs = corr.SnapshotCorrelations(snap.Ops, corrPID)
					chains = corr.SnapshotCausalChains(snap.Ops, corrPID)
				}
				// Store final chains (same as ctx.Done path).
				if eventStore != nil && len(chains) > 0 {
					eventStore.RecordChains(chainsToStored(chains))
				}
				renderTable(snap, droppedFn(), droppedDetailFn(), &linesDrawn, true, corrs, chains)
				return nil
			}

			// Track ALL sched_switch events for noisy neighbor detection
			// (pre-PID-filter). Needs peer cgroup data that the PID filter
			// would otherwise drop.
			if corr != nil && evt.Source == events.SourceHost &&
				events.HostOp(evt.Op) == events.HostSchedSwitch {
				corr.RecordCGroupSchedSwitch(evt.CGroupID, evt.Duration)
			}

			// PID filter: nil = accept all, non-nil = only listed PIDs.
			// IO and TCP events are system-wide (kernel tracepoints, not per-process
			// uprobes), so exempt them from PID filtering.
			if pidFilter != nil && !pidFilter[evt.PID] &&
				evt.Source != events.SourceIO && evt.Source != events.SourceTCP {
				continue
			}

			// Resolve cgroup → container ID for K8s container correlation.
			resolveCGroup(&evt, cgroupCache, eventStore, podCache)

			// Set node identity and rank on every event for multi-node correlation.
			evt.Node = nodeIdentity
			ri := rankCache.Lookup(evt.PID)
			evt.Rank = ri.Rank
			evt.LocalRank = ri.LocalRank
			evt.WorldSize = ri.WorldSize

			// Resolve stack symbols if enabled.
			if resolver != nil && len(evt.Stack) > 0 {
				resolver.ResolveStack(&evt)
				if !isStackResolved(evt.Stack) {
					evt.Stack = nil // all garbage, don't store
				}
				debugLogStack(&evt)
			}

			// Stats and correlator see ALL events (full accuracy).
			collector.Record(evt)
			debugEventCount++
			// v0.14 item C: per-direction memcpy aggregator. Cheap
			// (two map writes); only the cudaMemcpy* family of op
			// codes exercises the work path.
			recordMemcpyEvent(evt)

			// v0.16 inference baseliner. Sync events drive the per-
			// workload baseliner; kernel-launch / memcpy / NCCL events
			// also feed the phase classifier so the baseline split
			// works (apples-to-apples comparison against the
			// appropriate phase bucket). All hot-path methods inside
			// the helper short-circuit on non-relevant events.
			routeInferEvent(evt)

			// Memory balance tracker (--remediate): inline consumer, nil when inactive.
			if memTracker != nil {
				memTracker.ProcessEvent(evt)
			}

			// Straggler detector (--remediate): inline consumer, nil when inactive.
			if stragglerDet != nil {
				stragglerDet.ProcessEvent(evt)
			}

			// Selective storage: decide whether to store individually.
			if eventStore != nil {
				stored := shouldStore(evt, sessionStart, traceRecordAll, collector,
					traceStackSamples, stackSamples, cudaPIDs)
				if stored {
					eventStore.Record(evt)
					storedEventCount++
					// Track stack sample count for dedup limiting.
					// Uses HashStackSymbols (ASLR-independent) to match shouldStore().
					if len(evt.Stack) > 0 {
						stackSamples[events.HashStackSymbols(evt.Stack)]++
					}
				}
				recordAggregate(aggs, aggs5s, evt, stored)
				aggFlushCount++
				if aggFlushCount%flushEveryN == 0 {
					flushAggregates(aggs, aggs5s, eventStore, time.Now())
				}
			}

			// Dynamic PID tracking: register non-host event PIDs with
			// the host tracer so it collects host events for those processes.
			// Also track CUDA/Driver PIDs separately for shouldStore() filter.
			if evt.Source != events.SourceHost && len(onFork) > 0 && onFork[0] != nil {
				onFork[0](evt.PID)
			}
			if cudaPIDs != nil && (evt.Source == events.SourceCUDA || evt.Source == events.SourceDriver || evt.Source == events.SourceCUDAGraph) {
				cudaPIDs[evt.PID] = true
			}

			// Runtime library mismatch check: on first CUDA event per PID,
			// verify the process loaded a library we have probes on.
			if mismatchCheck != nil && evt.Source == events.SourceCUDA {
				mismatchCheck.Check(evt.PID)
			}

			// Resolve process name for non-host events (CUDA, Driver, IO, TCP, Net).
			// Host events excluded; sched_switch fires for hundreds of irrelevant
			// system PIDs that would pollute the cache.
			if procNames != nil && evt.Source != events.SourceHost {
				procNames.Lookup(evt.PID)
			}

			// Feed events into correlation engine for causal chain analysis.
			feedCorrelatorEvent(corr, pidFilter, evt)

			// Dynamic PID tracking: when a target process forks, auto-add child
			// to eBPF target_pids (for host event collection). Do NOT inherit
			// cudaPIDs — only actual CUDA/Driver events add PIDs there.
			// This prevents stress-ng workers (forked from Python subprocess)
			// from getting stored sched_switch despite having no CUDA activity.
			if evt.Source == events.SourceHost && events.HostOp(evt.Op) == events.HostProcessFork {
				childPID := uint32(evt.Args[1])
				if childPID > 0 && len(onFork) > 0 && onFork[0] != nil {
					onFork[0](childPID)
				}
				if procNames != nil && childPID > 0 {
					procNames.Lookup(childPID)
				}
			}

			// --py-walker=ebpf lifecycle: fork inherit, exec re-push, exit clear.
			handlePyLifecycle(evt, pyMaps)

		case <-ticker.C:
			updateSysCtx()
			checkPyReserveFailures(pyStatsMaps, &lastPyWarn)
			// Record system snapshot for post-hoc causal chain replay.
			if eventStore != nil {
				sys := sysColl.Snapshot()
				if snapFilter.ShouldEmit(sys.CPUPercent, sys.MemUsedPct, sys.MemAvailMB, sys.SwapUsedMB, sys.LoadAvg1) {
					eventStore.RecordSnapshot(store.SystemSnapshot{
						Timestamp:  sys.Timestamp,
						CPUPercent: sys.CPUPercent,
						MemUsedPct: sys.MemUsedPct,
						MemAvailMB: sys.MemAvailMB,
						SwapUsedMB: sys.SwapUsedMB,
						LoadAvg1:   sys.LoadAvg1,
					})
				}
			}
			// Drain K8s pod lifecycle events and inject as synthetic host events.
			drainK8sLifecycleEvents(podCache, collector, corr, eventStore)

			// Flush completed minute-buckets to SQLite.
			flushAggregates(aggs, aggs5s, eventStore, time.Now())

			// Flush per-cgroup scheduling stats for noisy neighbor detection.
			drainCGroupSchedStats(corr, eventStore, time.Now())

			snap := collector.Snapshot()
			attachSysSnapshot(snap, sysColl)
			if onSnapshot != nil {
				onSnapshot(snap)
			}
			if snap.TotalEvents > 0 || snap.System != nil {
				corrs, chains := drainCorrelatorChains(corr, eventStore, snap.Ops, corrPID, time.Now())
				renderTable(snap, droppedFn(), droppedDetailFn(), &linesDrawn, false, corrs, chains)
			}

		case <-debugTickerCh:
			debugf("throughput: %d events in last 10s (%.0f/sec), total=%d, stored=%d, dropped=%d",
				debugEventCount, float64(debugEventCount)/10.0, collector.Snapshot().TotalEvents, storedEventCount, droppedFn())
			debugEventCount = 0
		}
	}
}

// attachSysSnapshot copies the sysinfo snapshot into the stats snapshot.
func attachSysSnapshot(snap *stats.Snapshot, sysColl *sysinfo.Collector) {
	sys := sysColl.Snapshot()
	snap.System = &stats.SystemSnapshot{
		CPUPercent: sys.CPUPercent,
		MemUsedPct: sys.MemUsedPct,
		MemAvailMB: sys.MemAvailMB,
		MemTotalMB: sys.MemTotalMB,
		SwapUsedMB: sys.SwapUsedMB,
		LoadAvg1:   sys.LoadAvg1,
		LoadAvg5:   sys.LoadAvg5,
		PageFaults: sys.PageFaults,
	}
}

// renderOpsSection writes a titled stats table section for a set of ops.
func renderOpsSection(b *strings.Builder, lines *int, title string, ops []stats.OpStats) {
	// Section title.
	fmt.Fprintf(b, "  %s\033[K\n", title)
	*lines++

	// Table header.
	fmt.Fprintf(b, "  %-20s %8s %10s %10s %10s %10s %7s %8s\033[K\n",
		"OPERATION", "COUNT", "P50", "P95", "P99", "MAX", "WALL%", "SPIKES")
	*lines++

	// Separator.
	fmt.Fprintf(b, "  %s\033[K\n", strings.Repeat("-", 87))
	*lines++

	// One row per operation.
	for _, op := range ops {
		spikeIndicator := ""
		if op.AnomalyCount > 0 {
			spikeIndicator = "!"
		}

		fmt.Fprintf(b, "  %-20s %8d %10s %10s %10s %10s %6.1f%% %7d%s\033[K\n",
			op.Op,
			op.Count,
			formatDuration(op.P50),
			formatDuration(op.P95),
			formatDuration(op.P99),
			formatDuration(op.Max),
			op.TimeFraction*100,
			op.AnomalyCount,
			spikeIndicator,
		)
		*lines++
	}

	if len(ops) == 0 {
		fmt.Fprintf(b, "  (no events yet)\033[K\n")
		*lines++
	}
}

// renderBar renders an ASCII bar chart: [████████░░░░░░░░░░░░] 47%
func renderBar(value, max float64, width int) string {
	if max <= 0 {
		max = 100
	}
	pct := value / max
	if pct > 1 {
		pct = 1
	}
	filled := int(pct * float64(width))
	empty := width - filled

	bar := strings.Repeat("█", filled) + strings.Repeat("░", empty)

	// Flag if abnormal.
	flag := ""
	if value > 90 {
		flag = " [!]"
	}

	return fmt.Sprintf("[%s] %.0f%%%s", bar, value, flag)
}

// renderSystemLine renders the one-line system context with ASCII bars.
func renderSystemLine(b *strings.Builder, lines *int, sys *stats.SystemSnapshot) {
	if sys == nil {
		return
	}

	cpuBar := renderBar(sys.CPUPercent, 100, 20)
	memBar := renderBar(sys.MemUsedPct, 100, 20)

	line := fmt.Sprintf("  System: CPU %s | Mem %s (%d MB free) | Load %.1f",
		cpuBar, memBar, sys.MemAvailMB, sys.LoadAvg1)
	if sys.SwapUsedMB > 0 {
		line += fmt.Sprintf(" | Swap %d MB [!]", sys.SwapUsedMB)
	}
	fmt.Fprintf(b, "%s\033[K\n", line)
	*lines++

	// Separator after system line.
	fmt.Fprintf(b, "\033[K\n")
	*lines++
}

// parseDetailField extracts the numeric value of a `name=N` field from a
// whitespace-separated drop detail string (as produced by droppedDetailFn).
// Returns 0 if the field is absent or unparseable — callers should treat 0
// as "no event drops for this source".
func parseDetailField(detail, name string) uint64 {
	prefix := name + "="
	for _, tok := range strings.Fields(detail) {
		if strings.HasPrefix(tok, prefix) {
			v, err := strconv.ParseUint(tok[len(prefix):], 10, 64)
			if err != nil {
				return 0
			}
			return v
		}
	}
	return 0
}

func renderTable(snap *stats.Snapshot, dropped uint64, droppedDetail string, linesDrawn *int, final bool, correlations []correlate.Correlation, chains ...[]correlate.CausalChain) {
	var b strings.Builder

	// Move cursor up to overwrite previous output.
	if *linesDrawn > 0 {
		fmt.Fprintf(&b, "\033[%dA", *linesDrawn)
	}

	lines := 0

	// Section 1: System Context (one line with ASCII bars).
	renderSystemLine(&b, &lines, snap.System)

	// Split ops by Source for multi-section display.
	var cudaOps, driverOps, hostOps, ioOps, tcpOps, netOps []stats.OpStats
	for _, op := range snap.Ops {
		switch op.Source {
		case events.SourceHost:
			hostOps = append(hostOps, op)
		case events.SourceDriver:
			driverOps = append(driverOps, op)
		case events.SourceIO:
			ioOps = append(ioOps, op)
		case events.SourceTCP:
			tcpOps = append(tcpOps, op)
		case events.SourceNet:
			netOps = append(netOps, op)
		default:
			cudaOps = append(cudaOps, op)
		}
	}

	// Section 2: CUDA Runtime API table.
	renderOpsSection(&b, &lines, "CUDA Runtime API", cudaOps)

	// Section 3: CUDA Driver API table (if any driver events present).
	if len(driverOps) > 0 {
		fmt.Fprintf(&b, "\033[K\n")
		lines++
		renderOpsSection(&b, &lines, "CUDA Driver API", driverOps)
	}

	// Section 4: Host context table (if any host events present).
	if len(hostOps) > 0 {
		fmt.Fprintf(&b, "\033[K\n")
		lines++
		renderOpsSection(&b, &lines, "Host Context", hostOps)
	}

	// Section 5: Block I/O table (if any I/O events present).
	if len(ioOps) > 0 {
		fmt.Fprintf(&b, "\033[K\n")
		lines++
		renderOpsSection(&b, &lines, "Block I/O", ioOps)
	}

	// Section 6: TCP table (if any TCP events present).
	if len(tcpOps) > 0 {
		fmt.Fprintf(&b, "\033[K\n")
		lines++
		renderOpsSection(&b, &lines, "TCP", tcpOps)
	}

	// Section 7: Network socket table (if any net events present).
	if len(netOps) > 0 {
		fmt.Fprintf(&b, "\033[K\n")
		lines++
		renderOpsSection(&b, &lines, "Network Socket", netOps)
	}

	// Empty line.
	fmt.Fprintf(&b, "\033[K\n")
	lines++

	// Summary line.
	summary := fmt.Sprintf("  Wall: %s | Events: %d | Anomalies: %d",
		formatDuration(snap.WallClock), snap.TotalEvents, snap.AnomalyEvents)
	if dropped > 0 {
		summary += fmt.Sprintf(" | Dropped: %d", dropped)
	}
	fmt.Fprintf(&b, "%s\033[K\n", summary)
	lines++

	// Per-tracer drop breakdown (always shown when any drops occurred).
	if dropped > 0 && droppedDetail != "" {
		dropLine := fmt.Sprintf("  Events dropped: %s", droppedDetail)
		// WARN if drops exceed 5% of total events.
		totalEvts := snap.TotalEvents + dropped
		if totalEvts > 0 && float64(dropped)/float64(totalEvts) > 0.05 {
			dropLine += "  WARN: >5% of events dropped -- consider --ringbuf-size"
		}
		// Hard-failure WARN for any critical-event drop — OOM/exec/exit/fork
		// deliveries are guaranteed. Parse host_crit=N out of the detail
		// string (emitted by droppedDetailFn) and flag non-zero values.
		if parseDetailField(droppedDetail, "host_crit") > 0 {
			dropLine += "  WARN: CRITICAL events dropped -- this should never happen"
		}
		fmt.Fprintf(&b, "%s\033[K\n", dropLine)
		lines++
	}

	// Spike patterns.
	for _, op := range snap.Ops {
		if op.SpikePattern != "" {
			fmt.Fprintf(&b, "  Pattern: %s spikes %s\033[K\n", op.Op, op.SpikePattern)
			lines++
		}
	}

	// Correlation annotations (v0.2).
	if len(correlations) > 0 {
		for _, c := range correlations {
			fmt.Fprintf(&b, "  >> %s\033[K\n", c.String())
			lines++
		}
	}

	// Causal chain annotations (v0.3).
	if len(chains) > 0 {
		for _, chainList := range chains {
			for _, ch := range chainList {
				fmt.Fprintf(&b, "  [%s] %s\033[K\n", ch.Severity, ch.Summary)
				lines++
			}
		}
	}

	if final {
		fmt.Fprintf(&b, "\n  Done.\n")
	}

	// Write everything at once (reduces flicker).
	fmt.Fprint(os.Stdout, b.String())
	*linesDrawn = lines
}

// ---------------------------------------------------------------------------
// JSON mode — streaming JSON lines (JSONL)
// ---------------------------------------------------------------------------

// jsonStackFrame is the JSON-serializable stack frame.
type jsonStackFrame struct {
	IP     string `json:"ip,omitempty"`
	Symbol string `json:"symbol,omitempty"`
	File   string `json:"file,omitempty"`
	Line   int    `json:"line,omitempty"`
	PyFile string `json:"py_file,omitempty"`
	PyFunc string `json:"py_func,omitempty"`
	PyLine int    `json:"py_line,omitempty"`
}

type jsonEvent struct {
	Timestamp   string           `json:"timestamp"`
	PID         uint32           `json:"pid"`
	TID         uint32           `json:"tid"`
	ProcessName string           `json:"process_name,omitempty"`
	Source      string           `json:"source"`
	Op          string           `json:"op"`
	DurationNs  int64            `json:"duration_ns"`
	Duration    string           `json:"duration"`
	GPUID       uint32           `json:"gpu_id"`
	Args        [2]uint64        `json:"args"`
	ReturnCode  int32            `json:"return_code"`
	CGroupID    uint64           `json:"cgroup_id,omitempty"`
	Anomaly     bool             `json:"anomaly"`
	Stack       []jsonStackFrame `json:"stack,omitempty"`
}

// runJSONMode streams events as newline-delimited JSON (JSONL).
// cfg carries the ~16 shared dependencies; corrPID + corr drive the
// causal-chain correlator. JSON daemon mode is the production shape
// when --inference auto-sets traceJSON=true, so the correlator feeds +
// chain drains have to mirror table mode for chain-of-evidence
// persistence to work end-to-end. drop-detail reporting is omitted
// (it is a table-UI concern).
func runJSONMode(ctx context.Context, eventCh <-chan events.Event, cfg *eventLoopConfig, corrPID uint32, corr *correlate.Engine, onFork ...func(uint32)) error {
	// Alias config fields to local names — see runTableMode comment.
	collector := cfg.Collector
	pidFilter := cfg.PIDFilter
	onSnapshot := cfg.OnSnapshot
	eventStore := cfg.EventStore
	resolver := cfg.Resolver
	podCache := cfg.PodCache
	snapFilter := cfg.SnapFilter
	procNames := cfg.ProcNames
	cudaPIDs := cfg.CUDAPIDs
	memTracker := cfg.MemTracker
	stragglerDet := cfg.StragglerDet
	nodeIdentity := cfg.NodeIdentity
	rankCache := cfg.RankCache
	mismatchCheck := cfg.MismatchCheck
	pyMaps := cfg.PyMaps
	pyStatsMaps := cfg.PyStatsMaps

	enc := json.NewEncoder(os.Stdout)

	var lastPyWarn time.Time

	// Periodic debug throughput counter (same as runTableMode).
	var debugEventCount uint64
	var storedEventCount uint64
	var debugTickerCh <-chan time.Time
	if debugMode {
		dt := time.NewTicker(10 * time.Second)
		defer dt.Stop()
		debugTickerCh = dt.C
	}

	// Periodic Python-frame reserve-failure check (Bug 9).
	var pyWarnTickerCh <-chan time.Time
	if len(pyStatsMaps) > 0 {
		pwt := time.NewTicker(5 * time.Second)
		defer pwt.Stop()
		pyWarnTickerCh = pwt.C
	}

	// Selective storage: aggregate maps for events not individually stored.
	// Two granularities: 1-minute (deep history) and 5-second (sub-minute
	// health signal window).
	sessionStart := time.Now()
	aggs := make(map[aggKey]*aggValue)
	aggs5s := make(map[aggKey]*aggValue)
	var aggFlushCount int // inline flush counter (see flushEveryN)

	// Stack sampling: per-stack event counter (same as table mode).
	stackSamples := make(map[uint64]int)

	// Cgroup cache: cgroup_id → container_id (lazy, populated on first event).
	cgroupCache := make(map[uint64]string)

	// System context collector — needed for exporter snapshots, for
	// recording system snapshots to SQLite (post-hoc causal chain replay),
	// AND for the unconditional attachSysSnapshot call in the snap-tick
	// below. Initialized always (matches runTableMode); the cost is one
	// /proc-polling goroutine per JSON-mode run, negligible.
	sysColl := sysinfo.New()
	sysColl.Start()
	defer sysColl.Stop()

	// Periodic snapshot for exporters (OTLP, Prometheus) — every 1s, same as table mode.
	snapTicker := time.NewTicker(1 * time.Second)
	defer snapTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			flushAllAggregates(aggs, aggs5s, eventStore)
			if eventStore != nil {
				totalEvents := collector.Snapshot().TotalEvents
				if totalEvents > 0 {
					debugf("selective storage: %d/%d events stored individually (%.0f%% reduction)",
						storedEventCount, totalEvents,
						float64(uint64(totalEvents)-storedEventCount)/float64(totalEvents)*100)
				}
			}
			return nil

		case <-snapTicker.C:
			// Record system snapshot for post-hoc causal chain replay.
			if eventStore != nil {
				sys := sysColl.Snapshot()
				if snapFilter.ShouldEmit(sys.CPUPercent, sys.MemUsedPct, sys.MemAvailMB, sys.SwapUsedMB, sys.LoadAvg1) {
					eventStore.RecordSnapshot(store.SystemSnapshot{
						Timestamp:  sys.Timestamp,
						CPUPercent: sys.CPUPercent,
						MemUsedPct: sys.MemUsedPct,
						MemAvailMB: sys.MemAvailMB,
						SwapUsedMB: sys.SwapUsedMB,
						LoadAvg1:   sys.LoadAvg1,
					})
				}
			}
			// Drain K8s pod lifecycle events and inject as synthetic
			// host events. Mirrors runTableMode so the synthetic
			// events enter the correlator + event store identically.
			drainK8sLifecycleEvents(podCache, collector, corr, eventStore)

			// Flush completed minute-buckets to SQLite.
			flushAggregates(aggs, aggs5s, eventStore, time.Now())

			// Flush per-cgroup scheduling stats for noisy neighbor detection.
			drainCGroupSchedStats(corr, eventStore, time.Now())

			snap := collector.Snapshot()
			attachSysSnapshot(snap, sysColl)
			if onSnapshot != nil {
				onSnapshot(snap)
			}
			// Drain the correlator on every snap-tick so chains
			// land in the event store on the same cadence as table
			// mode. JSON mode discards corrs / chains beyond the
			// chain-of-evidence persistence; renderTable is a TUI
			// concern that does not apply here.
			if snap.TotalEvents > 0 || snap.System != nil {
				_, _ = drainCorrelatorChains(corr, eventStore, snap.Ops, corrPID, time.Now())
			}

		case evt, ok := <-eventCh:
			if !ok {
				flushAllAggregates(aggs, aggs5s, eventStore)
				if eventStore != nil {
					totalEvents := collector.Snapshot().TotalEvents
					if totalEvents > 0 {
						debugf("selective storage: %d/%d events stored individually (%.0f%% reduction)",
							storedEventCount, totalEvents,
							float64(uint64(totalEvents)-storedEventCount)/float64(totalEvents)*100)
					}
				}
				return nil
			}

			// PID filter: nil = accept all, non-nil = only listed PIDs.
			// IO and TCP events are system-wide (kernel tracepoints, not per-process
			// uprobes), so exempt them from PID filtering.
			if pidFilter != nil && !pidFilter[evt.PID] &&
				evt.Source != events.SourceIO && evt.Source != events.SourceTCP {
				continue
			}

			// Resolve cgroup → container ID for K8s container correlation.
			resolveCGroup(&evt, cgroupCache, eventStore, podCache)

			// Set node identity and rank on every event for multi-node correlation.
			evt.Node = nodeIdentity
			ri := rankCache.Lookup(evt.PID)
			evt.Rank = ri.Rank
			evt.LocalRank = ri.LocalRank
			evt.WorldSize = ri.WorldSize

			// Resolve stack symbols if enabled.
			if resolver != nil && len(evt.Stack) > 0 {
				resolver.ResolveStack(&evt)
				if !isStackResolved(evt.Stack) {
					evt.Stack = nil // all garbage, don't store
				}
				debugLogStack(&evt)
			}

			// Stats see ALL events (full accuracy).
			collector.Record(evt)
			debugEventCount++
			// v0.14 item C: per-direction memcpy aggregator.
			recordMemcpyEvent(evt)

			// v0.16 inference baseliner. Mirror of runTableMode's hook
			// via the shared routeInferEvent helper so both modes feed
			// the same engine without per-loop drift.
			routeInferEvent(evt)

			// Memory balance tracker (--remediate): inline consumer, nil when inactive.
			if memTracker != nil {
				memTracker.ProcessEvent(evt)
			}

			// Straggler detector (--remediate): inline consumer, nil when inactive.
			if stragglerDet != nil {
				stragglerDet.ProcessEvent(evt)
			}

			// Selective storage: decide whether to store individually.
			if eventStore != nil {
				stored := shouldStore(evt, sessionStart, traceRecordAll, collector,
					traceStackSamples, stackSamples, cudaPIDs)
				if stored {
					eventStore.Record(evt)
					storedEventCount++
					// Track stack sample count for dedup limiting.
					// Uses HashStackSymbols (ASLR-independent) to match shouldStore().
					if len(evt.Stack) > 0 {
						stackSamples[events.HashStackSymbols(evt.Stack)]++
					}
				}
				recordAggregate(aggs, aggs5s, evt, stored)
				aggFlushCount++
				if aggFlushCount%flushEveryN == 0 {
					flushAggregates(aggs, aggs5s, eventStore, time.Now())
				}
			}

			// Dynamic PID tracking: register non-host event PIDs with
			// the host tracer so it collects host events for those processes.
			// Also track CUDA/Driver PIDs separately for shouldStore() filter.
			if evt.Source != events.SourceHost && len(onFork) > 0 && onFork[0] != nil {
				onFork[0](evt.PID)
			}
			if cudaPIDs != nil && (evt.Source == events.SourceCUDA || evt.Source == events.SourceDriver || evt.Source == events.SourceCUDAGraph) {
				cudaPIDs[evt.PID] = true
			}

			// Runtime library mismatch check: on first CUDA event per PID,
			// verify the process loaded a library we have probes on.
			if mismatchCheck != nil && evt.Source == events.SourceCUDA {
				mismatchCheck.Check(evt.PID)
			}

			// Feed events into correlation engine for causal chain analysis.
			// Mirrors runTableMode so JSON daemon mode produces identical
			// chain rows; without this hook, the chain table stays empty
			// even though the agent is otherwise running.
			feedCorrelatorEvent(corr, pidFilter, evt)

			// Dynamic PID tracking: fork child -> eBPF target_pids only.
			// Do NOT inherit cudaPIDs; only actual CUDA/Driver events add PIDs.
			if evt.Source == events.SourceHost && events.HostOp(evt.Op) == events.HostProcessFork {
				childPID := uint32(evt.Args[1])
				if childPID > 0 && len(onFork) > 0 && onFork[0] != nil {
					onFork[0](childPID)
				}
				if procNames != nil && childPID > 0 {
					procNames.Lookup(childPID)
				}
			}

			// --py-walker=ebpf lifecycle: fork inherit, exec re-push, exit clear.
			handlePyLifecycle(evt, pyMaps)

			je := jsonEvent{
				Timestamp:   evt.Timestamp.Format(time.RFC3339Nano),
				PID:         evt.PID,
				TID:         evt.TID,
				ProcessName: procNames.Lookup(evt.PID),
				Source:      evt.Source.String(),
				Op:          evt.OpName(),
				DurationNs:  evt.Duration.Nanoseconds(),
				Duration:    formatDuration(evt.Duration),
				GPUID:       evt.GPUID,
				Args:        evt.Args,
				ReturnCode:  evt.RetCode,
				CGroupID:    evt.CGroupID,
				Anomaly:     collector.IsAnomaly(evt),
			}

			// Include stack trace if present.
			if len(evt.Stack) > 0 {
				je.Stack = make([]jsonStackFrame, len(evt.Stack))
				for i, f := range evt.Stack {
					sf := jsonStackFrame{
						Symbol: f.SymbolName,
						File:   f.File,
						Line:   f.Line,
						PyFile: f.PyFile,
						PyFunc: f.PyFunc,
						PyLine: f.PyLine,
					}
					// Only emit IP when we have no symbol (partial resolution).
					if f.SymbolName == "" && f.IP != 0 {
						sf.IP = fmt.Sprintf("0x%x", f.IP)
					}
					je.Stack[i] = sf
				}
			}

			if err := enc.Encode(je); err != nil {
				return fmt.Errorf("writing JSON: %w", err)
			}

		case <-debugTickerCh:
			debugf("throughput: %d events in last 10s (%.0f/sec), total=%d",
				debugEventCount, float64(debugEventCount)/10.0, collector.Snapshot().TotalEvents)
			debugEventCount = 0

		case <-pyWarnTickerCh:
			checkPyReserveFailures(pyStatsMaps, &lastPyWarn)
		}
	}
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

// printTraceHeader prints the initial banner with target info.
//
// When --json mode is active, the header is written to STDERR to avoid
// contaminating the JSON stream on stdout. This was a known bug through v0.4
// where the ASCII header mixed with JSONL output, breaking jq, MCP consumers,
// and any tool expecting valid JSONL on stdout.
func printTraceHeader(libPath string, pids []int, processNames []string, cudaProbeCount int, graphProbeCount int, hostProbeCount int, driverProbeCount int, ioProbeCount int, tcpProbeCount int, netProbeCount int, snapFilter *filter.SnapshotFilter) {
	// In JSON mode, redirect header to stderr so stdout is clean JSONL.
	w := os.Stdout
	if traceJSON {
		w = os.Stderr
	}

	fmt.Fprintln(w, "Ingero Trace — Live CUDA Event Stream")
	fmt.Fprintln(w)

	switch len(pids) {
	case 0:
		fmt.Fprintln(w, "  Target: all CUDA processes")
	case 1:
		name := ""
		if len(processNames) > 0 {
			name = processNames[0]
		}
		fmt.Fprintf(w, "  Target: PID %d (%s)\n", pids[0], name)
	default:
		// Multi-PID: "Target: 3 processes [1234 (python3), 5678 (worker), ...]"
		parts := make([]string, len(pids))
		for i, pid := range pids {
			name := ""
			if i < len(processNames) {
				name = processNames[i]
			}
			if name != "" {
				parts[i] = fmt.Sprintf("%d (%s)", pid, name)
			} else {
				parts[i] = fmt.Sprintf("%d", pid)
			}
		}
		fmt.Fprintf(w, "  Target: %d processes [%s]\n", len(pids), strings.Join(parts, ", "))
	}

	fmt.Fprintf(w, "  Library: %s\n", libPath)
	fmt.Fprintf(w, "  CUDA probes: %d attached\n", cudaProbeCount)
	if graphProbeCount > 0 {
		fmt.Fprintf(w, "  Graph probes: %d attached\n", graphProbeCount)
	}
	if driverProbeCount > 0 {
		fmt.Fprintf(w, "  Driver probes: %d attached\n", driverProbeCount)
	}
	if hostProbeCount > 0 {
		fmt.Fprintf(w, "  Host probes: %d attached\n", hostProbeCount)
	}
	if ioProbeCount > 0 {
		fmt.Fprintf(w, "  I/O probes: %d attached\n", ioProbeCount)
	}
	if tcpProbeCount > 0 {
		fmt.Fprintf(w, "  TCP probes: %d attached\n", tcpProbeCount)
	}
	if netProbeCount > 0 {
		fmt.Fprintf(w, "  Net probes: %d attached\n", netProbeCount)
	}

	if !traceRecord {
		fmt.Fprintln(w, "  Recording: disabled (--record=false)")
	} else {
		dbDisplay := traceDBPath
		if dbDisplay == "" {
			dbDisplay = store.DefaultDBPath()
		}
		mode := "selective"
		if traceRecordAll {
			mode = "all events"
		}
		if traceMaxDB != "" && traceMaxDB != "0" {
			fmt.Fprintf(w, "  Recording: %s (%s, max %s)\n", dbDisplay, mode, traceMaxDB)
		} else {
			fmt.Fprintf(w, "  Recording: %s (%s)\n", dbDisplay, mode)
		}
	}
	if traceStack {
		fmt.Fprintln(w, "  Stack traces: enabled")
	}
	if traceOTLP != "" {
		fmt.Fprintf(w, "  OTLP: %s\n", traceOTLP)
	}
	if snapFilter != nil {
		if traceHeartbeat > 0 {
			fmt.Fprintf(w, "  Deadband: %.1f%% threshold, heartbeat %s\n", traceDeadbandPct, traceHeartbeat)
		} else {
			fmt.Fprintf(w, "  Deadband: %.1f%% threshold\n", traceDeadbandPct)
		}
	}

	if traceDuration > 0 {
		fmt.Fprintf(w, "  Duration: %s\n", traceDuration)
	} else {
		fmt.Fprintln(w, "  Duration: until Ctrl+C")
	}

	fmt.Fprintln(w)
}

// resolveCGroup looks up the container ID for an event's cgroup and stores
// the metadata. Uses a cache to avoid repeated /proc reads. CGroupID == 1
// means root cgroup (bare-metal or cgroup v1 only) — skip resolution.
//
// When podCache is non-nil (K8s mode), also enriches with pod name/namespace
// by looking up the container ID in the PodCache.
func resolveCGroup(evt *events.Event, cache map[uint64]string, eventStore *store.Store, podCache *k8s.PodCache) {
	if evt.CGroupID <= 1 {
		return
	}
	if _, cached := cache[evt.CGroupID]; cached {
		return
	}

	// Mark as seen (empty string = resolved but no container ID found).
	cgroupPath, err := cgroup.ReadCGroupPath(evt.PID)
	if err != nil {
		debugf("cgroup: failed to read /proc/%d/cgroup: %v", evt.PID, err)
		cache[evt.CGroupID] = ""
		return
	}

	containerID := cgroup.ParseContainerID(cgroupPath)
	cache[evt.CGroupID] = containerID

	// Enrich with pod name/namespace from K8s API (no-op on bare metal).
	var podName, namespace string
	if podCache != nil && containerID != "" {
		if info := podCache.Lookup(containerID); info != nil {
			podName = info.Name
			namespace = info.Namespace
		}
	}

	if eventStore != nil && (containerID != "" || cgroupPath != "") {
		eventStore.StoreCGroupMetadata(evt.CGroupID, containerID, cgroupPath, podName, namespace)
	}

	if containerID != "" {
		shortID := containerID
		if len(shortID) > 12 {
			shortID = shortID[:12]
		}
		if podName != "" {
			debugf("cgroup: PID %d → cgroup_id=%d container=%s pod=%s/%s", evt.PID, evt.CGroupID, shortID, namespace, podName)
		} else {
			debugf("cgroup: PID %d → cgroup_id=%d container=%s", evt.PID, evt.CGroupID, shortID)
		}
	}
}

// formatDuration formats a duration with 1-2 significant digits per unit tier.
func formatDuration(d time.Duration) string {
	if d == 0 {
		return "0"
	}

	switch {
	case d < time.Microsecond:
		return fmt.Sprintf("%dns", d.Nanoseconds())
	case d < time.Millisecond:
		us := float64(d.Nanoseconds()) / 1000.0
		if us < 10 {
			return fmt.Sprintf("%.1fus", us)
		}
		return fmt.Sprintf("%.0fus", us)
	case d < time.Second:
		ms := float64(d.Nanoseconds()) / 1e6
		if ms < 10 {
			return fmt.Sprintf("%.1fms", ms)
		}
		return fmt.Sprintf("%.0fms", ms)
	case d < time.Minute:
		s := d.Seconds()
		if s < 10 {
			return fmt.Sprintf("%.1fs", s)
		}
		return fmt.Sprintf("%.0fs", s)
	default:
		return d.Truncate(time.Second).String()
	}
}

// debugLogStack logs a resolved stack trace to stderr when --debug is enabled.
// Only logs the first event with stacks (to avoid flooding), then periodically.
var debugStackLogCount uint64

func debugLogStack(evt *events.Event) {
	if !debugMode || len(evt.Stack) == 0 {
		return
	}

	// Log the first 3 events with stacks, then every 1000th.
	debugStackLogCount++
	if debugStackLogCount > 3 && debugStackLogCount%1000 != 0 {
		return
	}

	debugf("stack trace for %s (PID %d, TID %d, %d frames):",
		evt.OpName(), evt.PID, evt.TID, len(evt.Stack))
	for i, f := range evt.Stack {
		if f.PyFile != "" {
			debugf("  [%d] [Python] %s:%d in %s()", i, f.PyFile, f.PyLine, f.PyFunc)
		} else if f.SymbolName != "" {
			debugf("  [%d] %s (%s)", i, f.SymbolName, f.File)
		} else if f.IP != 0 {
			debugf("  [%d] 0x%x (%s)", i, f.IP, f.File)
		}
	}
}

// formatPIDFilter formats target PIDs as a comma-separated string for session metadata.
// Returns "" if no PID filter (tracing all processes).
func formatPIDFilter(pids []int) string {
	if len(pids) == 0 {
		return ""
	}
	parts := make([]string, len(pids))
	for i, p := range pids {
		parts[i] = fmt.Sprintf("%d", p)
	}
	return strings.Join(parts, ",")
}

// formatTraceFlags returns a comma-separated string of active trace flags.
// E.g. "stack,record,json,debug".
func formatTraceFlags() string {
	var flags []string
	if traceStack {
		flags = append(flags, "stack")
	}
	if traceRecord {
		if traceRecordAll {
			flags = append(flags, "record-all")
		} else {
			flags = append(flags, "record")
		}
	}
	if traceJSON {
		flags = append(flags, "json")
	}
	if debugMode {
		flags = append(flags, "debug")
	}
	if traceOTLP != "" {
		flags = append(flags, "otlp")
	}
	if traceProm != "" {
		flags = append(flags, "prometheus")
	}
	if traceMaxDB != "" && traceMaxDB != "0" {
		flags = append(flags, "max-db="+traceMaxDB)
	}
	if tracePyWalker != "" && tracePyWalker != "auto" {
		flags = append(flags, "py-walker="+tracePyWalker)
	}
	return strings.Join(flags, ",")
}

// chainsToStored converts correlate.CausalChain slice to store.StoredChain slice.
func chainsToStored(chains []correlate.CausalChain) []store.StoredChain {
	now := time.Now()
	out := make([]store.StoredChain, len(chains))
	for i, ch := range chains {
		tl := make([]store.TimelineEntry, len(ch.Timeline))
		for j, te := range ch.Timeline {
			tl[j] = store.TimelineEntry{
				Layer:      te.Layer,
				Op:         te.Op,
				Detail:     te.Detail,
				DurationUS: int64(te.Duration / time.Microsecond),
			}
		}
		// Parse tail ratio from summary (format: "op p99=Xms (Y.Zx p50)").
		// Use separate scan variables — Sscanf partial match on OOM chains
		// ("OOM killer...") would overwrite cudaOp with "OOM".
		var scanOp, scanDur string
		var tailRatio float64
		fmt.Sscanf(ch.Summary, "%s p99=%s (%fx", &scanOp, &scanDur, &tailRatio)

		// Extract CUDA op and p99 from timeline (last entry is the symptom).
		var cudaOp string
		var p99us, p50us int64
		if len(ch.Timeline) > 0 {
			last := ch.Timeline[len(ch.Timeline)-1]
			cudaOp = last.Op
			p99us = int64(last.Duration / time.Microsecond)
		}
		if tailRatio > 0 && p99us > 0 {
			p50us = int64(float64(p99us) / tailRatio)
		}

		// Extract node from chain ID if node-namespaced (format: "{node}:{descriptor}").
		chainNode := ""
		if idx := strings.Index(ch.ID, ":"); idx > 0 {
			chainNode = ch.ID[:idx]
		}
		out[i] = store.StoredChain{
			ID:              ch.ID,
			DetectedAt:      now,
			Severity:        ch.Severity,
			Summary:         ch.Summary,
			RootCause:       ch.RootCause,
			Explanation:     ch.Explanation,
			Recommendations: ch.Recommendations,
			CUDAOp:          cudaOp,
			CUDAP99us:       p99us,
			CUDAP50us:       p50us,
			TailRatio:       tailRatio,
			Timeline:        tl,
			Node:            chainNode,
		}
	}
	return out
}

// applyInferenceDefaults flips the flag-var defaults that participate
// in the --inference umbrella. Called once near the top of traceRunE
// AFTER the YAML/CLI resolver has produced resolvedInfer. We mutate
// the package-level flag vars in place so the rest of traceRunE
// (which reads them directly) sees the post-expansion values without
// any further plumbing.
//
// Each default is gated on cmd.Flags().Changed(...) so an explicit
// CLI override always wins. The pattern matches resolveOTLPConfig in
// fleet_push.go.
func applyInferenceDefaults(cmd *cobra.Command, r resolvedInference) error {
	if !r.Enabled {
		return nil
	}

	if !cmd.Flags().Changed("fleet-workload-type") {
		traceWorkloadType = "inference"
	}
	if !cmd.Flags().Changed("duration") {
		// 0 = run until ctx cancellation (the daemon shape).
		traceDuration = 0
	}
	if !cmd.Flags().Changed("json") {
		traceJSON = true
	}
	if !cmd.Flags().Changed("heartbeat") {
		traceHeartbeat = 30 * time.Second
	}
	if !cmd.Flags().Changed("remediate") {
		// FOSS UDS exposure: anyone (including the EE orchestrator)
		// can subscribe. The EE consumer lives in a separate repo.
		traceRemediate = true
	}

	// --max-db vs --db-rollover-size. Resolver already rejected
	// "both explicitly set"; here we only fill the rollover default
	// when neither was explicitly set.
	if !cmd.Flags().Changed("max-db") && !cmd.Flags().Changed("db-rollover-size") {
		// Disable in-place pruning, enable file rollover at 1g.
		traceMaxDB = "0"
		traceDBRolloverSize = "1g"
	}

	// Daemon log path: prefer YAML, then a process-local temp file,
	// only when the operator left --log unset. Keeps stderr clean for
	// systemd / k8s log collectors that consume one-line-per-event.
	if !cmd.Flags().Changed("log") {
		switch {
		case r.DaemonLogPath != "":
			traceLogPath = r.DaemonLogPath
		default:
			traceLogPath = filepath.Join(os.TempDir(),
				fmt.Sprintf("ingero-trace-%d.log", os.Getpid()))
		}
	}

	// Push the resolver's parsed warmup / threshold / pause / sampler
	// fields back onto the flag vars so the engine construction below
	// reads them uniformly from the package vars.
	if r.WarmupSamples > 0 {
		traceInferenceWarmup = r.WarmupSamples
	}
	if r.OutlierThresholdRatio > 0 {
		traceInferenceOutlierRatio = r.OutlierThresholdRatio
	}
	if r.PauseOnSeverity != "" {
		traceInferencePauseSeverity = r.PauseOnSeverity
	}
	if r.SamplerDegradeOn != "" {
		traceInferenceSamplerDegradeOn = r.SamplerDegradeOn
	}
	if r.DBRolloverSize != "" {
		traceDBRolloverSize = r.DBRolloverSize
	}
	if r.DBRolloverKeep > 0 {
		traceDBRolloverKeep = r.DBRolloverKeep
	}
	return nil
}

// configureInferenceEngine constructs the per-workload step-duration
// baseliner + outlier classifier and attaches a sampler to the store.
// No-op when --inference is not engaged. Returns the engine handle so
// the trace event-loop hooks below can route sync events into it.
//
// Caller invokes this AFTER the eventStore is open and SetSampler is
// safe to call. Uses package-level flag vars populated by
// applyInferenceDefaults.
func configureInferenceEngine(eventStore *store.Store) (*infer.Engine, *sampling.Sampler) {
	if !traceInference || traceWorkloadType != "inference" {
		return nil, nil
	}
	smp := sampling.New(
		"inference",
		sampling.DefaultHealthyRate,
		sampling.DefaultCooldownDuration,
	)
	if eventStore != nil {
		eventStore.SetSampler(smp)
	}
	cfg := infer.Config{
		WarmupSamples:         traceInferenceWarmup,
		OutlierThresholdRatio: traceInferenceOutlierRatio,
		PauseOnSeverity:       traceInferencePauseSeverity,
		SamplerDegradeOn:      infer.OutlierBucket(strings.ToLower(strings.TrimSpace(traceInferenceSamplerDegradeOn))),
		Sampler:               smp,
		// v0.16.1: phase-aware baseline split. Default-on (rule
		// classifier) so the umbrella ships robust against
		// heterogeneous-task streams without operator action.
		PhaseClassifierEnabled: strings.ToLower(strings.TrimSpace(traceInferencePhaseClassifier)) != "off",
		PhaseConfig:            buildPhaseConfig(),
		// v0.16.5b: optional kernel-fingerprint dimension.
		FingerprintKeyEnabled: traceInferenceFingerprintKey,
	}
	// Optional KV-cache lineage tracker.
	if traceInferenceKVCacheLineage {
		cfg.KVCacheTracker = kvcache.New(kvcache.Config{
			MaxAllocsPerPID: traceInferenceKVCacheMaxPerPID,
		})
		cfg.KVCacheTopN = traceInferenceKVCacheTopN
	}
	if cfg.SamplerDegradeOn == "off" {
		cfg.SamplerDegradeOn = infer.BucketNone
	}
	eng := infer.New(cfg, slog.Default())
	// Build the PID->cgroup_path_hash resolver here so the event
	// hot-path doesn't need to re-construct it. Default capacity
	// (1024) matches the existing per-cgroup metrics LRU.
	inferCgroupCache = health.NewPIDCGroupHashCache(0)

	// v0.16.3 wiring: throttle reasons. The NVML throttle poller
	// (already running when --otlp / --prometheus is on) writes the
	// OR-folded bitmap into a process-level atomic; the engine reads
	// it on each sync to attach thermal context to outlier events.
	eng.SetThrottleReader(readCurrentThrottleReasons)

	// v0.16.3 wiring: memfrag IOCTL events. The closed-driver kprobe
	// (gated on --enable-experimental-kprobes + the driver/kernel
	// allowlist) calls recordMemfragEvent per event; we install a
	// hook so the same events also feed the inference engine's
	// per-PID observable bucket. The hook is nil-safe and inert when
	// the kprobe didn't load.
	setMemfragInferenceHook(func(ev memfrag.Event) {
		cgroupHash := ""
		if inferCgroupCache != nil {
			cgroupHash = inferCgroupCache.Resolve(ev.PID)
		}
		eng.OnMemfragEvent(ev.PID, cgroupHash, time.Unix(0, int64(ev.TimestampNs)))
	})

	return eng, smp
}

// configureInferenceScraper constructs the periodic engine /metrics
// scraper when --inference is set AND --inference-scrape is "auto".
// The scraper auto-detects vLLM/TGI/SGLang/Triton against the agent's
// target PID set when --pid is explicit, or walks /proc continuously
// when running system-wide (v0.16.4 #10). Engines that boot after the
// agent are picked up on the next re-detection tick.
//
// Returns nil when scraping is disabled. Caller is responsible for
// invoking Run() on a goroutine.
func configureInferenceScraper(targetPIDs []int) *scrape.Scraper {
	if !traceInference {
		return nil
	}
	if strings.EqualFold(strings.TrimSpace(traceInferenceScrape), "off") {
		return nil
	}

	host := traceInferenceScrapeHost
	if host == "" {
		host = "127.0.0.1"
	}

	sink := func(target scrape.Target, samples []scrape.ScrapedSample) {
		// v0.16.2 ships the scrape primitives + sample channel; the
		// OTLP exporter integration that emits these as OTel GenAI
		// metric points is a separate v0.16.x story (the existing
		// internal/export/otlp.go is the natural home but its
		// extension is non-trivial). For now the sink logs at
		// Debug-level so operators with --debug see the canonical
		// names flowing — confirms the scraper is alive.
		debugf("infer scrape: %s %d samples (PID %d, engine %s)",
			target.URL(), len(samples), target.PID, target.Engine)
	}

	// PIDLister: when --pid is explicit, the candidate set is the
	// declared PIDs (re-detection still catches engine swap on a
	// recycled PID). Without --pid, walk /proc to find every running
	// engine on the host so the scraper picks up engines that started
	// after the agent.
	var pidLister scrape.PIDLister
	if len(targetPIDs) > 0 {
		fixed := make([]uint32, 0, len(targetPIDs))
		for _, pid := range targetPIDs {
			if pid > 0 {
				fixed = append(fixed, uint32(pid))
			}
		}
		pidLister = func() []uint32 { return fixed }
	} else {
		pidLister = func() []uint32 { return enginedetect.ListEnginePIDs("") }
	}

	cfg := scrape.Config{
		Interval: traceInferenceScrapeInterval,
		// Timeout = min(interval/2, 5s) so the scraper never blocks
		// past half its own interval.
		Timeout:          capDuration(traceInferenceScrapeInterval/2, 5*time.Second),
		RedetectInterval: traceInferenceScrapeRedetectInterval,
		PIDLister:        pidLister,
	}
	s := scrape.NewScraper(cfg, sink, slog.Default())

	// Eager initial detection so the agent's startup banner reflects
	// engines that are already running. The scraper's own first
	// re-detection tick (fired immediately on Run) reproduces this on
	// the goroutine side, but doing it here too means the operator-
	// visible banner has the right data.
	for _, pid := range targetPIDs {
		if pid <= 0 {
			continue
		}
		det, ok := enginedetect.Detect(uint32(pid))
		if !ok {
			continue
		}
		s.AddTarget(scrape.Target{
			Engine: det.Engine,
			Host:   host,
			Port:   det.Port,
			Path:   det.Engine.MetricsPath(),
			PID:    uint32(pid),
			Model:  det.Model,
		})
		fmt.Fprintf(os.Stderr, "  Inference: detected %s (PID %d, scraping http://%s:%d/metrics)\n",
			det.Engine, pid, host, det.Port)
	}
	return s
}

// capDuration returns the smaller of a and b; a fallback of 1ms is
// used when both are <= 0 to avoid degenerate zero-timeout HTTP
// clients.
func capDuration(a, b time.Duration) time.Duration {
	if a <= 0 && b <= 0 {
		return time.Millisecond
	}
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	if a < b {
		return a
	}
	return b
}

// configureRollover wires the resolved rollover policy into the
// store. No-op when --inference (and therefore the rollover defaults)
// is not engaged AND the operator did not explicitly set
// --db-rollover-size.
func configureRollover(eventStore *store.Store) error {
	if eventStore == nil || strings.TrimSpace(traceDBRolloverSize) == "" {
		return nil
	}
	maxBytes, err := store.ParseSize(traceDBRolloverSize)
	if err != nil {
		return fmt.Errorf("parsing --db-rollover-size %q: %w", traceDBRolloverSize, err)
	}
	keep := traceDBRolloverKeep
	if keep <= 0 {
		keep = 6
	}
	eventStore.SetRolloverConfig(store.RolloverConfig{
		MaxSize:   maxBytes,
		KeepFiles: keep,
	})
	return nil
}

// emitInferOutlier publishes one outlier event through every channel
// the operator has wired in: OTLP histogram + counter, Prometheus
// counters, and the FOSS UDS socket (when --remediate is on). Drops
// silently when the matching channel is not configured. Called from
// the snapshot-tick callback in onSnapshot.
func emitInferOutlier(udsServer *remediate.Server, nodeID, clusterID string, ev infer.OutlierEvent) {
	// UDS publish (FOSS-side only — consumers, including the EE
	// orchestrator, react however they choose). When the socket is
	// not enabled, udsServer is nil and we skip.
	//
	// Send-return is intentionally discarded: per-send drops are
	// counted on the server side via bumpDropLocked and surfaced
	// through DroppedByReason() so observability lives in one place
	// rather than at every emitter. Mirrors the established
	// fire-and-forget convention at internal/straggler/detector.go.
	if udsServer != nil {
		_ = udsServer.SendInferenceOutlier(remediate.InferenceOutlier{
			Timestamp:             ev.At,
			NodeID:                nodeID,
			ClusterID:             clusterID,
			EventID:               ev.EventID,
			CGroupPathHash:        ev.Key.CGroupHash,
			PID:                   ev.Key.PID,
			StreamHandle:          ev.Key.StreamHandle,
			Phase:                 string(ev.Key.Phase),
			StepDurationNs:        ev.StepDurationNs,
			BaselineP95Ns:         ev.BaselineP95Ns,
			BaselineMeanNs:        ev.BaselineMeanNs,
			Bucket:                string(ev.Bucket),
			MemfragEventsInStep:   ev.MemfragEvents,
			ThrottleReasons:       ev.ThrottleReasons,
			MinSMClockMHz:         ev.MinSMClockMHz,
			KVCacheTopAllocAgesMs: ev.KVCacheTopAllocAgesMs,
		})
	}
	// OTLP histogram + counter emission goes through the existing
	// stats / export pipelines on the next snapshot tick. The infer
	// engine's SnapshotForExport is read by the snapshot builder
	// below; no per-outlier OTLP push is needed here.
}

// buildOutlierSpans converts a drained slice of OutlierEvent into the
// stats.OutlierSpan wire shape consumed by the OTLP /v1/traces emit
// path. Walks the scraper (when available) once per outlier to
// enrich with model + engine identity so spans carry the same
// gen_ai.* attributes the metric path emits.
func buildOutlierSpans(outliers []infer.OutlierEvent, scraper *scrape.Scraper) []stats.OutlierSpan {
	if len(outliers) == 0 {
		return nil
	}
	out := make([]stats.OutlierSpan, 0, len(outliers))
	for _, oe := range outliers {
		var model, engine string
		if scraper != nil {
			model = scraper.LookupModel(oe.Key.PID)
			engine = scraper.LookupEngine(oe.Key.PID)
		}
		out = append(out, stats.OutlierSpan{
			EventID:               oe.EventID,
			Bucket:                string(oe.Bucket),
			StepStart:             oe.At.Add(-time.Duration(oe.StepDurationNs)),
			StepEnd:               oe.At,
			StepDurationNs:        oe.StepDurationNs,
			BaselineP95Ns:         oe.BaselineP95Ns,
			BaselineMeanNs:        oe.BaselineMeanNs,
			CGroupHash:            oe.Key.CGroupHash,
			PID:                   oe.Key.PID,
			StreamHandle:          oe.Key.StreamHandle,
			Phase:                 string(oe.Key.Phase),
			KernelFingerprint:     oe.Key.KernelFingerprint,
			MemfragEvents:         oe.MemfragEvents,
			ThrottleReasons:       oe.ThrottleReasons,
			MinSMClockMHz:         oe.MinSMClockMHz,
			KVCacheTopAllocAgesMs: oe.KVCacheTopAllocAgesMs,
			ModelName:             model,
			EngineSystem:          engine,
		})
	}
	return out
}

// emitInferSamplerDegraded publishes one sampler-degraded edge event
// to the UDS socket. v0.16.3 sibling of emitInferOutlier; OTLP /
// Prometheus emission goes through the snapshot path via
// InferSamplerState gauges + Sum. Same fire-and-forget UDS-send
// convention as emitInferOutlier; drops are visible via the server's
// DroppedByReason counter rather than per-emitter logging.
func emitInferSamplerDegraded(udsServer *remediate.Server, nodeID, clusterID string, ev infer.SamplerDegradedEvent) {
	if udsServer == nil {
		return
	}
	_ = udsServer.SendInferenceSamplerDegraded(remediate.InferenceSamplerDegraded{
		Timestamp:      ev.At,
		NodeID:         nodeID,
		ClusterID:      clusterID,
		CGroupPathHash: ev.Key.CGroupHash,
		PID:            ev.Key.PID,
		StreamHandle:   ev.Key.StreamHandle,
		Phase:          string(ev.Key.Phase),
		Bucket:         string(ev.Bucket),
		Cause:          ev.Cause,
		CooldownEnd:    ev.CooldownEnd,
	})
}

// buildPhaseConfig assembles the infer.PhaseConfig from the v0.16.1
// flag values. Human-friendly size strings (1m, 10m) are parsed via
// the existing store.ParseSize helper so the syntax matches --max-db
// and --db-rollover-size. Parse failures fall through to the
// PhaseConfig.Resolved defaults — operator typos do not crash the
// agent's startup, the phase classifier just runs with safer
// thresholds and an INFO log alerts the operator.
func buildPhaseConfig() infer.PhaseConfig {
	cfg := infer.PhaseConfig{
		DecodeMaxLaunches:    traceInferencePhaseDecodeMaxLaunch,
		PrefillMinLaunches:   traceInferencePhasePrefillMinLaunch,
		PrefillMinAvgKernel:  traceInferencePhasePrefillMinAvgKern,
		MixedLaunchLow:       traceInferencePhaseMixedLaunchLow,
		MixedLaunchHigh:      traceInferencePhaseMixedLaunchHigh,
		MemfragDecodeMin:     traceInferencePhaseMemfragDecodeMin,
	}
	if traceInferencePhaseDecodeMaxMemcpy != "" {
		if n, err := store.ParseSize(traceInferencePhaseDecodeMaxMemcpy); err == nil {
			cfg.DecodeMaxMemcpy = n
		} else {
			slog.Default().Warn("infer: --inference-phase-decode-max-memcpy parse failed, using default",
				"value", traceInferencePhaseDecodeMaxMemcpy, "err", err.Error())
		}
	}
	if traceInferencePhaseMixedMemcpy != "" {
		if n, err := store.ParseSize(traceInferencePhaseMixedMemcpy); err == nil {
			cfg.MixedMemcpyThreshold = n
		} else {
			slog.Default().Warn("infer: --inference-phase-mixed-memcpy parse failed, using default",
				"value", traceInferencePhaseMixedMemcpy, "err", err.Error())
		}
	}
	return cfg
}
