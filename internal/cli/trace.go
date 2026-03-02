package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/ingero-io/ingero/internal/cgroup"
	"github.com/ingero-io/ingero/internal/correlate"
	"github.com/ingero-io/ingero/internal/discover"
	"github.com/ingero-io/ingero/internal/ebpf/blockio"
	"github.com/ingero-io/ingero/internal/ebpf/cuda"
	"github.com/ingero-io/ingero/internal/ebpf/driver"
	"github.com/ingero-io/ingero/internal/ebpf/host"
	nettracer "github.com/ingero-io/ingero/internal/ebpf/net"
	"github.com/ingero-io/ingero/internal/ebpf/tcp"
	"github.com/ingero-io/ingero/internal/export"
	"github.com/ingero-io/ingero/internal/filter"
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
)

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
	traceCmd.Flags().StringVar(&traceOTLP, "otlp", "", "OTLP endpoint for metric export (e.g., localhost:4317). Disabled by default.")
	traceCmd.Flags().StringVar(&traceProm, "prometheus", "", "Prometheus /metrics listen address (e.g., :9090). Disabled by default.")
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

	rootCmd.AddCommand(traceCmd)
}

// ---------------------------------------------------------------------------
// Main trace logic
// ---------------------------------------------------------------------------

func traceRunE(cmd *cobra.Command, args []string) error {
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
	libPath, targetPIDs, processNames, err := resolveTargets(tracePIDs)
	if err != nil {
		return err
	}
	debugf("targets resolved: lib=%s pids=%v names=%v", libPath, targetPIDs, processNames)

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

	// Step 2: Create CUDA tracer and attach eBPF uprobes.
	var cudaOpts []cuda.Option
	if traceStack {
		cudaOpts = append(cudaOpts, cuda.WithStackCapture(true))
	}
	cudaTracer := cuda.New(libPath, cudaOpts...)
	if err := cudaTracer.Attach(); err != nil {
		return fmt.Errorf("attaching CUDA probes: %w", err)
	}
	defer cudaTracer.Close()
	debugf("CUDA tracer: %d probes attached to %s", cudaTracer.ProbeCount(), libPath)

	// Step 3: Create host tracer (non-fatal — graceful degradation).
	// Always attach host tracepoints. When no PIDs are targeted, the BPF
	// target_pids map starts empty — PIDs are added dynamically as CUDA
	// events arrive. This ensures host correlation works even without --pid.
	var hostTracer *host.Tracer
	hostProbeCount := 0
	{
		// Seed with first PID (host.New takes one initial PID, 0 = none).
		initialPID := singlePIDOrZero(targetPIDs)
		ht := host.New(uint32(initialPID))
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
			// Seed target PIDs into net PID filter.
			for _, pid := range targetPIDs {
				if pid > 0 {
					nt.SetTargetPID(uint32(pid))
				}
			}
		}
	}
	if netTracer != nil {
		defer netTracer.Close()
	}

	// Step 4: Create stats collector and correlation engine.
	collector := stats.New()
	corr := correlate.New()

	// Step 4b: Create OTEL exporters (disabled by default).
	var otlpExporter *export.OTLPExporter
	if traceOTLP != "" {
		otlpExporter = export.NewOTLP(export.OTLPConfig{
			Endpoint: traceOTLP,
			Insecure: true, // Default to insecure for localhost development.
			DebugLog: debugf,
		})
		debugf("OTLP: configured endpoint=%s", traceOTLP)
	}

	var promSrv *export.PrometheusServer
	if traceProm != "" {
		promSrv = export.NewPrometheus(traceProm)
	}

	// Step 4c: Open SQLite store (recording is on by default).
	if !traceRecord && traceRecordAll {
		fmt.Fprintf(os.Stderr, "  Warning: --record-all ignored because --record=false\n")
	}
	var eventStore *store.Store
	var sessionID int64
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
		debugf("recording to %s", dbPath)

		// Set size-based DB limit if --max-db is specified.
		if traceMaxDB != "" && traceMaxDB != "0" {
			maxBytes, err := store.ParseSize(traceMaxDB)
			if err != nil {
				s.Close() // close DB before returning — defer isn't registered yet
				return fmt.Errorf("parsing --max-db %q: %w", traceMaxDB, err)
			}
			eventStore.SetMaxDBSize(maxBytes)
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
			if sessionID > 0 {
				if err := eventStore.StopSession(sessionID, time.Now()); err != nil {
					debugf("failed to stop session: %v", err)
				}
			}
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

	// Step 6: Print header.
	printTraceHeader(libPath, targetPIDs, processNames, cudaTracer.ProbeCount(), hostProbeCount, driverProbeCount, ioProbeCount, tcpProbeCount, netProbeCount, snapFilter)

	// Step 7: Launch tracers and merge event channels.
	go cudaTracer.Run(ctx)
	if hostTracer != nil {
		go hostTracer.Run(ctx)
	}
	if driverTracer != nil {
		go driverTracer.Run(ctx)
	}

	// Launch new tracers (I/O, TCP, net) — each feeds into the merged channel.
	var extraChs [](<-chan events.Event)
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

	// Start SQLite writer if recording.
	if eventStore != nil {
		go eventStore.Run(ctx)
	}

	// Fan-in: merge all event channels into one.
	merged := mergeAllEventChannels(ctx, cudaTracer.Events(), hostTracer, driverTracer, extraChs...)

	// Start Prometheus server if configured.
	if promSrv != nil {
		go promSrv.Start(ctx)
	}

	// Start OTLP exporter if configured.
	if otlpExporter != nil {
		go otlpExporter.Start(ctx)
	}

	// Snapshot callback for exporters (OTLP, Prometheus).
	// Called every 1s from the table/JSON mode tickers.
	// OTLP push is rate-limited: only every ExportInterval seconds (default 10s).
	// Only created when at least one exporter is configured, to avoid spinning
	// a sysinfo.Collector goroutine (reading /proc every second) for nothing.
	var onSnapshot func(*stats.Snapshot)
	if promSrv != nil || otlpExporter != nil {
		var otlpPushCount int
		onSnapshot = func(snap *stats.Snapshot) {
			if promSrv != nil {
				promSrv.UpdateSnapshot(snap)
			}
			if otlpExporter != nil {
				otlpPushCount++
				if otlpPushCount%otlpExporter.Interval() == 0 {
					if err := otlpExporter.Push(snap); err != nil {
						debugf("OTLP: push error: %v", err)
					}
				}
			}
		}
	}

	// Combined dropped count from all tracers.
	droppedFn := func() uint64 {
		d := cudaTracer.Dropped()
		if hostTracer != nil {
			d += hostTracer.Dropped()
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

	// Dynamic PID tracking: when tracing all processes (no --pid), we add
	// PIDs to the host tracer's BPF map as we discover them from CUDA events.
	// This ensures host correlation works without --pid.
	trackedPIDs := make(map[uint32]bool)
	for _, pid := range targetPIDs {
		if pid > 0 {
			trackedPIDs[uint32(pid)] = true
		}
	}

	// Correlator PID: single PID for single-process, 0 for multi/all (aggregate).
	corrPID := singlePIDOrZero(targetPIDs)
	trackPID := func(pid uint32) {
		if hostTracer == nil || pid == 0 {
			return
		}
		if !trackedPIDs[pid] {
			trackedPIDs[pid] = true
			hostTracer.SetTargetPID(pid)
			debugf("host tracer: dynamically added PID %d", pid)
		}
	}

	// Step 8: Build PID→name cache for JSON output enrichment.
	procNames := newPIDNameCache(targetPIDs, processNames)

	// Step 9: Run the event loop.
	// trackPID is passed as onFork — called for both fork children and
	// newly-discovered CUDA process PIDs (dynamic host tracer enrollment).
	if traceJSON {
		return runJSONMode(ctx, merged, collector, pidFilter, eventStore, resolver, onSnapshot, podCache, snapFilter, procNames, trackPID)
	}
	return runTableMode(ctx, merged, collector, corrPID, pidFilter, droppedFn, onSnapshot, eventStore, corr, resolver, podCache, snapFilter, trackPID)
}

// ---------------------------------------------------------------------------
// Selective storage — store only investigation-valuable events, aggregate rest
// ---------------------------------------------------------------------------

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
//  5. sched_switch → store (causal chain critical)
//  6. Sync ops (StreamSync/DeviceSync/CtxSync) → store (latency symptoms)
//  7. Anomalous (duration > 3x p50) → store (the interesting stuff)
//  8. Everything else → aggregate only (cuLaunchKernel, sched_wakeup, etc.)
func shouldStore(evt events.Event, sessionStart time.Time, recordAll bool,
	collector *stats.Collector, maxStackSamples int, stackSamples map[uint64]int) bool {

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
			// sched_switch is causal chain critical — always store.
			return true
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

// flushAggregates converts the in-memory aggregate map into store.Aggregate
// slice and writes them to SQLite. Only flushes buckets older than the current
// minute (completed buckets).
func flushAggregates(aggs map[aggKey]*aggValue, eventStore *store.Store, now time.Time) {
	if eventStore == nil || len(aggs) == 0 {
		return
	}

	currentBucket := truncateMinute(now)
	var batch []store.Aggregate

	for k, v := range aggs {
		// Only flush completed minute-buckets (not the current one).
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

	if len(batch) > 0 {
		eventStore.RecordAggregates(batch)
	}
}

// flushAllAggregates flushes ALL aggregate buckets (including current minute).
// Called at shutdown to ensure no data is lost.
func flushAllAggregates(aggs map[aggKey]*aggValue, eventStore *store.Store) {
	if eventStore == nil || len(aggs) == 0 {
		return
	}

	batch := make([]store.Aggregate, 0, len(aggs))
	for k, v := range aggs {
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
	}

	eventStore.RecordAggregates(batch)
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

// recordAggregate updates the in-memory aggregate for an event.
// Called for every event regardless of whether it's individually stored.
// The 'stored' flag indicates whether the event was also written to the events table.
func recordAggregate(aggs map[aggKey]*aggValue, evt events.Event, stored bool) {
	key := aggKey{
		Bucket: truncateMinute(evt.Timestamp),
		Source: uint8(evt.Source),
		Op:     evt.Op,
		PID:    evt.PID,
	}

	durNanos := int64(evt.Duration)

	v, ok := aggs[key]
	if !ok {
		v = &aggValue{
			MinDur: durNanos,
			MaxDur: durNanos,
		}
		aggs[key] = v
	}

	v.Count++
	v.SumDur += durNanos
	if durNanos < v.MinDur {
		v.MinDur = durNanos
	}
	if durNanos > v.MaxDur {
		v.MaxDur = durNanos
	}
	// Only accumulate arg0 for byte-count ops (malloc size, memcpy count,
	// page alloc bytes). Skip pointer-valued ops to avoid int64 overflow.
	if isArgBytes(evt.Source, evt.Op) {
		v.SumArg0 += int64(evt.Args[0])
	}
	if stored {
		v.Stored++
	}
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

// resolveTargets finds the libcudart.so path and target processes.
//
// Resolution order:
//  1. If --pid is specified: validate each PID in FindCUDAProcesses(), return lib from first
//  2. If auto-detect: return ALL found CUDA processes (not just the first)
//  3. Fallback: search filesystem for libcudart.so (attach to library,
//     probes fire for ANY process that loads it)
//
// Returns (libPath, pids, processNames, error). Empty pids means "all processes".
func resolveTargets(pids []int) (string, []int, []string, error) {
	if len(pids) > 0 {
		// User specified PID(s) — find their libcudart.so.
		procs, err := discover.FindCUDAProcesses()
		if err != nil {
			return "", nil, nil, fmt.Errorf("scanning for CUDA processes: %w", err)
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
				return "", nil, nil, fmt.Errorf("PID %d not found or not using CUDA — is it running?", pid)
			}
			if libPath == "" {
				libPath = p.LibCUDAPath
			}
			resolvedPIDs = append(resolvedPIDs, p.PID)
			names = append(names, p.Name)
		}
		return libPath, resolvedPIDs, names, nil
	}

	// Auto-detect: find ALL CUDA processes.
	procs, err := discover.FindCUDAProcesses()
	if err != nil {
		return "", nil, nil, fmt.Errorf("scanning for CUDA processes: %w", err)
	}

	if len(procs) > 0 {
		var resolvedPIDs []int
		var names []string
		for _, p := range procs {
			resolvedPIDs = append(resolvedPIDs, p.PID)
			names = append(names, p.Name)
		}
		return procs[0].LibCUDAPath, resolvedPIDs, names, nil
	}

	// No running CUDA processes — attach to library on disk.
	// Probes fire when any process later loads it.
	libPath, err := discover.FindLibCUDART()
	if err != nil {
		return "", nil, nil, fmt.Errorf(
			"no CUDA processes found and libcudart.so not found.\n"+
				"  Start a GPU workload first, or install CUDA toolkit.\n"+
				"  Run 'ingero check' for detailed diagnostics")
	}

	fmt.Fprintf(os.Stderr, "  No CUDA processes running — attaching to %s\n", libPath)
	fmt.Fprintf(os.Stderr, "  Probes will fire when a CUDA workload starts.\n\n")
	return libPath, nil, nil, nil
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

// ---------------------------------------------------------------------------
// Table mode — live-updating stats display
// ---------------------------------------------------------------------------

// runTableMode consumes events and refreshes a stats table every second.
// corrPID is the correlator PID (single PID or 0 for aggregate).
// pidFilter is the event-loop filter (nil = accept all).
func runTableMode(ctx context.Context, eventCh <-chan events.Event, collector *stats.Collector, corrPID uint32, pidFilter map[uint32]bool, droppedFn func() uint64, onSnapshot func(*stats.Snapshot), eventStore *store.Store, corr *correlate.Engine, resolver *symtab.Resolver, podCache *k8s.PodCache, snapFilter *filter.SnapshotFilter, onFork ...func(uint32)) error {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

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

	// Selective storage: aggregate map for events not individually stored.
	sessionStart := time.Now()
	aggs := make(map[aggKey]*aggValue)

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
			flushAllAggregates(aggs, eventStore)
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
				corrs = corr.SnapshotCorrelations(snap.Ops, corrPID)
				chains = corr.SnapshotCausalChains(snap.Ops, corrPID)
			}
			// Store final chains before rendering (ticker stores intermediate
			// chains, but the last snapshot may detect new/upgraded chains).
			if eventStore != nil && len(chains) > 0 {
				eventStore.RecordChains(chainsToStored(chains))
			}
			renderTable(snap, droppedFn(), &linesDrawn, true, corrs, chains)
			return nil

		case evt, ok := <-eventCh:
			if !ok {
				flushAllAggregates(aggs, eventStore)
				updateSysCtx()
				snap := collector.Snapshot()
				attachSysSnapshot(snap, sysColl)
				var corrs []correlate.Correlation
				var chains []correlate.CausalChain
				if corr != nil {
					corrs = corr.SnapshotCorrelations(snap.Ops, corrPID)
					chains = corr.SnapshotCausalChains(snap.Ops, corrPID)
				}
				// Store final chains (same as ctx.Done path).
				if eventStore != nil && len(chains) > 0 {
					eventStore.RecordChains(chainsToStored(chains))
				}
				renderTable(snap, droppedFn(), &linesDrawn, true, corrs, chains)
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

			// Selective storage: decide whether to store individually.
			if eventStore != nil {
				stored := shouldStore(evt, sessionStart, traceRecordAll, collector,
					traceStackSamples, stackSamples)
				if stored {
					eventStore.Record(evt)
					storedEventCount++
					// Track stack sample count for dedup limiting.
					// Uses HashStackSymbols (ASLR-independent) to match shouldStore().
					if len(evt.Stack) > 0 {
						stackSamples[events.HashStackSymbols(evt.Stack)]++
					}
				}
				recordAggregate(aggs, evt, stored)
			}

			// Dynamic PID tracking: register CUDA/driver event PIDs with
			// the host tracer so it collects host events for those processes.
			if evt.Source != events.SourceHost && len(onFork) > 0 && onFork[0] != nil {
				onFork[0](evt.PID)
			}

			// Feed events into correlation engine for causal chain analysis.
			if corr != nil {
				switch evt.Source {
				case events.SourceHost:
					corr.RecordHost(evt)
				case events.SourceIO, events.SourceTCP, events.SourceNet:
					corr.RecordEvent(evt)
				}
				// Auto-register target cgroups for noisy neighbor detection.
				if evt.CGroupID > 1 && pidFilter != nil && pidFilter[evt.PID] {
					corr.SetTargetCGroup(evt.CGroupID)
				}
			}

			// Dynamic PID tracking: when a target process forks, auto-add child.
			if evt.Source == events.SourceHost && events.HostOp(evt.Op) == events.HostProcessFork {
				childPID := uint32(evt.Args[1])
				if childPID > 0 && len(onFork) > 0 && onFork[0] != nil {
					onFork[0](childPID)
				}
			}

		case <-ticker.C:
			updateSysCtx()
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
			if podCache != nil {
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

			// Flush completed minute-buckets to SQLite.
			flushAggregates(aggs, eventStore, time.Now())

			// Flush per-cgroup scheduling stats for noisy neighbor detection.
			if eventStore != nil && corr != nil {
				cgStats := corr.SnapshotCGroupSchedStats()
				if len(cgStats) > 0 {
					storeStats := make([]store.CGroupSchedStat, len(cgStats))
					now := time.Now()
					for i, cs := range cgStats {
						storeStats[i] = store.CGroupSchedStat{
							CGroupID:    cs.CGroupID,
							P99OffCPU:   int64(cs.P99OffCPU),
							TotalOffCPU: int64(cs.TotalOffCPU),
							EventCount:  cs.EventCount,
							WindowEnd:   now.UnixNano(),
						}
					}
					eventStore.RecordCGroupSchedStats(storeStats)
				}
			}

			snap := collector.Snapshot()
			attachSysSnapshot(snap, sysColl)
			if onSnapshot != nil {
				onSnapshot(snap)
			}
			if snap.TotalEvents > 0 || snap.System != nil {
				var corrs []correlate.Correlation
				var chains []correlate.CausalChain
				if corr != nil {
					corrs = corr.SnapshotCorrelations(snap.Ops, corrPID)
					chains = corr.SnapshotCausalChains(snap.Ops, corrPID)
				}
				if eventStore != nil && len(chains) > 0 {
					eventStore.RecordChains(chainsToStored(chains))
				}
				renderTable(snap, droppedFn(), &linesDrawn, false, corrs, chains)
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

func renderTable(snap *stats.Snapshot, dropped uint64, linesDrawn *int, final bool, correlations []correlate.Correlation, chains ...[]correlate.CausalChain) {
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
// pidFilter is the event-loop filter (nil = accept all).
func runJSONMode(ctx context.Context, eventCh <-chan events.Event, collector *stats.Collector, pidFilter map[uint32]bool, eventStore *store.Store, resolver *symtab.Resolver, onSnapshot func(*stats.Snapshot), podCache *k8s.PodCache, snapFilter *filter.SnapshotFilter, procNames *pidNameCache, onFork ...func(uint32)) error {
	enc := json.NewEncoder(os.Stdout)

	// Periodic debug throughput counter (same as runTableMode).
	var debugEventCount uint64
	var storedEventCount uint64
	var debugTickerCh <-chan time.Time
	if debugMode {
		dt := time.NewTicker(10 * time.Second)
		defer dt.Stop()
		debugTickerCh = dt.C
	}

	// Selective storage: aggregate map for events not individually stored.
	sessionStart := time.Now()
	aggs := make(map[aggKey]*aggValue)

	// Stack sampling: per-stack event counter (same as table mode).
	stackSamples := make(map[uint64]int)

	// Cgroup cache: cgroup_id → container_id (lazy, populated on first event).
	cgroupCache := make(map[uint64]string)

	// System context collector — needed for exporter snapshots and for
	// recording system snapshots to SQLite (post-hoc causal chain replay).
	var sysColl *sysinfo.Collector
	if onSnapshot != nil || eventStore != nil {
		sysColl = sysinfo.New()
		sysColl.Start()
		defer sysColl.Stop()
	}

	// Periodic snapshot for exporters (OTLP, Prometheus) — every 1s, same as table mode.
	snapTicker := time.NewTicker(1 * time.Second)
	defer snapTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			flushAllAggregates(aggs, eventStore)
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
			if onSnapshot != nil {
				snap := collector.Snapshot()
				attachSysSnapshot(snap, sysColl)
				onSnapshot(snap)
			}
			// Record system snapshot for post-hoc causal chain replay.
			if eventStore != nil && sysColl != nil {
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
			// Flush completed minute-buckets to SQLite.
			flushAggregates(aggs, eventStore, time.Now())

		case evt, ok := <-eventCh:
			if !ok {
				flushAllAggregates(aggs, eventStore)
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

			// Selective storage: decide whether to store individually.
			if eventStore != nil {
				stored := shouldStore(evt, sessionStart, traceRecordAll, collector,
					traceStackSamples, stackSamples)
				if stored {
					eventStore.Record(evt)
					storedEventCount++
					// Track stack sample count for dedup limiting.
					// Uses HashStackSymbols (ASLR-independent) to match shouldStore().
					if len(evt.Stack) > 0 {
						stackSamples[events.HashStackSymbols(evt.Stack)]++
					}
				}
				recordAggregate(aggs, evt, stored)
			}

			// Dynamic PID tracking: register CUDA/driver event PIDs with
			// the host tracer so it collects host events for those processes.
			if evt.Source != events.SourceHost && len(onFork) > 0 && onFork[0] != nil {
				onFork[0](evt.PID)
			}

			// Dynamic PID tracking: when a target process forks, auto-add child.
			if evt.Source == events.SourceHost && events.HostOp(evt.Op) == events.HostProcessFork {
				childPID := uint32(evt.Args[1])
				if childPID > 0 && len(onFork) > 0 && onFork[0] != nil {
					onFork[0](childPID)
				}
			}

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
func printTraceHeader(libPath string, pids []int, processNames []string, cudaProbeCount int, hostProbeCount int, driverProbeCount int, ioProbeCount int, tcpProbeCount int, netProbeCount int, snapFilter *filter.SnapshotFilter) {
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
		}
	}
	return out
}
