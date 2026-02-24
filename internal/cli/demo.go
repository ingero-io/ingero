package cli

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/ingero-io/ingero/internal/correlate"
	"github.com/ingero-io/ingero/internal/discover"
	"github.com/ingero-io/ingero/internal/ebpf/cuda"
	"github.com/ingero-io/ingero/internal/ebpf/driver"
	"github.com/ingero-io/ingero/internal/ebpf/host"
	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/internal/symtab"
	"github.com/ingero-io/ingero/internal/synth"
	"github.com/ingero-io/ingero/internal/version"
	"github.com/ingero-io/ingero/pkg/events"
)

// Flag variables for the demo command.
var (
	demoDuration time.Duration
	demoJSON     bool
	demoVerbose  bool
	demoSpeed    float64
	demoNoGPU    bool
	demoGPU      bool
	demoStack    bool
)

var demoCmd = &cobra.Command{
	Use:   "demo [scenario]",
	Short: "Run demo scenarios — auto-detects GPU, defaults to all 6 scenarios",
	Long: `Run demo scenarios that showcase Ingero's CUDA profiling insights.

Auto-detect mode (default):
  Automatically detects GPU availability and runs all scenarios.
  Uses real eBPF tracing if GPU is present, synthetic events otherwise.

Examples:
  ingero demo                           # auto-detect GPU, run all
  ingero demo cold-start                # auto-detect GPU, one scenario
  sudo ingero demo --gpu                # force GPU mode
  ingero demo --no-gpu                  # force synthetic mode
  ingero demo --no-gpu --speed 5        # fast synthetic mode`,

	Args: cobra.MaximumNArgs(1),
	RunE: demoRunE,
}

func init() {
	demoCmd.Flags().DurationVarP(&demoDuration, "duration", "d", 0, "stop after duration (e.g., 30s, 5m). 0 = run until scenario completes or Ctrl+C")
	demoCmd.Flags().BoolVar(&demoJSON, "json", false, "output events as JSON lines")
	demoCmd.Flags().BoolVarP(&demoVerbose, "verbose", "v", false, "show detailed event data including raw args")
	demoCmd.Flags().Float64Var(&demoSpeed, "speed", 1.0, "playback speed multiplier for synthetic mode (e.g., 5 = 5x faster)")
	demoCmd.Flags().BoolVar(&demoNoGPU, "no-gpu", false, "force synthetic mode (no root, no GPU needed)")
	demoCmd.Flags().BoolVar(&demoGPU, "gpu", false, "force GPU mode (requires sudo + GPU + PyTorch)")
	demoCmd.Flags().BoolVar(&demoStack, "stack", true, "capture userspace stack traces (GPU mode only, use --stack=false to disable)")
	demoCmd.MarkFlagsMutuallyExclusive("no-gpu", "gpu")

	rootCmd.AddCommand(demoCmd)
}

// ---------------------------------------------------------------------------
// Main demo logic
// ---------------------------------------------------------------------------

func demoRunE(cmd *cobra.Command, args []string) error {
	// Default to "all" when no scenario specified.
	name := "all"
	if len(args) > 0 {
		name = strings.ToLower(strings.TrimSpace(args[0]))
	}

	// Set up graceful shutdown.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if demoDuration > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, demoDuration)
		defer cancel()
	}

	// Decide mode: --no-gpu, --gpu, or auto-detect.
	debugf("demo: scenario=%s speed=%.1f no-gpu=%v gpu=%v", name, demoSpeed, demoNoGPU, demoGPU)
	useGPU := false
	switch {
	case demoNoGPU:
		fmt.Println("  Running synthetic demo (--no-gpu)")
		fmt.Println()
	case demoGPU:
		// Force GPU mode — fail hard if prereqs missing.
		if err := checkGPUPrereqs(); err != nil {
			return err
		}
		useGPU = true
	default:
		// Auto-detect: try GPU, fall back to synthetic.
		if reason := detectGPU(); reason == "" {
			fmt.Println("  GPU detected — running real eBPF tracing")
			fmt.Println()
			useGPU = true
		} else {
			fmt.Printf("  No GPU detected (%s) — running synthetic demo\n", reason)
			fmt.Println()
		}
	}

	// Print GPU detection header before any scenario.
	printDemoHeader(useGPU)

	if useGPU {
		return runGPUDemo(ctx, name)
	}
	return runSyntheticDemo(ctx, name)
}

// printDemoHeader prints GPU detection info at the start of any demo.
func printDemoHeader(gpuMode bool) {
	fmt.Printf("  Ingero %s — Demo Mode\n", version.Version())

	// GPU model detection.
	gpuCheck := discover.CheckGPUModel()
	if gpuCheck.OK {
		// Also get driver version.
		driverCheck := discover.CheckNVIDIA()
		driverStr := ""
		if driverCheck.OK {
			driverStr = fmt.Sprintf(" | Driver: %s", driverCheck.Value)
		}
		fmt.Printf("  GPU: %s%s\n", gpuCheck.Value, driverStr)
	} else {
		fmt.Println("  GPU: not detected")
	}

	if gpuMode {
		fmt.Println("  Mode: GPU (eBPF tracing active)")
	} else {
		fmt.Println("  Mode: Synthetic (simulated events)")
	}
	fmt.Println()
}

// ---------------------------------------------------------------------------
// Scenario listing
// ---------------------------------------------------------------------------

func listScenarios() {
	fmt.Println("Available demo scenarios:")
	fmt.Println()
	for _, s := range synth.Registry {
		fmt.Printf("  %-20s %s\n", s.Name, s.Description)
	}
	fmt.Println()
	fmt.Println("GPU mode (default — requires sudo + GPU + python3 + torch):")
	fmt.Println("  sudo ingero demo <name>")
	fmt.Println("  sudo ingero demo all")
	fmt.Println()
	fmt.Println("Synthetic mode (no GPU needed):")
	fmt.Println("  ingero demo --no-gpu <name>")
	fmt.Println("  ingero demo --no-gpu all")
}

// ---------------------------------------------------------------------------
// Synthetic mode (--no-gpu) — same as original demo
// ---------------------------------------------------------------------------

func runSyntheticDemo(ctx context.Context, name string) error {
	if name == "all" {
		return runAllSyntheticScenarios(ctx)
	}

	scenario := synth.Find(name)
	if scenario == nil {
		fmt.Fprintf(os.Stderr, "Unknown scenario: %q\n\n", name)
		listScenarios()
		return fmt.Errorf("unknown scenario: %s", name)
	}

	return runSyntheticScenario(ctx, scenario, true)
}

func runAllSyntheticScenarios(ctx context.Context) error {
	for i, s := range synth.Registry {
		if ctx.Err() != nil {
			return nil
		}
		if i > 0 {
			fmt.Println()
			fmt.Println("─────────────────────────────────────────")
			fmt.Println()
		}
		if err := runSyntheticScenario(ctx, s, false); err != nil {
			return err
		}
	}
	return nil
}

func runSyntheticScenario(ctx context.Context, s *synth.Scenario, loop bool) error {
	fmt.Printf("Ingero Demo — %s (synthetic)\n", s.Title)
	fmt.Println()
	fmt.Printf("  Scenario: %s\n", s.Name)
	fmt.Printf("  %s\n", s.Description)

	if demoDuration > 0 {
		fmt.Printf("  Duration: %s\n", demoDuration)
	} else if loop {
		fmt.Println("  Duration: until Ctrl+C (looping)")
	} else {
		fmt.Println("  Duration: one cycle")
	}
	if demoSpeed != 1.0 {
		fmt.Printf("  Speed: %.1fx\n", demoSpeed)
	}
	fmt.Println()

	ch := make(chan events.Event, 256)
	collector := stats.New()
	correlator := correlate.New()

	go func() {
		defer close(ch)
		if loop {
			for ctx.Err() == nil {
				s.Generate(ctx, ch, demoSpeed)
			}
		} else {
			s.Generate(ctx, ch, demoSpeed)
		}
	}()

	noDrops := func() uint64 { return 0 }
	traceVerbose = demoVerbose

	var err error
	if demoJSON {
		err = runJSONMode(ctx, ch, collector, 0, nil, nil, nil)
	} else {
		err = runTableMode(ctx, ch, collector, 0, noDrops, nil, nil, correlator, nil)
	}

	if !demoJSON {
		fmt.Println()
		fmt.Printf("  Insight: %s\n", s.Insight)
	}

	return err
}

// ---------------------------------------------------------------------------
// GPU mode (default) — real workload + eBPF tracing
// ---------------------------------------------------------------------------

func runGPUDemo(ctx context.Context, name string) error {
	if name == "all" {
		return runAllGPUScenarios(ctx)
	}

	scenario := synth.Find(name)
	if scenario == nil {
		fmt.Fprintf(os.Stderr, "Unknown scenario: %q\n\n", name)
		listScenarios()
		return fmt.Errorf("unknown scenario: %s", name)
	}

	return runGPUScenario(ctx, scenario)
}

func runAllGPUScenarios(ctx context.Context) error {
	for i, s := range synth.Registry {
		if ctx.Err() != nil {
			return nil
		}
		if i > 0 {
			fmt.Println()
			fmt.Println("─────────────────────────────────────────")
			fmt.Println()
		}
		if err := runGPUScenario(ctx, s); err != nil {
			return err
		}
	}
	return nil
}

// runGPUScenario runs a real GPU workload with eBPF tracing.
//
// The flow mirrors traceRunE but launches a Python subprocess as the workload:
//   1. Write embedded Python script to temp file
//   2. Find libcudart.so
//   3. Attach eBPF probes
//   4. Launch workload subprocess
//   5. Start eBPF ring buffer reader
//   6. Run display loop
//   7. Subprocess exits or Ctrl+C → cleanup
func runGPUScenario(ctx context.Context, s *synth.Scenario) error {
	if s.GPUScript == "" {
		return fmt.Errorf("scenario %q has no GPU script", s.Name)
	}

	// Step 1: Write Python script to temp file.
	tmpDir, err := os.MkdirTemp("", "ingero-demo-*")
	if err != nil {
		return fmt.Errorf("creating temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)
	os.Chmod(tmpDir, 0755)

	scriptPath := filepath.Join(tmpDir, s.Name+".py")
	if err := os.WriteFile(scriptPath, []byte(s.GPUScript), 0755); err != nil {
		return fmt.Errorf("writing workload script: %w", err)
	}

	// Step 2: Find libcudart.so.
	libPath, err := discover.FindLibCUDART()
	if err != nil {
		return fmt.Errorf("finding libcudart.so: %w\nHint: use --no-gpu for synthetic mode", err)
	}

	// Step 3: Attach all three eBPF tracer layers (same as trace).
	var cudaOpts []cuda.Option
	if demoStack {
		cudaOpts = append(cudaOpts, cuda.WithStackCapture(true))
	}
	cudaTracer := cuda.New(libPath, cudaOpts...)
	if err := cudaTracer.Attach(); err != nil {
		return fmt.Errorf("attaching CUDA probes: %w", err)
	}
	defer cudaTracer.Close()

	cudaProbes := cudaTracer.ProbeCount()
	hostProbes := 0
	driverProbes := 0

	// Host tracer (sched_switch, mm_page_alloc — needed for causal chains).
	var hostTracer *host.Tracer
	ht := host.New(0) // PID 0 = dynamic tracking
	if err := ht.Attach(); err != nil {
		fmt.Fprintf(os.Stderr, "  Warning: host tracepoints unavailable: %v\n", err)
	} else {
		hostTracer = ht
		defer hostTracer.Close()
		hostProbes = 4 // sched_switch, sched_wakeup, mm_page_alloc, oom_kill
	}

	// Driver tracer (cuLaunchKernel, cuMemcpy — captures cuBLAS/cuDNN).
	var driverTracer *driver.Tracer
	if libcudaPath, err := discover.FindLibCUDA(); err == nil {
		var driverOpts []driver.Option
		if demoStack {
			driverOpts = append(driverOpts, driver.WithStackCapture(true))
		}
		dt := driver.New(libcudaPath, driverOpts...)
		if err := dt.Attach(); err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: driver API tracing unavailable: %v\n", err)
		} else {
			driverTracer = dt
			defer driverTracer.Close()
			driverProbes = driverTracer.ProbeCount()
		}
	}

	debugf("demo GPU: CUDA=%d host=%d driver=%d probes, script=%s", cudaProbes, hostProbes, driverProbes, scriptPath)

	// Print header.
	fmt.Printf("Ingero Demo — %s (GPU)\n", s.Title)
	fmt.Println()
	fmt.Printf("  Scenario: %s\n", s.Name)
	fmt.Printf("  %s\n", s.Description)
	fmt.Printf("  Library: %s\n", libPath)
	fmt.Printf("  Probes: %d CUDA + %d host + %d driver\n", cudaProbes, hostProbes, driverProbes)
	if demoStack {
		fmt.Println("  Stack traces: enabled")
	}
	fmt.Println()

	// Step 4: Launch workload subprocess.
	var cmd *exec.Cmd
	if sudoUser := os.Getenv("SUDO_USER"); sudoUser != "" {
		cmd = exec.CommandContext(ctx, "su", "-", sudoUser, "-c", "python3 "+scriptPath)
	} else {
		cmd = exec.CommandContext(ctx, "python3", scriptPath)
	}
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("launching workload: %w", err)
	}

	targetPID := 0
	if os.Getenv("SUDO_USER") == "" {
		targetPID = cmd.Process.Pid
	}
	fmt.Fprintf(os.Stderr, "  Workload PID: %d\n\n", cmd.Process.Pid)

	// Step 5: Start eBPF readers and merge event channels.
	go cudaTracer.Run(ctx)
	if hostTracer != nil {
		go hostTracer.Run(ctx)
		if targetPID > 0 {
			hostTracer.SetTargetPID(uint32(targetPID))
		}
	}
	if driverTracer != nil {
		go driverTracer.Run(ctx)
	}

	merged := mergeAllEventChannels(ctx, cudaTracer.Events(), hostTracer, driverTracer)
	collector := stats.New()

	// Symbol resolver for stack traces.
	var resolver *symtab.Resolver
	if demoStack {
		if debugMode {
			symtab.SetDebugLog(debugf)
		}
		resolver = symtab.NewResolver()
	}

	// Correlator for causal chain detection.
	corr := correlate.New()

	// Dynamic PID tracking (same as trace).
	trackedPIDs := make(map[uint32]bool)
	if targetPID > 0 {
		trackedPIDs[uint32(targetPID)] = true
	}
	trackPID := func(pid uint32) {
		if hostTracer == nil || pid == 0 {
			return
		}
		if !trackedPIDs[pid] {
			trackedPIDs[pid] = true
			hostTracer.SetTargetPID(pid)
			debugf("demo: dynamically added PID %d", pid)
		}
	}

	// Dropped events counter.
	droppedFn := func() uint64 {
		d := cudaTracer.Dropped()
		if hostTracer != nil {
			d += hostTracer.Dropped()
		}
		if driverTracer != nil {
			d += driverTracer.Dropped()
		}
		return d
	}

	// Step 6: Wait for subprocess in background.
	workloadCtx, workloadCancel := context.WithCancel(ctx)
	defer workloadCancel()

	go func() {
		cmd.Wait()
		workloadCancel()
	}()

	// Step 7: Run display loop.
	traceVerbose = demoVerbose

	if demoJSON {
		runJSONMode(workloadCtx, merged, collector, targetPID, nil, resolver, nil, trackPID)
	} else {
		runTableMode(workloadCtx, merged, collector, targetPID, droppedFn, nil, nil, corr, resolver, trackPID)
	}

	// Print insight after the table.
	if !demoJSON {
		fmt.Println()
		fmt.Printf("  Insight: %s\n", s.Insight)
	}

	return nil
}

// ---------------------------------------------------------------------------
// GPU prerequisites check
// ---------------------------------------------------------------------------

// detectGPU checks whether GPU demo mode is possible.
// Returns "" if all prereqs are met, or a short reason string if not.
func detectGPU() string {
	if os.Geteuid() != 0 {
		return "not root"
	}
	if _, err := exec.LookPath("python3"); err != nil {
		return "python3 not found"
	}
	if _, err := discover.FindLibCUDART(); err != nil {
		return "libcudart.so not found"
	}

	pyCheck := "import torch; print('cuda' if torch.cuda.is_available() else 'no-cuda')"
	var out []byte
	var pyErr error
	if sudoUser := os.Getenv("SUDO_USER"); sudoUser != "" {
		out, pyErr = exec.Command("su", "-", sudoUser, "-c", fmt.Sprintf("python3 -c %q", pyCheck)).Output()
	} else {
		out, pyErr = exec.Command("python3", "-c", pyCheck).Output()
	}
	if pyErr != nil {
		return "PyTorch not installed"
	}
	if strings.TrimSpace(string(out)) == "no-cuda" {
		return "no CUDA device"
	}
	return ""
}

// checkGPUPrereqs verifies all requirements for GPU demo mode.
// Used when --gpu is explicitly passed — returns hard errors.
func checkGPUPrereqs() error {
	if os.Geteuid() != 0 {
		return fmt.Errorf("GPU demo requires root for eBPF tracing; run: sudo ingero demo --gpu")
	}
	if _, err := exec.LookPath("python3"); err != nil {
		return fmt.Errorf("python3 not found; install Python 3.8+ to run GPU demos")
	}

	pyCheck := "import torch; print('cuda' if torch.cuda.is_available() else 'no-cuda')"
	var out []byte
	var pyErr error
	if sudoUser := os.Getenv("SUDO_USER"); sudoUser != "" {
		out, pyErr = exec.Command("su", "-", sudoUser, "-c", fmt.Sprintf("python3 -c %q", pyCheck)).Output()
	} else {
		out, pyErr = exec.Command("python3", "-c", pyCheck).Output()
	}
	if pyErr != nil {
		return fmt.Errorf("PyTorch not found; install with: pip install torch")
	}
	if strings.TrimSpace(string(out)) == "no-cuda" {
		return fmt.Errorf("PyTorch found but no CUDA device available; ensure NVIDIA driver is installed and a GPU is present")
	}
	return nil
}
