package discover

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"syscall"
)

// CheckResult represents the outcome of a single system readiness check.
type CheckResult struct {
	Name     string // What we checked (e.g., "Kernel version")
	OK       bool   // Whether the check passed
	Optional bool   // If true, failure is informational (not a blocker)
	Value    string // What we found (e.g., "5.15.0-generic")
	Detail   string // Extra context (e.g., "need 5.15+, got 5.15.0")
}

// KernelVersion returns the running kernel version string via uname(2).
func KernelVersion() (string, error) {
	var uname syscall.Utsname
	if err := syscall.Uname(&uname); err != nil {
		return "", fmt.Errorf("uname syscall: %w", err)
	}

	return int8sToString(uname.Release[:]), nil
}

// int8sToString converts a null-terminated int8 array (syscall.Utsname) to a Go string.
// Linux maps C char to int8, so we convert element-by-element.
func int8sToString(chars []int8) string {
	buf := make([]byte, 0, len(chars))
	for _, c := range chars {
		if c == 0 {
			break
		}
		buf = append(buf, byte(c))
	}
	return string(buf)
}

// ParseKernelMajorMinor extracts the major.minor version from a kernel string.
// "5.15.0-100-generic" → (5, 15)
func ParseKernelMajorMinor(version string) (major, minor int, err error) {
	parts := strings.SplitN(version, ".", 3)
	if len(parts) < 2 {
		return 0, 0, fmt.Errorf("unexpected kernel version format: %s", version)
	}

	major, err = strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, fmt.Errorf("parsing major version %q: %w", parts[0], err)
	}

	minor, err = strconv.Atoi(parts[1])
	if err != nil {
		return 0, 0, fmt.Errorf("parsing minor version %q: %w", parts[1], err)
	}

	return major, minor, nil
}

// CheckKernel verifies the kernel version is 5.15+ (minimum for CO-RE eBPF + BTF).
func CheckKernel() CheckResult {
	version, err := KernelVersion()
	if err != nil {
		return CheckResult{
			Name:   "Kernel version",
			OK:     false,
			Detail: fmt.Sprintf("failed to read: %v", err),
		}
	}

	major, minor, err := ParseKernelMajorMinor(version)
	if err != nil {
		return CheckResult{
			Name:   "Kernel version",
			OK:     false,
			Value:  version,
			Detail: fmt.Sprintf("failed to parse: %v", err),
		}
	}

	ok := major > 5 || (major == 5 && minor >= 15)
	detail := "need 5.15+"
	if !ok {
		detail = fmt.Sprintf("need 5.15+, got %d.%d", major, minor)
	}

	return CheckResult{
		Name:   "Kernel version",
		OK:     ok,
		Value:  version,
		Detail: detail,
	}
}

// CheckBTF verifies that BTF (BPF Type Format) is available at /sys/kernel/btf/vmlinux.
// Required for CO-RE: compile once, load on any kernel 5.15+.
func CheckBTF() CheckResult {
	btfPath := "/sys/kernel/btf/vmlinux"
	info, err := os.Stat(btfPath)
	if err != nil {
		return CheckResult{
			Name:   "BTF support",
			OK:     false,
			Detail: "CONFIG_DEBUG_INFO_BTF=y not enabled — required for CO-RE eBPF",
		}
	}

	return CheckResult{
		Name:   "BTF support",
		OK:     true,
		Value:  btfPath,
		Detail: fmt.Sprintf("available (%d bytes)", info.Size()),
	}
}

// nvidiaVersionRe extracts the driver version from nvidia-smi or /proc output.
var nvidiaVersionRe = regexp.MustCompile(`\d+\.\d+(?:\.\d+)?`)

// runNvidiaSMI executes nvidia-smi with the given arguments. In containers
// (e.g., Alpine with NVIDIA Container Toolkit), the injected NVIDIA libraries
// may not be on the default linker search path. If the first attempt fails
// because the binary couldn't be found or loaded (not a legitimate GPU error),
// retry with LD_LIBRARY_PATH set to common container mount points.
func runNvidiaSMI(args ...string) ([]byte, error) {
	// Direct attempt — works on bare metal and glibc-based containers.
	out, err := exec.Command("nvidia-smi", args...).Output()
	if err == nil {
		return out, nil
	}

	// Only retry with LD_LIBRARY_PATH for binary-not-found or shared lib
	// loading failures. Legitimate nvidia-smi errors (GPU in error state,
	// driver bug) should not be retried.
	if !errors.Is(err, exec.ErrNotFound) {
		if exitErr, ok := err.(*exec.ExitError); ok {
			// Exit code 127 = shell "command not found" or dynamic linker failure.
			// Any other exit code means nvidia-smi ran but reported a GPU error.
			if exitErr.ExitCode() != 127 {
				return nil, err
			}
		}
	}

	// Container fallback: NVIDIA Container Toolkit mounts driver libs to
	// /usr/lib64 or /usr/lib/x86_64-linux-gnu but Alpine's musl linker
	// doesn't search these by default.
	cmd := exec.Command("nvidia-smi", args...)
	cmd.Env = append(os.Environ(),
		"LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/aarch64-linux-gnu",
	)
	return cmd.Output()
}

// CheckNVIDIA verifies the NVIDIA driver is installed and reports its version.
// Checks for 550+ (open kernel modules, needed for future kprobe support).
func CheckNVIDIA() CheckResult {
	// Try nvidia-smi first (most reliable).
	out, err := runNvidiaSMI("--query-gpu=driver_version", "--format=csv,noheader")
	if err == nil {
		version := strings.TrimSpace(string(out))
		return checkNVIDIAVersion(version)
	}

	// Fallback: /proc/driver/nvidia/version
	data, err := os.ReadFile("/proc/driver/nvidia/version")
	if err == nil {
		if match := nvidiaVersionRe.FindString(string(data)); match != "" {
			return checkNVIDIAVersion(match)
		}
	}

	// Container fallback: read driver version from host /proc via hostPID.
	data, err = os.ReadFile("/proc/1/root/proc/driver/nvidia/version")
	if err == nil {
		if match := nvidiaVersionRe.FindString(string(data)); match != "" {
			return checkNVIDIAVersion(match)
		}
	}

	return CheckResult{
		Name:   "NVIDIA driver",
		OK:     false,
		Detail: "not found — nvidia-smi not in PATH and /proc/driver/nvidia/ missing",
	}
}

func checkNVIDIAVersion(version string) CheckResult {
	parts := strings.SplitN(version, ".", 2)
	major, _ := strconv.Atoi(parts[0])

	ok := major >= 550
	detail := "open kernel modules (550+)"
	if !ok {
		detail = fmt.Sprintf("found %s — need 550+ for kprobe support (v0.3). uprobes (v0.1) work with any driver", version)
	}

	return CheckResult{
		Name:   "NVIDIA driver",
		OK:     ok,
		Value:  version,
		Detail: detail,
	}
}

// isContainer returns true if we appear to be running inside a container.
func isContainer() bool {
	if _, err := os.Stat("/.dockerenv"); err == nil {
		return true
	}
	// Check cgroup for container signatures
	if data, err := os.ReadFile("/proc/1/cgroup"); err == nil {
		s := string(data)
		if strings.Contains(s, "docker") || strings.Contains(s, "containerd") || strings.Contains(s, "kubepods") {
			return true
		}
	}
	return false
}

// CheckCUDALibrary verifies libcudart.so (CUDA Runtime API) is present.
func CheckCUDALibrary() CheckResult {
	// First check if any process has it loaded (most accurate)
	procs, _ := FindCUDAProcesses()
	if len(procs) > 0 {
		return CheckResult{
			Name:   "CUDA runtime",
			OK:     true,
			Value:  procs[0].LibCUDAPath,
			Detail: fmt.Sprintf("loaded by %d process(es)", len(procs)),
		}
	}

	// No running processes — search filesystem
	path, err := FindLibCUDART()
	if err != nil {
		// In containers, libcudart.so is not injected by the NVIDIA Container
		// Toolkit (it only mounts driver libs like libcuda.so). Ingero discovers
		// libcudart.so from running CUDA processes via /proc/*/maps at trace time.
		if isContainer() {
			return CheckResult{
				Name:     "CUDA runtime",
				OK:       true,
				Optional: true,
				Detail:   "container mode — libcudart.so discovered from host CUDA processes at trace time",
			}
		}
		return CheckResult{
			Name:   "CUDA runtime",
			OK:     false,
			Detail: "libcudart.so not found — install CUDA toolkit",
		}
	}

	return CheckResult{
		Name:   "CUDA runtime",
		OK:     true,
		Value:  path,
		Detail: "found (no CUDA processes currently running)",
	}
}

// CheckLibCUDA checks for the CUDA driver API library (libcuda.so).
func CheckLibCUDA() CheckResult {
	path, err := FindLibCUDA()
	if err != nil {
		return CheckResult{
			Name:   "CUDA driver (libcuda.so)",
			OK:     false,
			Value:  "not found",
			Detail: "needed for driver API tracing (cuBLAS, cuDNN)",
		}
	}
	return CheckResult{
		Name:   "CUDA driver (libcuda.so)",
		OK:     true,
		Value:  path,
		Detail: "available for driver API tracing",
	}
}

// CheckCUDAProcesses finds running CUDA workloads.
func CheckCUDAProcesses() CheckResult {
	procs, err := FindCUDAProcesses()
	if err != nil {
		return CheckResult{
			Name:   "CUDA processes",
			OK:     false,
			Detail: fmt.Sprintf("scan failed: %v", err),
		}
	}

	if len(procs) == 0 {
		return CheckResult{
			Name:   "CUDA processes",
			OK:     true, // Not an error — just nothing running right now
			Value:  "none",
			Detail: "no CUDA processes detected (start a GPU workload to trace)",
		}
	}

	// Build summary
	var names []string
	for _, p := range procs {
		names = append(names, fmt.Sprintf("PID %d (%s)", p.PID, p.Name))
	}

	return CheckResult{
		Name:   "CUDA processes",
		OK:     true,
		Value:  fmt.Sprintf("%d found", len(procs)),
		Detail: strings.Join(names, ", "),
	}
}

// CheckGPUModel queries nvidia-smi for GPU model and memory.
// Reports "NVIDIA RTX 3090 (24576 MB)" or "No GPU detected".
func CheckGPUModel() CheckResult {
	out, err := runNvidiaSMI("--query-gpu=name,memory.total", "--format=csv,noheader")
	if err != nil {
		return CheckResult{
			Name:   "GPU model",
			OK:     false,
			Detail: "No GPU detected",
		}
	}

	model := strings.TrimSpace(string(out))
	return CheckResult{
		Name:  "GPU model",
		OK:    true,
		Value: model,
	}
}

// CPUModel reads the CPU model name.
// Tries /proc/cpuinfo "model name" first (x86_64), then falls back to
// lscpu "Model name" (works on both x86_64 and aarch64).
// Returns e.g. "AMD EPYC 7713 64-Core Processor" or "Neoverse-V2".
func CPUModel() string {
	// /proc/cpuinfo — works on x86_64
	if data, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "model name") {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					return strings.TrimSpace(parts[1])
				}
			}
		}
	}
	// Fallback: lscpu — works on x86_64 and aarch64 (ARM64)
	if out, err := exec.Command("lscpu").Output(); err == nil {
		for _, line := range strings.Split(string(out), "\n") {
			if strings.HasPrefix(line, "Model name:") {
				return strings.TrimSpace(strings.TrimPrefix(line, "Model name:"))
			}
		}
	}
	return ""
}

// CPUCores returns the number of online logical CPU cores.
// Counts "processor" lines in /proc/cpuinfo.
func CPUCores() int {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return 0
	}
	count := 0
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "processor") {
			count++
		}
	}
	return count
}

// OSRelease reads the OS pretty name from /etc/os-release.
// Returns e.g. "Ubuntu 22.04.5 LTS". Empty string on failure.
func OSRelease() string {
	data, err := os.ReadFile("/etc/os-release")
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "PRETTY_NAME=") {
			val := strings.TrimPrefix(line, "PRETTY_NAME=")
			return strings.Trim(val, "\"")
		}
	}
	return ""
}

// CUDAVersion queries nvidia-smi for the CUDA version string.
// Parses "CUDA Version: 12.4" from the nvidia-smi banner output.
// Returns e.g. "12.4". Empty string if nvidia-smi unavailable.
func CUDAVersion() string {
	out, err := runNvidiaSMI()
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(out), "\n") {
		if idx := strings.Index(line, "CUDA Version:"); idx >= 0 {
			part := strings.TrimSpace(line[idx+len("CUDA Version:"):])
			fields := strings.Fields(part)
			if len(fields) > 0 {
				return fields[0]
			}
		}
	}
	return ""
}

// PythonVersion runs "python3 --version" and returns e.g. "3.10.12".
// Empty string if python3 not found.
func PythonVersion() string {
	out, err := exec.Command("python3", "--version").Output()
	if err != nil {
		return ""
	}
	// Output is "Python 3.10.12\n"
	s := strings.TrimSpace(string(out))
	if strings.HasPrefix(s, "Python ") {
		return strings.TrimPrefix(s, "Python ")
	}
	return s
}

// RunAllChecks executes all system readiness checks and returns them in order.
func RunAllChecks() []CheckResult {
	checks := []func() CheckResult{
		CheckKernel,
		CheckBTF,
		CheckNVIDIA,
		CheckGPUModel,
		CheckCUDALibrary,
		CheckLibCUDA,
		CheckCUDAProcesses,
	}

	results := make([]CheckResult, 0, len(checks))
	for _, check := range checks {
		results = append(results, check())
	}
	return results
}
