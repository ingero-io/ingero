// Package discover detects CUDA processes, libraries, and system capabilities.
// Scans /proc/*/maps for libcudart.so to auto-detect uprobe targets.
package discover

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// CUDAProcess represents a running process that has loaded CUDA runtime.
type CUDAProcess struct {
	PID         int    // Process ID
	Name        string // Short name from /proc/<pid>/comm
	Cmdline     string // Full command line
	LibCUDAPath string // Actual path to libcudart.so this process loaded
}

// String implements fmt.Stringer.
func (p CUDAProcess) String() string {
	return fmt.Sprintf("PID %d (%s) → %s", p.PID, p.Name, p.LibCUDAPath)
}

// FindCUDAProcesses scans /proc to find all processes with libcudart.so loaded.
func FindCUDAProcesses() ([]CUDAProcess, error) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, fmt.Errorf("reading /proc: %w", err)
	}

	var processes []CUDAProcess

	for _, entry := range entries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue // Not a PID directory, skip
		}

		libPath, err := findCUDAInMaps(pid)
		if err != nil || libPath == "" {
			continue // No CUDA loaded or can't read maps (permission denied for other users' processes)
		}

		name := readProcFile(fmt.Sprintf("/proc/%d/comm", pid))
		cmdline := readProcCmdline(pid)

		processes = append(processes, CUDAProcess{
			PID:         pid,
			Name:        name,
			Cmdline:     cmdline,
			LibCUDAPath: libPath,
		})
	}

	return processes, nil
}

// findCUDAInMaps reads /proc/<pid>/maps for a libcudart.so mapping.
func findCUDAInMaps(pid int) (string, error) {
	mapsPath := fmt.Sprintf("/proc/%d/maps", pid)
	f, err := os.Open(mapsPath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "libcudart.so") {
			// Extract the file path (last field on the line)
			path := extractPathFromMapsLine(line)
			if path != "" {
				return resolveContainerPath(pid, path), nil
			}
		}
	}

	return "", scanner.Err()
}

// libcudartRe matches the file path at the end of a /proc/pid/maps line.
var libcudartRe = regexp.MustCompile(`\S+libcudart\.so\S*`)

func extractPathFromMapsLine(line string) string {
	return libcudartRe.FindString(line)
}

// cudaSearchPaths are standard locations for CUDA libraries.
var cudaSearchPaths = []string{
	// NVIDIA installer default
	"/usr/local/cuda/lib64",
	// Versioned CUDA installs (e.g., CUDA 12.2)
	"/usr/local/cuda-12/lib64",
	"/usr/local/cuda-11/lib64",
	// Ubuntu/Debian package — architecture-specific multiarch paths
	"/usr/lib/x86_64-linux-gnu",
	"/usr/lib/aarch64-linux-gnu",
	// Conda environments
	"/opt/conda/lib",
	// Generic
	"/usr/lib64",
	"/usr/lib",
}

// FindLibCUDART searches standard paths for libcudart.so.
// Falls back to Python's nvidia.cuda_runtime package, then to host filesystem
// via /proc/1/root/ (for containerized agents with hostPID: true).
func FindLibCUDART() (string, error) {
	if path, err := findLibInPaths(cudaSearchPaths, "libcudart.so*"); err == nil {
		return path, nil
	}

	// Fallback: ask Python's nvidia.cuda_runtime package for the library path.
	if path, err := findLibCUDARTViaPython(); err == nil {
		return path, nil
	}

	// Container fallback: search host filesystem via /proc/1/root/ (hostPID).
	// When ingero runs in a K8s pod with hostPID: true, PID 1 is the host's
	// init process. /proc/1/root/ is a traversable symlink to the host root.
	if path, err := findLibOnHost(cudaSearchPaths, "libcudart.so*"); err == nil {
		return path, nil
	}

	return "", fmt.Errorf("libcudart.so not found in standard paths or Python packages")
}

// findLibInPaths searches directories for a glob pattern and returns the
// resolved (symlink-followed) path of the first match.
func findLibInPaths(dirs []string, pattern string) (string, error) {
	for _, dir := range dirs {
		matches, err := filepath.Glob(filepath.Join(dir, pattern))
		if err != nil {
			continue
		}
		if len(matches) > 0 {
			resolved, err := filepath.EvalSymlinks(matches[0])
			if err != nil {
				return matches[0], nil
			}
			return resolved, nil
		}
	}
	return "", fmt.Errorf("not found")
}

// findLibOnHost searches host filesystem via /proc/1/root/ for a library.
// Only works when running with hostPID: true (K8s DaemonSet or privileged container).
func findLibOnHost(dirs []string, pattern string) (string, error) {
	const hostRoot = "/proc/1/root"
	// Quick check: is /proc/1/root traversable?
	if _, err := os.Stat(hostRoot); err != nil {
		return "", fmt.Errorf("host root not accessible: %w", err)
	}
	for _, dir := range dirs {
		hostDir := filepath.Join(hostRoot, dir)
		matches, err := filepath.Glob(filepath.Join(hostDir, pattern))
		if err != nil {
			continue
		}
		if len(matches) > 0 {
			// Return the /proc/1/root/... path — cilium/ebpf's
			// link.OpenExecutable() can open it for uprobe attachment.
			resolved, err := filepath.EvalSymlinks(matches[0])
			if err != nil {
				return matches[0], nil
			}
			return resolved, nil
		}
	}
	return "", fmt.Errorf("not found on host")
}

// findLibCUDARTViaPython discovers libcudart.so from Python's nvidia.cuda_runtime package.
// This handles the common case where PyTorch is pip-installed and bundles its own CUDA runtime.
// When running as root (via sudo), also tries running Python as the original user (SUDO_USER).
func findLibCUDARTViaPython() (string, error) {
	pyCmd := `import nvidia.cuda_runtime, os; print(os.path.join(nvidia.cuda_runtime.__path__[0], 'lib'))`

	// Try directly first.
	if dir, err := runPythonCmd(pyCmd); err == nil {
		if path, err := findLibInDir(dir); err == nil {
			return path, nil
		}
	}

	// When running as root via sudo, Python packages may be installed for the
	// original user only. Try running as SUDO_USER.
	if sudoUser := os.Getenv("SUDO_USER"); sudoUser != "" {
		if dir, err := runPythonCmdAsUser(pyCmd, sudoUser); err == nil {
			if path, err := findLibInDir(dir); err == nil {
				return path, nil
			}
		}
	}

	return "", fmt.Errorf("python nvidia.cuda_runtime not available")
}

func runPythonCmd(code string) (string, error) {
	out, err := exec.Command("python3", "-c", code).Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func runPythonCmdAsUser(code, user string) (string, error) {
	out, err := exec.Command("su", "-", user, "-c", fmt.Sprintf("python3 -c %q", code)).Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func findLibInDir(dir string) (string, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "libcudart.so*"))
	if err != nil || len(matches) == 0 {
		return "", fmt.Errorf("no libcudart.so in %s", dir)
	}
	resolved, err := filepath.EvalSymlinks(matches[0])
	if err != nil {
		return matches[0], nil
	}
	return resolved, nil
}

// driverSearchPaths are standard locations for the CUDA driver library.
var driverSearchPaths = []string{
	"/usr/lib/x86_64-linux-gnu",
	"/usr/lib/aarch64-linux-gnu",
	"/usr/local/cuda/lib64",
	"/usr/local/cuda/compat",
	"/usr/lib64",
	"/usr/lib",
}

// FindLibCUDA searches standard paths for libcuda.so (the CUDA driver API library).
// This is separate from libcudart.so (the runtime API). cuBLAS, cuDNN, and other
// NVIDIA math libraries call libcuda.so directly, bypassing the runtime API.
func FindLibCUDA() (string, error) {
	if path, err := findLibInPaths(driverSearchPaths, "libcuda.so*"); err == nil {
		return path, nil
	}

	// Check /proc/*/maps for a running process that has libcuda.so loaded.
	entries, _ := os.ReadDir("/proc")
	for _, entry := range entries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		if path, err := findLibInProcMaps(pid, "libcuda.so"); err == nil && path != "" {
			return path, nil
		}
	}

	// Container fallback: search host filesystem via /proc/1/root/.
	if path, err := findLibOnHost(driverSearchPaths, "libcuda.so*"); err == nil {
		return path, nil
	}

	return "", fmt.Errorf("libcuda.so not found")
}

// findLibInProcMaps searches /proc/<pid>/maps for a library matching substr.
// Extracts the file path from the last whitespace-delimited field in the maps line,
// avoiding per-call regex compilation.
func findLibInProcMaps(pid int, substr string) (string, error) {
	f, err := os.Open(fmt.Sprintf("/proc/%d/maps", pid))
	if err != nil {
		return "", err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.Contains(line, substr) {
			continue
		}
		// /proc/pid/maps format: addr perms offset dev inode [pathname]
		// The pathname is the last field (field 6+). Use Fields to extract it.
		fields := strings.Fields(line)
		if len(fields) >= 6 {
			path := fields[len(fields)-1]
			if strings.Contains(path, substr) {
				return resolveContainerPath(pid, path), nil
			}
		}
	}
	return "", scanner.Err()
}

// resolveContainerPath handles library path resolution when ingero runs in a
// container (e.g., K8s DaemonSet). Paths from /proc/[pid]/maps refer to the
// target process's mount namespace. If the path doesn't exist in our namespace
// (container), resolve through /proc/[pid]/root/ which traverses the target's
// filesystem. On bare metal this is a no-op (direct path always exists).
func resolveContainerPath(pid int, path string) string {
	if _, err := os.Stat(path); err == nil {
		return path // Direct path works (bare metal or shared mount)
	}
	altPath := fmt.Sprintf("/proc/%d/root%s", pid, path)
	if _, err := os.Stat(altPath); err == nil {
		return altPath
	}
	return path // Return original; let caller handle the error
}

// readProcFile reads a small /proc file and returns its trimmed content.
// Returns "" on any error (permission denied, process gone, etc).
func readProcFile(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// readProcCmdline reads /proc/<pid>/cmdline, replacing null separators with spaces.
func readProcCmdline(pid int) string {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
	if err != nil {
		return ""
	}
	// Replace null bytes with spaces, trim trailing null
	cmdline := strings.TrimRight(string(data), "\x00")
	return strings.ReplaceAll(cmdline, "\x00", " ")
}
