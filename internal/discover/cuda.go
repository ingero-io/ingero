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
				return path, nil
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

// FindLibCUDART searches standard paths for libcudart.so.
// Falls back to Python's nvidia.cuda_runtime package (pip-installed PyTorch).
func FindLibCUDART() (string, error) {
	searchPaths := []string{
		// NVIDIA installer default
		"/usr/local/cuda/lib64",
		// Versioned CUDA installs (e.g., CUDA 12.2)
		"/usr/local/cuda-12/lib64",
		"/usr/local/cuda-11/lib64",
		// Ubuntu/Debian package
		"/usr/lib/x86_64-linux-gnu",
		// Conda environments
		"/opt/conda/lib",
		// Generic
		"/usr/lib64",
		"/usr/lib",
	}

	for _, dir := range searchPaths {
		matches, err := filepath.Glob(filepath.Join(dir, "libcudart.so*"))
		if err != nil {
			continue
		}
		if len(matches) > 0 {
			// Resolve symlinks to get the actual file
			// (libcudart.so → libcudart.so.12 → libcudart.so.12.2.140)
			resolved, err := filepath.EvalSymlinks(matches[0])
			if err != nil {
				return matches[0], nil // Return unresolved if symlink resolution fails
			}
			return resolved, nil
		}
	}

	// Fallback: ask Python's nvidia.cuda_runtime package for the library path.
	// pip-installed PyTorch bundles libcudart.so in a non-standard location like:
	//   ~/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
	if path, err := findLibCUDARTViaPython(); err == nil {
		return path, nil
	}

	return "", fmt.Errorf("libcudart.so not found in standard paths or Python packages")
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

// FindLibCUDA searches standard paths for libcuda.so (the CUDA driver API library).
// This is separate from libcudart.so (the runtime API). cuBLAS, cuDNN, and other
// NVIDIA math libraries call libcuda.so directly, bypassing the runtime API.
func FindLibCUDA() (string, error) {
	searchPaths := []string{
		"/usr/lib/x86_64-linux-gnu",
		"/usr/local/cuda/lib64",
		"/usr/local/cuda/compat",
		"/usr/lib64",
		"/usr/lib",
	}

	for _, dir := range searchPaths {
		matches, err := filepath.Glob(filepath.Join(dir, "libcuda.so*"))
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

	// Also check /proc/*/maps for a running process that has libcuda.so loaded.
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

	return "", fmt.Errorf("libcuda.so not found")
}

// findLibInProcMaps searches /proc/<pid>/maps for a library matching substr.
func findLibInProcMaps(pid int, substr string) (string, error) {
	f, err := os.Open(fmt.Sprintf("/proc/%d/maps", pid))
	if err != nil {
		return "", err
	}
	defer f.Close()

	re := regexp.MustCompile(`\S+` + regexp.QuoteMeta(substr) + `\S*`)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, substr) {
			if path := re.FindString(line); path != "" {
				return path, nil
			}
		}
	}
	return "", scanner.Err()
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
