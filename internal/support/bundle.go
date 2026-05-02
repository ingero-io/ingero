// Package support builds a single tarball that an operator can attach
// to a support case. The bundle's contents are exactly what an Ingero
// developer would ask for during triage: kernel + BTF info, GPU +
// driver state, build identity, recent agent logs, and a redacted
// environment dump.
//
// Each input is best-effort: if a step fails (file missing, command
// not on PATH, permission denied), the failure is captured as a
// per-file error.txt instead of aborting the whole bundle. Operators
// always end up with something useful on disk.
package support

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// Bundle writes a gzipped tarball at outPath with the diagnostic
// inputs listed in the package comment. Returns the absolute path
// of the file actually written and the number of entries it
// contains. The function NEVER aborts on input errors; every entry
// is best-effort.
func Bundle(outPath, ingeroVersion, ingeroCommit string) (string, int, error) {
	abs, err := filepath.Abs(outPath)
	if err != nil {
		return "", 0, fmt.Errorf("resolve out path: %w", err)
	}
	f, err := os.Create(abs)
	if err != nil {
		return "", 0, fmt.Errorf("create %s: %w", abs, err)
	}
	defer f.Close()

	gz := gzip.NewWriter(f)
	defer gz.Close()
	tw := tar.NewWriter(gz)
	defer tw.Close()

	count := 0
	collect := newCollector(tw)

	collect.addText("metadata.txt", buildMetadata(ingeroVersion, ingeroCommit))
	collect.addCmd("kernel-uname.txt", "uname", "-a")
	collect.addFile("kernel-version.txt", "/proc/version")
	collect.addCmd("os-release.txt", "cat", "/etc/os-release")
	collect.addCmd("nvidia-smi.txt", "nvidia-smi")
	collect.addCmd("nvidia-smi-query.csv", "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory", "--format=csv")
	collect.addFile("nvidia-driver-version.txt", "/proc/driver/nvidia/version")
	collect.addCmd("lscpu.txt", "lscpu")
	collect.addCmd("cgroup-version.txt", "stat", "-f", "-c", "%T", "/sys/fs/cgroup")
	collect.addBTFInfo("btf-info.txt")
	collect.addCmd("bpftool-features.txt", "bpftool", "feature", "probe")
	collect.addCmd("ptrace-scope.txt", "cat", "/proc/sys/kernel/yama/ptrace_scope")
	collect.addEnv("env-redacted.txt")
	collect.addLogTail("agent-trace.log", os.Getenv("HOME")+"/.ingero/trace.log", 1000)
	collect.addLogTail("sink-stderr.log", os.Getenv("HOME")+"/.ingero/sink.log", 1000)
	count = collect.count

	if collect.firstErr != nil {
		// Tar writer or gzip writer failed mid-bundle; surface the
		// error but the file on disk may still contain partial
		// content the operator can salvage.
		return abs, count, collect.firstErr
	}
	return abs, count, nil
}

// collector wraps a tar writer and tracks the first hard write error
// (input failures are inlined as error.txt entries; output errors
// terminate the bundle).
type collector struct {
	tw       *tar.Writer
	count    int
	firstErr error
}

func newCollector(tw *tar.Writer) *collector { return &collector{tw: tw} }

func (c *collector) addText(name, body string) {
	if c.firstErr != nil {
		return
	}
	hdr := &tar.Header{
		Name:    "ingero-support/" + name,
		Mode:    0o644,
		Size:    int64(len(body)),
		ModTime: time.Now().UTC(),
	}
	if err := c.tw.WriteHeader(hdr); err != nil {
		c.firstErr = err
		return
	}
	if _, err := io.WriteString(c.tw, body); err != nil {
		c.firstErr = err
		return
	}
	c.count++
}

func (c *collector) addFile(name, srcPath string) {
	body, err := os.ReadFile(srcPath)
	if err != nil {
		c.addText(name+".error.txt", fmt.Sprintf("read %s: %v\n", srcPath, err))
		return
	}
	c.addText(name, string(body))
}

func (c *collector) addCmd(name, prog string, args ...string) {
	out, err := exec.Command(prog, args...).CombinedOutput()
	if err != nil {
		c.addText(name+".error.txt", fmt.Sprintf("%s %s: %v\noutput:\n%s\n", prog, strings.Join(args, " "), err, out))
		return
	}
	c.addText(name, string(out))
}

func (c *collector) addBTFInfo(name string) {
	const path = "/sys/kernel/btf/vmlinux"
	st, err := os.Stat(path)
	if err != nil {
		c.addText(name+".error.txt", fmt.Sprintf("stat %s: %v\n", path, err))
		return
	}
	c.addText(name, fmt.Sprintf("%s: %d bytes (mode=%v)\n", path, st.Size(), st.Mode()))
}

// addEnv writes the process's environment with values masked for any
// key whose name suggests a credential. The mask preserves the key
// so support readers can confirm WHICH variable was set without
// reading the value.
func (c *collector) addEnv(name string) {
	env := os.Environ()
	sort.Strings(env)
	var b bytes.Buffer
	for _, kv := range env {
		eq := strings.IndexByte(kv, '=')
		if eq < 0 {
			b.WriteString(kv + "\n")
			continue
		}
		k, v := kv[:eq], kv[eq+1:]
		if isSensitive(k) {
			b.WriteString(k + "=<redacted, " + fmt.Sprint(len(v)) + " bytes>\n")
		} else {
			b.WriteString(k + "=" + v + "\n")
		}
	}
	c.addText(name, b.String())
}

// addLogTail writes the last `n` lines of the named log file. Missing
// or empty file is recorded as a one-line note (NOT an error.txt) so
// the bundle's contents document the absence cleanly.
func (c *collector) addLogTail(name, srcPath string, n int) {
	body, err := os.ReadFile(srcPath)
	if err != nil {
		if os.IsNotExist(err) {
			c.addText(name, fmt.Sprintf("(file not present: %s)\n", srcPath))
			return
		}
		c.addText(name+".error.txt", fmt.Sprintf("read %s: %v\n", srcPath, err))
		return
	}
	lines := strings.Split(strings.TrimRight(string(body), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	c.addText(name, strings.Join(lines, "\n")+"\n")
}

func buildMetadata(version, commit string) string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "ingero support bundle\n")
	fmt.Fprintf(&b, "generated_at: %s\n", time.Now().UTC().Format(time.RFC3339))
	fmt.Fprintf(&b, "ingero_version: %s\n", version)
	fmt.Fprintf(&b, "ingero_commit:  %s\n", commit)
	fmt.Fprintf(&b, "go_runtime:     %s\n", runtime.Version())
	fmt.Fprintf(&b, "goos:           %s\n", runtime.GOOS)
	fmt.Fprintf(&b, "goarch:         %s\n", runtime.GOARCH)
	hn, _ := os.Hostname()
	fmt.Fprintf(&b, "hostname:       %s\n", hn)
	return b.String()
}

// sensitiveKeys lists case-insensitive substrings; an env-var key
// containing any of these has its value masked. Conservative:
// false positives are fine, false negatives leak.
var sensitiveKeys = []string{
	"TOKEN", "SECRET", "PASS", "KEY", "API_", "_API", "CREDENTIALS",
	"AUTH", "PRIVATE", "SESSION", "COOKIE", "AWS_", "GCP_",
	"AZURE_", "GH_TOKEN", "GITHUB_TOKEN",
}

func isSensitive(key string) bool {
	upper := strings.ToUpper(key)
	for _, m := range sensitiveKeys {
		if strings.Contains(upper, m) {
			return true
		}
	}
	return false
}
