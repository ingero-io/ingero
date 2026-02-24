// Package sysinfo reads system-level CPU, memory, and load metrics from /proc.
//
// Call chain: sysinfo.Collector.Start() polls /proc once/sec →
//   watch.go reads sysinfo.Snapshot() each display tick →
//   correlate.Engine.SetSystemSnapshot() uses it for causal chain context →
//   export.OTLP pushes system.* metrics
//
// No eBPF, no root required. Works on any Linux.
package sysinfo

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// SystemSnapshot holds point-in-time CPU/memory/load metrics.
// Read from /proc — no eBPF, no root required.
type SystemSnapshot struct {
	CPUPercent float64 // overall CPU utilization (from /proc/stat deltas)
	MemUsedPct float64 // RAM used % (from /proc/meminfo)
	MemAvailMB int64   // available MB
	MemTotalMB int64   // total MB
	SwapUsedMB int64   // swap used (0 = healthy)
	LoadAvg1   float64 // 1-minute load average (from /proc/loadavg)
	LoadAvg5   float64 // 5-minute load average
	PageFaults int64   // major page faults delta since last read (from /proc/vmstat)
	Timestamp  time.Time
}

// IsAbnormal returns true if any metric is in a concerning range.
func (s SystemSnapshot) IsAbnormal() bool {
	return s.CPUPercent > 90 || s.MemUsedPct > 95 || s.SwapUsedMB > 0 || s.LoadAvg1 > 10
}

// cpuJiffies holds cumulative CPU time counters from /proc/stat.
type cpuJiffies struct {
	user, nice, system, idle, iowait, irq, softirq, steal uint64
}

func (j cpuJiffies) total() uint64 {
	return j.user + j.nice + j.system + j.idle + j.iowait + j.irq + j.softirq + j.steal
}

func (j cpuJiffies) idle64() uint64 {
	return j.idle + j.iowait
}

// Collector reads system metrics from /proc once per second.
type Collector struct {
	mu       sync.RWMutex
	snapshot SystemSnapshot

	// State for delta computation.
	prevCPU        cpuJiffies
	prevPageFaults int64
	initialized    bool

	// For testing: override /proc paths.
	procStat    string
	procLoadavg string
	procMeminfo string
	procVmstat  string

	stopCh chan struct{}
	doneCh chan struct{}
}

// New creates a new system info collector.
func New() *Collector {
	return &Collector{
		procStat:    "/proc/stat",
		procLoadavg: "/proc/loadavg",
		procMeminfo: "/proc/meminfo",
		procVmstat:  "/proc/vmstat",
		stopCh:      make(chan struct{}),
		doneCh:      make(chan struct{}),
	}
}

// Start begins polling /proc once per second in a background goroutine.
func (c *Collector) Start() {
	go c.run()
}

// Stop halts the background polling goroutine.
func (c *Collector) Stop() {
	close(c.stopCh)
	<-c.doneCh
}

// Snapshot returns the latest system metrics reading.
func (c *Collector) Snapshot() SystemSnapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.snapshot
}

func (c *Collector) run() {
	defer close(c.doneCh)

	// Initial read.
	c.poll()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-c.stopCh:
			return
		case <-ticker.C:
			c.poll()
		}
	}
}

// poll reads all /proc sources and updates the snapshot.
func (c *Collector) poll() {
	snap := SystemSnapshot{Timestamp: time.Now()}

	// CPU utilization from /proc/stat delta.
	if cpu, err := c.readCPU(); err == nil {
		if c.initialized {
			totalDelta := cpu.total() - c.prevCPU.total()
			idleDelta := cpu.idle64() - c.prevCPU.idle64()
			if totalDelta > 0 {
				snap.CPUPercent = 100.0 * float64(totalDelta-idleDelta) / float64(totalDelta)
			}
		}
		c.prevCPU = cpu
	}

	// Memory from /proc/meminfo.
	if mem, err := c.readMeminfo(); err == nil {
		snap.MemTotalMB = mem.totalKB / 1024
		snap.MemAvailMB = mem.availKB / 1024
		if mem.totalKB > 0 {
			snap.MemUsedPct = 100.0 * float64(mem.totalKB-mem.availKB) / float64(mem.totalKB)
		}
		swapUsedKB := mem.swapTotalKB - mem.swapFreeKB
		if swapUsedKB > 0 {
			snap.SwapUsedMB = swapUsedKB / 1024
		}
	}

	// Load averages from /proc/loadavg.
	if l1, l5, err := c.readLoadavg(); err == nil {
		snap.LoadAvg1 = l1
		snap.LoadAvg5 = l5
	}

	// Page faults from /proc/vmstat.
	if pgmajfault, err := c.readPageFaults(); err == nil {
		if c.initialized {
			snap.PageFaults = pgmajfault - c.prevPageFaults
		}
		c.prevPageFaults = pgmajfault
	}

	c.initialized = true

	c.mu.Lock()
	c.snapshot = snap
	c.mu.Unlock()
}

// readCPU parses /proc/stat for the aggregate CPU line.
func (c *Collector) readCPU() (cpuJiffies, error) {
	f, err := os.Open(c.procStat)
	if err != nil {
		return cpuJiffies{}, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "cpu ") {
			return parseCPULine(line)
		}
	}
	return cpuJiffies{}, fmt.Errorf("no cpu line in %s", c.procStat)
}

func parseCPULine(line string) (cpuJiffies, error) {
	fields := strings.Fields(line)
	if len(fields) < 8 {
		return cpuJiffies{}, fmt.Errorf("too few fields in cpu line: %d", len(fields))
	}
	var j cpuJiffies
	var err error
	if j.user, err = strconv.ParseUint(fields[1], 10, 64); err != nil {
		return j, err
	}
	if j.nice, err = strconv.ParseUint(fields[2], 10, 64); err != nil {
		return j, err
	}
	if j.system, err = strconv.ParseUint(fields[3], 10, 64); err != nil {
		return j, err
	}
	if j.idle, err = strconv.ParseUint(fields[4], 10, 64); err != nil {
		return j, err
	}
	if j.iowait, err = strconv.ParseUint(fields[5], 10, 64); err != nil {
		return j, err
	}
	if j.irq, err = strconv.ParseUint(fields[6], 10, 64); err != nil {
		return j, err
	}
	if j.softirq, err = strconv.ParseUint(fields[7], 10, 64); err != nil {
		return j, err
	}
	if len(fields) > 8 {
		j.steal, _ = strconv.ParseUint(fields[8], 10, 64)
	}
	return j, nil
}

// meminfo holds parsed /proc/meminfo values (in KB).
type meminfo struct {
	totalKB     int64
	availKB     int64
	swapTotalKB int64
	swapFreeKB  int64
}

func (c *Collector) readMeminfo() (meminfo, error) {
	f, err := os.Open(c.procMeminfo)
	if err != nil {
		return meminfo{}, err
	}
	defer f.Close()

	var m meminfo
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "MemTotal:"):
			m.totalKB = parseKBValue(line)
		case strings.HasPrefix(line, "MemAvailable:"):
			m.availKB = parseKBValue(line)
		case strings.HasPrefix(line, "SwapTotal:"):
			m.swapTotalKB = parseKBValue(line)
		case strings.HasPrefix(line, "SwapFree:"):
			m.swapFreeKB = parseKBValue(line)
		}
	}
	return m, nil
}

// parseKBValue extracts the integer value from a /proc/meminfo line like "MemTotal:  16384000 kB".
func parseKBValue(line string) int64 {
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return 0
	}
	v, _ := strconv.ParseInt(fields[1], 10, 64)
	return v
}

func (c *Collector) readLoadavg() (float64, float64, error) {
	f, err := os.Open(c.procLoadavg)
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	if !scanner.Scan() {
		return 0, 0, fmt.Errorf("empty %s", c.procLoadavg)
	}
	fields := strings.Fields(scanner.Text())
	if len(fields) < 2 {
		return 0, 0, fmt.Errorf("too few fields in %s", c.procLoadavg)
	}
	l1, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return 0, 0, err
	}
	l5, err := strconv.ParseFloat(fields[1], 64)
	if err != nil {
		return 0, 0, err
	}
	return l1, l5, nil
}

func (c *Collector) readPageFaults() (int64, error) {
	f, err := os.Open(c.procVmstat)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "pgmajfault ") {
			fields := strings.Fields(line)
			if len(fields) < 2 {
				return 0, fmt.Errorf("invalid pgmajfault line")
			}
			return strconv.ParseInt(fields[1], 10, 64)
		}
	}
	return 0, fmt.Errorf("pgmajfault not found in %s", c.procVmstat)
}

// ReadOnce performs a single poll and returns the snapshot.
// Useful for testing or one-shot reads. Requires two calls to get CPU delta.
func (c *Collector) ReadOnce() SystemSnapshot {
	c.poll()
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.snapshot
}
