package fleet

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// ClockSkewResult represents the estimated clock offset between two nodes.
type ClockSkewResult struct {
	NodeA    string
	NodeB    string
	OffsetMs float64 // positive = NodeB ahead of NodeA
	RTTMs    float64 // round-trip time of the measurement
}

// String returns a human-readable clock skew description.
func (r ClockSkewResult) String() string {
	dir := "ahead of"
	offset := r.OffsetMs
	if offset < 0 {
		dir = "behind"
		offset = -offset
	}
	return fmt.Sprintf("%s is ~%.0fms %s %s (RTT: %.0fms)", r.NodeB, offset, dir, r.NodeA, r.RTTMs)
}

// timeResponse is the JSON from /api/v1/time.
type timeResponse struct {
	TimestampNS int64 `json:"timestamp_ns"`
}

// nodeOffset holds the measured offset of one node relative to the local clock.
type nodeOffset struct {
	node     string
	offsetNs int64   // remote - local midpoint
	rttNs    int64
	err      error
}

// EstimateClockSkew estimates pairwise clock offsets across all configured nodes.
// Uses NTP-style one-shot estimation with 3 samples per node (median).
// The querying machine's clock is used as the reference.
func (c *Client) EstimateClockSkew(ctx context.Context) ([]ClockSkewResult, error) {
	offsets := make([]nodeOffset, len(c.nodes))
	var wg sync.WaitGroup

	for i, node := range c.nodes {
		wg.Add(1)
		go func(idx int, addr string) {
			defer wg.Done()
			offsets[idx] = c.measureOffset(ctx, addr)
		}(i, node)
	}
	wg.Wait()

	// Compute pairwise offsets.
	var results []ClockSkewResult
	for i := 0; i < len(offsets); i++ {
		if offsets[i].err != nil {
			continue
		}
		for j := i + 1; j < len(offsets); j++ {
			if offsets[j].err != nil {
				continue
			}
			// offset[j] - offset[i] = how much j is ahead of i
			pairOffset := offsets[j].offsetNs - offsets[i].offsetNs
			maxRTT := offsets[i].rttNs
			if offsets[j].rttNs > maxRTT {
				maxRTT = offsets[j].rttNs
			}
			results = append(results, ClockSkewResult{
				NodeA:    offsets[i].node,
				NodeB:    offsets[j].node,
				OffsetMs: float64(pairOffset) / 1e6,
				RTTMs:    float64(maxRTT) / 1e6,
			})
		}
	}

	return results, nil
}

// measureOffset takes 3 samples of the clock offset for a single node
// and returns the median.
func (c *Client) measureOffset(ctx context.Context, addr string) nodeOffset {
	const samples = 3
	var offsets []int64
	var rtts []int64

	for i := 0; i < samples; i++ {
		off, rtt, err := c.sampleOffset(ctx, addr)
		if err != nil {
			return nodeOffset{node: addr, err: err}
		}
		offsets = append(offsets, off)
		rtts = append(rtts, rtt)
	}

	sort.Slice(offsets, func(i, j int) bool { return offsets[i] < offsets[j] })
	sort.Slice(rtts, func(i, j int) bool { return rtts[i] < rtts[j] })

	return nodeOffset{
		node:     addr,
		offsetNs: offsets[samples/2], // median
		rttNs:    rtts[samples/2],
	}
}

// sampleOffset takes a single NTP-style clock offset measurement.
func (c *Client) sampleOffset(ctx context.Context, addr string) (offsetNs, rttNs int64, err error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	url := fmt.Sprintf("%s://%s/api/v1/time", c.scheme, addr)
	t1 := time.Now().UnixNano()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return 0, 0, err
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()

	t2 := time.Now().UnixNano()

	if resp.StatusCode != http.StatusOK {
		return 0, 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	var tr timeResponse
	if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
		return 0, 0, err
	}

	rtt := t2 - t1
	localMidpoint := t1 + rtt/2
	offset := tr.TimestampNS - localMidpoint

	return offset, rtt, nil
}

// FormatClockSkewWarnings returns warning strings for offsets exceeding the threshold.
func FormatClockSkewWarnings(results []ClockSkewResult, thresholdMs float64) []string {
	var warnings []string
	for _, r := range results {
		if math.Abs(r.OffsetMs) > thresholdMs {
			warnings = append(warnings, "WARNING: "+r.String())
		}
	}
	return warnings
}

// PrintClockSkewWarnings prints warnings to stderr-style output string.
func PrintClockSkewWarnings(results []ClockSkewResult, thresholdMs float64) string {
	warnings := FormatClockSkewWarnings(results, thresholdMs)
	if len(warnings) == 0 {
		return ""
	}
	return strings.Join(warnings, "\n") + "\n"
}
