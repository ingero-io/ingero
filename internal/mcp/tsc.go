// Package mcp — tsc.go implements Telegraphic Compression (TSC) for MCP responses.
//
// TSC reduces MCP response token count by ~60% for AI agent consumption.
// Field names are abbreviated: "timestamp" → "t", "duration_us" → "d_us", etc.
// Enabled by default (--tsc flag, default: true). Per-request override: {"tsc": false}.
//
// Call chain: MCP tool handlers → formatEventList/formatStatsSnapshot →
//   tscFieldMap applied if tsc=true → compact JSON response
package mcp

// tscFieldMap maps verbose field names to abbreviated TSC equivalents.
var tscFieldMap = map[string]string{
	"timestamp":       "t",
	"pid":             "p",
	"tid":             "tid",
	"operation":       "op",
	"duration_us":     "d_us",
	"flags":           "f",
	"count":           "n",
	"p50_us":          "p50",
	"p95_us":          "p95",
	"p99_us":          "p99",
	"time_fraction":   "tf",
	"correlated":      "cor",
	"sched_switch":    "ss",
	"off_cpu_us":      "off",
	"severity":        "sev",
	"summary":         "sum",
	"root_cause":      "rc",
	"recommendations": "rec",
	"layer":           "l",
	"detail":          "det",
	"cpu_percent":     "cpu",
	"mem_percent":     "mem",
	"swap_mb":         "swp",
	"load_avg":        "la",
	"anomaly_count":   "an",
	"wall_percent":    "w%",
	"command":         "cmd",
}

// tscReverseMap maps abbreviated names back to verbose names.
var tscReverseMap map[string]string

func init() {
	tscReverseMap = make(map[string]string, len(tscFieldMap))
	for k, v := range tscFieldMap {
		tscReverseMap[v] = k
	}
}

// TSCKey returns the TSC-abbreviated key if tsc is true, otherwise the original key.
func TSCKey(key string, tsc bool) string {
	if !tsc {
		return key
	}
	if abbr, ok := tscFieldMap[key]; ok {
		return abbr
	}
	return key
}

// TSCMap creates a map with keys abbreviated according to TSC mode.
// Usage: TSCMap(true, "timestamp", "15:41:22", "pid", 4821, ...)
// Returns: {"t": "15:41:22", "p": 4821, ...}
func TSCMap(tsc bool, kvs ...interface{}) map[string]interface{} {
	m := make(map[string]interface{}, len(kvs)/2)
	for i := 0; i+1 < len(kvs); i += 2 {
		key, ok := kvs[i].(string)
		if !ok {
			continue
		}
		m[TSCKey(key, tsc)] = kvs[i+1]
	}
	return m
}
