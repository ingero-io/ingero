package dashboard

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/pkg/events"
)

// parseSince extracts a "since" query parameter as a Go duration.
// Returns 0 (all data) if the parameter is missing or empty.
func parseSince(r *http.Request) (time.Duration, error) {
	s := r.URL.Query().Get("since")
	if s == "" {
		return 0, nil
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return 0, fmt.Errorf("invalid since %q: %w", s, err)
	}
	return d, nil
}

// writeJSON marshals v as JSON and writes it to w.
func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// writeError sends a JSON error response with the given status code.
func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// overviewResponse is the JSON structure for GET /api/v1/overview.
type overviewResponse struct {
	EventCount  int64              `json:"event_count"`
	ChainCount  int                `json:"chain_count"`
	SessionCount int               `json:"session_count"`
	System      *systemOverview    `json:"system,omitempty"`
	TopChain    *chainSummary      `json:"top_chain,omitempty"`
	GPU         *gpuInfo           `json:"gpu,omitempty"`
}

type gpuInfo struct {
	Model         string `json:"model,omitempty"`
	DriverVersion string `json:"driver_version,omitempty"`
	CUDAVersion   string `json:"cuda_version,omitempty"`
}

type systemOverview struct {
	CPUPercent float64 `json:"cpu_percent"`
	MemPercent float64 `json:"mem_percent"`
	SwapMB     int64   `json:"swap_mb"`
	LoadAvg    float64 `json:"load_avg"`
}

type chainSummary struct {
	Severity        string   `json:"severity"`
	Summary         string   `json:"summary"`
	RootCause       string   `json:"root_cause"`
	Recommendations []string `json:"recommendations,omitempty"`
}

// handleOverview returns a high-level summary: event count, chain count,
// latest system snapshot, and the top causal chain.
func (s *Server) handleOverview(w http.ResponseWriter, r *http.Request) {
	if s.store == nil {
		writeJSON(w, overviewResponse{})
		return
	}

	resp := overviewResponse{
		GPU: s.gpuInfo,
	}

	// Event count.
	if count, err := s.store.Count(); err == nil {
		resp.EventCount = count
	}

	// Causal chains.
	chains, err := s.store.QueryChains(0)
	if err == nil {
		resp.ChainCount = len(chains)
		if len(chains) > 0 {
			top := chains[0] // most recent (ordered DESC)
			resp.TopChain = &chainSummary{
				Severity:        top.Severity,
				Summary:         top.Summary,
				RootCause:       top.RootCause,
				Recommendations: top.Recommendations,
			}
		}
	}

	// Sessions.
	sessions, err := s.store.QuerySessions(0)
	if err == nil {
		resp.SessionCount = len(sessions)
	}

	// Latest system snapshot.
	snapshots, err := s.store.QuerySnapshots(store.QueryParams{Since: 60 * time.Second})
	if err == nil && len(snapshots) > 0 {
		latest := snapshots[len(snapshots)-1]
		resp.System = &systemOverview{
			CPUPercent: latest.CPUPercent,
			MemPercent: latest.MemUsedPct,
			SwapMB:     latest.SwapUsedMB,
			LoadAvg:    latest.LoadAvg1,
		}
	}

	writeJSON(w, resp)
}

// opStatsResponse is the JSON structure for GET /api/v1/ops.
type opStatsResponse struct {
	Mode   string          `json:"mode"` // "percentile" or "aggregate"
	Ops    []opStatsEntry  `json:"ops"`
	System *systemOverview `json:"system,omitempty"`
}

type opStatsEntry struct {
	Operation string `json:"operation"`
	Source    string `json:"source"`
	Count    int64  `json:"count"`
	P50US    int64  `json:"p50_us,omitempty"`
	P95US    int64  `json:"p95_us,omitempty"`
	P99US    int64  `json:"p99_us,omitempty"`
	AvgUS    int64  `json:"avg_us,omitempty"`
	MinUS    int64  `json:"min_us,omitempty"`
	MaxUS    int64  `json:"max_us,omitempty"`
}

// handleOps returns per-operation latency statistics.
// Uses percentiles for small DBs, aggregates for large DBs.
func (s *Server) handleOps(w http.ResponseWriter, r *http.Request) {
	since, err := parseSince(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	if s.store == nil {
		writeJSON(w, opStatsResponse{Mode: "empty", Ops: []opStatsEntry{}})
		return
	}

	qparams := store.QueryParams{Since: since, Limit: -1}

	// Check event count to decide path.
	count, countErr := s.store.Count()
	if countErr != nil || count > 500_000 {
		// Large DB: aggregate path.
		ops, err := s.store.QueryAggregatePerOp(qparams)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		entries := make([]opStatsEntry, 0, len(ops))
		for _, op := range ops {
			avgUS := int64(0)
			if op.Count > 0 {
				avgUS = (op.SumDur / op.Count) / 1000
			}
			entries = append(entries, opStatsEntry{
				Operation: op.OpName,
				Source:    sourceString(op.Source),
				Count:    op.Count,
				AvgUS:    avgUS,
				MinUS:    op.MinDur / 1000,
				MaxUS:    op.MaxDur / 1000,
			})
		}
		writeJSON(w, opStatsResponse{Mode: "aggregate", Ops: entries})
		return
	}

	// Small DB: full-fidelity path.
	evts, err := s.store.Query(qparams)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	resp := opStatsResponse{Mode: "percentile", Ops: []opStatsEntry{}}

	if len(evts) > 0 {
		collector := stats.New()
		for _, evt := range evts {
			collector.Record(evt)
		}
		snap := collector.Snapshot()
		for _, op := range snap.Ops {
			resp.Ops = append(resp.Ops, opStatsEntry{
				Operation: op.Op,
				Source:    op.Source.String(),
				Count:    int64(op.Count),
				P50US:    op.P50.Microseconds(),
				P95US:    op.P95.Microseconds(),
				P99US:    op.P99.Microseconds(),
			})
		}
		if snap.System != nil {
			resp.System = &systemOverview{
				CPUPercent: snap.System.CPUPercent,
				MemPercent: snap.System.MemUsedPct,
				SwapMB:     snap.System.SwapUsedMB,
				LoadAvg:    snap.System.LoadAvg1,
			}
		}
	}

	writeJSON(w, resp)
}

// chainResponse is the JSON structure for GET /api/v1/chains.
type chainResponse struct {
	Chains []chainEntry `json:"chains"`
}

type chainEntry struct {
	ID              string           `json:"id"`
	DetectedAt      string           `json:"detected_at"`
	Severity        string           `json:"severity"`
	Summary         string           `json:"summary"`
	RootCause       string           `json:"root_cause"`
	Explanation     string           `json:"explanation"`
	Recommendations []string         `json:"recommendations,omitempty"`
	CUDAOp          string           `json:"cuda_op,omitempty"`
	CUDAP99US       int64            `json:"cuda_p99_us,omitempty"`
	CUDAP50US       int64            `json:"cuda_p50_us,omitempty"`
	TailRatio       float64          `json:"tail_ratio,omitempty"`
	Timeline        []timelineEntry  `json:"timeline,omitempty"`
}

type timelineEntry struct {
	Layer  string `json:"layer"`
	Detail string `json:"detail"`
}

// handleChains returns stored causal chains.
func (s *Server) handleChains(w http.ResponseWriter, r *http.Request) {
	if s.store == nil {
		writeJSON(w, chainResponse{Chains: []chainEntry{}})
		return
	}

	since, err := parseSince(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	stored, err := s.store.QueryChains(since)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	entries := make([]chainEntry, 0, len(stored))
	for _, ch := range stored {
		e := chainEntry{
			ID:              ch.ID,
			DetectedAt:      ch.DetectedAt.Format(time.RFC3339),
			Severity:        ch.Severity,
			Summary:         ch.Summary,
			RootCause:       ch.RootCause,
			Explanation:     ch.Explanation,
			Recommendations: ch.Recommendations,
			CUDAOp:          ch.CUDAOp,
			CUDAP99US:       ch.CUDAP99us,
			CUDAP50US:       ch.CUDAP50us,
			TailRatio:       ch.TailRatio,
		}
		for _, te := range ch.Timeline {
			e.Timeline = append(e.Timeline, timelineEntry{
				Layer:  te.Layer,
				Detail: te.Detail,
			})
		}
		entries = append(entries, e)
	}

	writeJSON(w, chainResponse{Chains: entries})
}

// snapshotResponse is the JSON structure for GET /api/v1/snapshots.
type snapshotResponse struct {
	Snapshots []snapshotEntry `json:"snapshots"`
}

type snapshotEntry struct {
	Timestamp  string  `json:"timestamp"`
	CPUPercent float64 `json:"cpu_percent"`
	MemPercent float64 `json:"mem_percent"`
	MemAvailMB int64   `json:"mem_avail_mb"`
	SwapMB     int64   `json:"swap_mb"`
	LoadAvg    float64 `json:"load_avg"`
}

// handleSnapshots returns system metric time series.
func (s *Server) handleSnapshots(w http.ResponseWriter, r *http.Request) {
	if s.store == nil {
		writeJSON(w, snapshotResponse{Snapshots: []snapshotEntry{}})
		return
	}

	since, err := parseSince(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if since == 0 {
		since = 60 * time.Second // default: last 60s
	}

	snapshots, err := s.store.QuerySnapshots(store.QueryParams{Since: since})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	entries := make([]snapshotEntry, 0, len(snapshots))
	for _, snap := range snapshots {
		entries = append(entries, snapshotEntry{
			Timestamp:  snap.Timestamp.Format(time.RFC3339),
			CPUPercent: snap.CPUPercent,
			MemPercent: snap.MemUsedPct,
			MemAvailMB: snap.MemAvailMB,
			SwapMB:     snap.SwapUsedMB,
			LoadAvg:    snap.LoadAvg1,
		})
	}

	writeJSON(w, snapshotResponse{Snapshots: entries})
}

// handleCapabilities returns the metric availability manifest.
func (s *Server) handleCapabilities(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, Capabilities())
}

// sourceString converts a source ID to a human-readable string.
func sourceString(src uint8) string {
	switch events.Source(src) {
	case events.SourceCUDA:
		return "CUDA"
	case events.SourceHost:
		return "Host"
	case events.SourceDriver:
		return "Driver"
	case events.SourceIO:
		return "IO"
	case events.SourceTCP:
		return "TCP"
	case events.SourceNet:
		return "Net"
	default:
		return fmt.Sprintf("src_%d", src)
	}
}
