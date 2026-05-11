package contract

import (
	"go/ast"
	"go/parser"
	"go/token"
	"sort"
	"strings"
	"testing"
)

// pinnedMetrics is the canonical (constant_name -> wire_value) pinning for
// every Metric* exported constant. New entries here MUST land in the same
// PR that adds the constant; TestExportedConstants_HavePinnedTestEntry
// will fail otherwise.
//
// Allowed prefixes: "ingero." (agent-defined), "gen_ai." (OTel GenAI
// semconv), "gpu." (sensor-surface family), "nccl." (NCCL collective
// family). Drift in either name or value breaks the Fleet consumer.
var pinnedMetrics = map[string]string{
	"MetricHealthScore":                   "ingero.node.health_score",
	"MetricThroughputRatio":               "ingero.node.throughput_ratio",
	"MetricComputeEfficiency":             "ingero.node.compute_efficiency",
	"MetricMemoryHeadroom":                "ingero.node.memory_headroom",
	"MetricCPUAvailability":               "ingero.node.cpu_availability",
	"MetricDegradationWarning":            "ingero.node.degradation_warning",
	"MetricDetectionMode":                 "ingero.agent.detection_mode",
	"MetricFleetReachable":                "ingero.agent.fleet_reachable",
	"MetricStragglerEvent":                "ingero.node.straggler_event",
	"MetricNodeInfo":                      "ingero.node.info",
	"MetricNodeWorldSize":                 "ingero.node.world_size",
	"MetricCUDAKernelLaunchTotal":         "ingero.node.cuda_kernel_launch_total",
	"MetricCUDAMemcpyBytesTotal":          "ingero.node.cuda_memcpy_bytes_total",
	"MetricCPUStallNanosTotal":            "ingero.node.cpu_stall_nanos_total",
	"MetricGPUNCCLProcessLoaded":          "gpu.nccl.process_loaded",
	"MetricGPUNCCLProcessesTotal":         "gpu.nccl.processes_total",
	"MetricGPUMemoryFragmentation":        "gpu.memory.fragmentation_estimate",
	"MetricGPUMemoryProcessAllocated":     "gpu.memory.process.allocated_bytes",
	"MetricGPUMemcpyBytesTotal":           "gpu.memcpy.bytes_total",
	"MetricGPUMemcpyDurationMS":           "gpu.memcpy.duration_ms",
	"MetricInferStepDurationNs":           "ingero.infer.step_duration_ns",
	"MetricInferOutlierTotal":             "ingero.infer.outlier_total",
	"MetricInferBaselineMeanNs":           "ingero.infer.baseline_mean_ns",
	"MetricInferBaselineP95Ns":            "ingero.infer.baseline_p95_ns",
	"MetricInferWorkloadsTracked":         "ingero.infer.workloads_tracked",
	"MetricInferSamplerDegraded":          "ingero.infer.sampler.degraded",
	"MetricInferSamplerDegradationsTotal": "ingero.infer.sampler.degradations_total",
	"MetricInferThrottleActiveTotal":      "ingero.infer.throttle_active_total",
	"MetricInferKVCacheAllocAgeMs":        "ingero.infer.kvcache.alloc_age_ms",
	"MetricGenAITTFT":                     "gen_ai.client.operation.time_to_first_token",
	"MetricGenAITPOT":                     "gen_ai.server.time_per_output_token",
	"MetricGenAIRequestDuration":          "gen_ai.client.operation.duration",
	"MetricGenAIPrefillDuration":          "gen_ai.server.request.duration.prefill",
	"MetricGenAIDecodeDuration":           "gen_ai.server.request.duration.decode",
	"MetricGenAITokenUsage":               "gen_ai.client.token.usage",
	"MetricNCCLDuration":                  "nccl.collective.duration_ms",
	"MetricNCCLBarrierWait":               "nccl.collective.barrier_wait_ms",
	"MetricNCCLPeerLag":                   "nccl.collective.peer_lag_ms",
	"MetricNCCLBytes":                     "nccl.collective.bytes",
}

// pinnedAttrs is the canonical (constant_name -> wire_value) pinning for
// every Attr* exported constant.
//
// Most attrs use "ingero.", "gen_ai.", or "nccl." prefixes. The straggler
// event attrs (AttrThreshold, AttrScore, AttrDominantSignal) intentionally
// publish bare names because they piggy-back as data-point attributes on
// metrics whose own metric name already carries the namespace.
var pinnedAttrs = map[string]string{
	"AttrGenAISystem":            "gen_ai.system",
	"AttrGenAIRequestModel":      "gen_ai.request.model",
	"AttrGenAIResponseModel":     "gen_ai.response.model",
	"AttrGenAIOperationName":     "gen_ai.operation.name",
	"AttrIngeroEnginePID":        "ingero.engine.pid",
	"AttrThreshold":              "threshold",
	"AttrScore":                  "score",
	"AttrDominantSignal":         "dominant_signal",
	"AttrEventID":                "ingero.event.id",
	"AttrNodeID":                 "ingero.node.id",
	"AttrClusterID":              "ingero.cluster.id",
	"AttrProvider":               "ingero.provider",
	"AttrGPUModel":               "ingero.gpu_model",
	"AttrGPUCount":               "ingero.gpu_count",
	"AttrCgroupPathHash":         "ingero.cgroup_path_hash",
	"AttrMemcpyDirection":        "ingero.memcpy.direction",
	"AttrInferStreamHandle":      "ingero.infer.stream_handle",
	"AttrInferOutlierBucket":     "ingero.infer.outlier_bucket",
	"AttrInferPhase":             "ingero.infer.phase",
	"AttrInferSamplerCause":      "ingero.infer.sampler.cause",
	"AttrInferKernelFingerprint": "ingero.infer.kernel_fingerprint",
	"AttrNodeState":              "ingero.node.state",
	"AttrWorkloadType":           "ingero.workload_type",
	"AttrWorldSize":              "ingero.world_size",
	"AttrNodeRank":               "ingero.node.rank",
	"AttrDetectionMode":          "ingero.detection_mode",
	"AttrNCCLOpType":                   "nccl.op_type",
	"AttrNCCLCommIDHash":               "nccl.comm_id_hash",
	"AttrNCCLRank":                     "nccl.rank",
	"AttrNCCLNRanks":                   "nccl.nranks",
	"AttrNCCLDatatype":                 "nccl.datatype",
	"AttrNCCLReduceOp":                 "nccl.reduce_op",
	"AttrNCCLPeerRank":                 "nccl.peer_rank",
	"AttrInferStepDurationNs":          "ingero.infer.step_duration_ns",
	"AttrInferBaselineP95Ns":           "ingero.infer.baseline_p95_ns",
	"AttrInferBaselineMeanNs":          "ingero.infer.baseline_mean_ns",
	"AttrInferMemfragEventsInStep":     "ingero.infer.memfrag_events_in_step",
	"AttrInferThrottleReasons":         "ingero.infer.throttle_reasons",
	"AttrInferKVCacheOldestAllocAgeMs": "ingero.infer.kvcache.oldest_alloc_age_ms",
	"AttrPID":                          "pid",
	"AttrComm":                         "comm",
	"AttrLibNCCLPath":                  "libnccl_path",
	"AttrLibNCCLVersion":               "libnccl_version",
	"AttrGPUUUID":                      "gpu.uuid",
	"AttrSource":                       "source",
	"AttrOperation":                    "operation",
	"AttrPercentile":                   "percentile",
}

// constantValues looks up the runtime value of every Metric*/Attr* constant.
// Hand-maintained alongside pinnedMetrics/pinnedAttrs because Go's
// reflection cannot see untyped string constants from the same package.
// TestExportedConstants_HavePinnedTestEntry guards against forgetting an
// entry here when adding a new constant.
var constantValues = map[string]string{
	"MetricHealthScore":                   MetricHealthScore,
	"MetricThroughputRatio":               MetricThroughputRatio,
	"MetricComputeEfficiency":             MetricComputeEfficiency,
	"MetricMemoryHeadroom":                MetricMemoryHeadroom,
	"MetricCPUAvailability":               MetricCPUAvailability,
	"MetricDegradationWarning":            MetricDegradationWarning,
	"MetricDetectionMode":                 MetricDetectionMode,
	"MetricFleetReachable":                MetricFleetReachable,
	"MetricStragglerEvent":                MetricStragglerEvent,
	"MetricNodeInfo":                      MetricNodeInfo,
	"MetricNodeWorldSize":                 MetricNodeWorldSize,
	"MetricCUDAKernelLaunchTotal":         MetricCUDAKernelLaunchTotal,
	"MetricCUDAMemcpyBytesTotal":          MetricCUDAMemcpyBytesTotal,
	"MetricCPUStallNanosTotal":            MetricCPUStallNanosTotal,
	"MetricGPUNCCLProcessLoaded":          MetricGPUNCCLProcessLoaded,
	"MetricGPUNCCLProcessesTotal":         MetricGPUNCCLProcessesTotal,
	"MetricGPUMemoryFragmentation":        MetricGPUMemoryFragmentation,
	"MetricGPUMemoryProcessAllocated":     MetricGPUMemoryProcessAllocated,
	"MetricGPUMemcpyBytesTotal":           MetricGPUMemcpyBytesTotal,
	"MetricGPUMemcpyDurationMS":           MetricGPUMemcpyDurationMS,
	"MetricInferStepDurationNs":           MetricInferStepDurationNs,
	"MetricInferOutlierTotal":             MetricInferOutlierTotal,
	"MetricInferBaselineMeanNs":           MetricInferBaselineMeanNs,
	"MetricInferBaselineP95Ns":            MetricInferBaselineP95Ns,
	"MetricInferWorkloadsTracked":         MetricInferWorkloadsTracked,
	"MetricInferSamplerDegraded":          MetricInferSamplerDegraded,
	"MetricInferSamplerDegradationsTotal": MetricInferSamplerDegradationsTotal,
	"MetricInferThrottleActiveTotal":      MetricInferThrottleActiveTotal,
	"MetricInferKVCacheAllocAgeMs":        MetricInferKVCacheAllocAgeMs,
	"MetricGenAITTFT":                     MetricGenAITTFT,
	"MetricGenAITPOT":                     MetricGenAITPOT,
	"MetricGenAIRequestDuration":          MetricGenAIRequestDuration,
	"MetricGenAIPrefillDuration":          MetricGenAIPrefillDuration,
	"MetricGenAIDecodeDuration":           MetricGenAIDecodeDuration,
	"MetricGenAITokenUsage":               MetricGenAITokenUsage,
	"MetricNCCLDuration":                  MetricNCCLDuration,
	"MetricNCCLBarrierWait":               MetricNCCLBarrierWait,
	"MetricNCCLPeerLag":                   MetricNCCLPeerLag,
	"MetricNCCLBytes":                     MetricNCCLBytes,
	"AttrGenAISystem":                     AttrGenAISystem,
	"AttrGenAIRequestModel":               AttrGenAIRequestModel,
	"AttrGenAIResponseModel":              AttrGenAIResponseModel,
	"AttrGenAIOperationName":              AttrGenAIOperationName,
	"AttrIngeroEnginePID":                 AttrIngeroEnginePID,
	"AttrThreshold":                       AttrThreshold,
	"AttrScore":                           AttrScore,
	"AttrDominantSignal":                  AttrDominantSignal,
	"AttrEventID":                         AttrEventID,
	"AttrNodeID":                          AttrNodeID,
	"AttrClusterID":                       AttrClusterID,
	"AttrProvider":                        AttrProvider,
	"AttrGPUModel":                        AttrGPUModel,
	"AttrGPUCount":                        AttrGPUCount,
	"AttrCgroupPathHash":                  AttrCgroupPathHash,
	"AttrMemcpyDirection":                 AttrMemcpyDirection,
	"AttrInferStreamHandle":               AttrInferStreamHandle,
	"AttrInferOutlierBucket":              AttrInferOutlierBucket,
	"AttrInferPhase":                      AttrInferPhase,
	"AttrInferSamplerCause":               AttrInferSamplerCause,
	"AttrInferKernelFingerprint":          AttrInferKernelFingerprint,
	"AttrNodeState":                       AttrNodeState,
	"AttrWorkloadType":                    AttrWorkloadType,
	"AttrWorldSize":                       AttrWorldSize,
	"AttrNodeRank":                        AttrNodeRank,
	"AttrDetectionMode":                   AttrDetectionMode,
	"AttrNCCLOpType":                      AttrNCCLOpType,
	"AttrNCCLCommIDHash":                  AttrNCCLCommIDHash,
	"AttrNCCLRank":                        AttrNCCLRank,
	"AttrNCCLNRanks":                      AttrNCCLNRanks,
	"AttrNCCLDatatype":                    AttrNCCLDatatype,
	"AttrNCCLReduceOp":                    AttrNCCLReduceOp,
	"AttrNCCLPeerRank":                    AttrNCCLPeerRank,
	"AttrInferStepDurationNs":             AttrInferStepDurationNs,
	"AttrInferBaselineP95Ns":              AttrInferBaselineP95Ns,
	"AttrInferBaselineMeanNs":             AttrInferBaselineMeanNs,
	"AttrInferMemfragEventsInStep":        AttrInferMemfragEventsInStep,
	"AttrInferThrottleReasons":            AttrInferThrottleReasons,
	"AttrInferKVCacheOldestAllocAgeMs":    AttrInferKVCacheOldestAllocAgeMs,
	"AttrPID":                             AttrPID,
	"AttrComm":                            AttrComm,
	"AttrLibNCCLPath":                     AttrLibNCCLPath,
	"AttrLibNCCLVersion":                  AttrLibNCCLVersion,
	"AttrGPUUUID":                         AttrGPUUUID,
	"AttrSource":                          AttrSource,
	"AttrOperation":                       AttrOperation,
	"AttrPercentile":                      AttrPercentile,
}

// TestAllMetrics_PinnedToCurrentValues asserts every Metric* constant
// matches its pinned value. Drift in either direction (rename, value
// change) breaks the test loudly so the same PR is forced to update Fleet.
func TestAllMetrics_PinnedToCurrentValues(t *testing.T) {
	for name, want := range pinnedMetrics {
		got, ok := constantValues[name]
		if !ok {
			t.Errorf("constant %s referenced in pinnedMetrics is missing from constantValues", name)
			continue
		}
		if got != want {
			t.Errorf("%s = %q, pinned %q; if intentional, update both contract.go and pinnedMetrics together", name, got, want)
		}
		if got == "" {
			t.Errorf("%s is empty", name)
		}
	}
}

// TestAllAttrs_PinnedToCurrentValues asserts every Attr* constant matches
// its pinned value. Same drift-guard semantics as the metrics test.
func TestAllAttrs_PinnedToCurrentValues(t *testing.T) {
	for name, want := range pinnedAttrs {
		got, ok := constantValues[name]
		if !ok {
			t.Errorf("constant %s referenced in pinnedAttrs is missing from constantValues", name)
			continue
		}
		if got != want {
			t.Errorf("%s = %q, pinned %q; if intentional, update both contract.go and pinnedAttrs together", name, got, want)
		}
	}
}

// TestExportedConstants_HavePinnedTestEntry parses contract.go via go/ast
// and asserts every exported constant whose identifier starts with
// "Metric" or "Attr" appears in the pinning maps above. Catches the
// "operator added a constant but forgot to update the test" case that
// previously let inference and GPU sensor metric families ship with no
// drift guard at all.
func TestExportedConstants_HavePinnedTestEntry(t *testing.T) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "contract.go", nil, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse contract.go: %v", err)
	}

	var declared []string
	for _, decl := range f.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.CONST {
			continue
		}
		for _, spec := range gen.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			for _, n := range vs.Names {
				name := n.Name
				if !strings.HasPrefix(name, "Metric") && !strings.HasPrefix(name, "Attr") {
					continue
				}
				declared = append(declared, name)
			}
		}
	}

	sort.Strings(declared)
	var missing []string
	for _, name := range declared {
		_, hasMetric := pinnedMetrics[name]
		_, hasAttr := pinnedAttrs[name]
		if !hasMetric && !hasAttr {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("declared in contract.go but not pinned in tests: %v "+
			"(add to pinnedMetrics or pinnedAttrs AND constantValues)", missing)
	}

	// Also catch the reverse: pinned but not declared (rename / removal).
	declaredSet := make(map[string]struct{}, len(declared))
	for _, name := range declared {
		declaredSet[name] = struct{}{}
	}
	var stale []string
	for name := range pinnedMetrics {
		if _, ok := declaredSet[name]; !ok {
			stale = append(stale, name)
		}
	}
	for name := range pinnedAttrs {
		if _, ok := declaredSet[name]; !ok {
			stale = append(stale, name)
		}
	}
	if len(stale) > 0 {
		sort.Strings(stale)
		t.Fatalf("pinned in tests but no longer declared in contract.go: %v "+
			"(remove from pinnedMetrics/pinnedAttrs/constantValues)", stale)
	}
}

// TestNCCLContract validates the v0.12.0 NCCL metric + attribute names.
// Catches accidental rename / typo across the agent emitter, ncclprobe,
// and Fleet ncclprocessor — drift in any direction silently breaks the
// dashboard and the pkg/contract is the single source of truth.
func TestNCCLContract(t *testing.T) {
	metrics := map[string]string{
		"MetricNCCLDuration":    MetricNCCLDuration,
		"MetricNCCLBarrierWait": MetricNCCLBarrierWait,
		"MetricNCCLPeerLag":     MetricNCCLPeerLag,
		"MetricNCCLBytes":       MetricNCCLBytes,
	}
	for name, val := range metrics {
		if val == "" {
			t.Errorf("%s is empty", name)
		}
		if !strings.HasPrefix(val, "nccl.collective.") {
			t.Errorf("%s = %q: expected 'nccl.collective.' prefix", name, val)
		}
	}
	attrs := map[string]string{
		"AttrNCCLOpType":     AttrNCCLOpType,
		"AttrNCCLCommIDHash": AttrNCCLCommIDHash,
		"AttrNCCLRank":       AttrNCCLRank,
		"AttrNCCLNRanks":     AttrNCCLNRanks,
		"AttrNCCLDatatype":   AttrNCCLDatatype,
		"AttrNCCLReduceOp":   AttrNCCLReduceOp,
	}
	seen := map[string]string{}
	for name, val := range attrs {
		if val == "" {
			t.Errorf("%s is empty", name)
		}
		if !strings.HasPrefix(val, "nccl.") {
			t.Errorf("%s = %q: expected 'nccl.' prefix", name, val)
		}
		if other, dup := seen[val]; dup {
			t.Errorf("attr %q used by both %s and %s", val, name, other)
		}
		seen[val] = name
	}
}

func TestHeaders_NotEmpty(t *testing.T) {
	if HeaderThreshold == "" {
		t.Error("HeaderThreshold is empty")
	}
	if HeaderQuorumMet == "" {
		t.Error("HeaderQuorumMet is empty")
	}
	if GRPCMetaThreshold == "" {
		t.Error("GRPCMetaThreshold is empty")
	}
	if GRPCMetaQuorumMet == "" {
		t.Error("GRPCMetaQuorumMet is empty")
	}
}

func TestGRPCMeta_Lowercase(t *testing.T) {
	if GRPCMetaThreshold != strings.ToLower(GRPCMetaThreshold) {
		t.Errorf("GRPCMetaThreshold = %q: gRPC metadata keys must be lowercase", GRPCMetaThreshold)
	}
	if GRPCMetaQuorumMet != strings.ToLower(GRPCMetaQuorumMet) {
		t.Errorf("GRPCMetaQuorumMet = %q: gRPC metadata keys must be lowercase", GRPCMetaQuorumMet)
	}
}

func TestStates_Valid(t *testing.T) {
	states := []string{StateActive, StateCalibrating, StateIdle}
	for _, s := range states {
		if s == "" {
			t.Error("empty state constant")
		}
	}
	if StateActive == StateCalibrating || StateActive == StateIdle || StateCalibrating == StateIdle {
		t.Error("state constants must be distinct")
	}
}

func TestAPIPath_StartsWithSlash(t *testing.T) {
	if !strings.HasPrefix(ThresholdAPIPath, "/") {
		t.Errorf("ThresholdAPIPath = %q: must start with /", ThresholdAPIPath)
	}
}

// TestOTelGenAISemconv_PinnedToV137 guards against silent drift if
// the OTel GenAI semantic conventions change name on us. v0.16.2
// pins to v1.37 (May 2026 experimental). When OTel GA lands, audit
// each name against
// https://opentelemetry.io/docs/specs/semconv/gen-ai/ and update
// both this test AND the constants together — the test is a
// build-time alarm, not a one-way ratchet.
func TestOTelGenAISemconv_PinnedToV137(t *testing.T) {
	want := map[string]string{
		"MetricGenAITTFT":            "gen_ai.client.operation.time_to_first_token",
		"MetricGenAITPOT":            "gen_ai.server.time_per_output_token",
		"MetricGenAIRequestDuration": "gen_ai.client.operation.duration",
		"MetricGenAIPrefillDuration": "gen_ai.server.request.duration.prefill",
		"MetricGenAIDecodeDuration":  "gen_ai.server.request.duration.decode",
		"MetricGenAITokenUsage":      "gen_ai.client.token.usage",
		"AttrGenAISystem":            "gen_ai.system",
		"AttrGenAIRequestModel":      "gen_ai.request.model",
		"AttrGenAIResponseModel":     "gen_ai.response.model",
		"AttrGenAIOperationName":     "gen_ai.operation.name",
	}
	got := map[string]string{
		"MetricGenAITTFT":            MetricGenAITTFT,
		"MetricGenAITPOT":            MetricGenAITPOT,
		"MetricGenAIRequestDuration": MetricGenAIRequestDuration,
		"MetricGenAIPrefillDuration": MetricGenAIPrefillDuration,
		"MetricGenAIDecodeDuration":  MetricGenAIDecodeDuration,
		"MetricGenAITokenUsage":      MetricGenAITokenUsage,
		"AttrGenAISystem":            AttrGenAISystem,
		"AttrGenAIRequestModel":      AttrGenAIRequestModel,
		"AttrGenAIResponseModel":     AttrGenAIResponseModel,
		"AttrGenAIOperationName":     AttrGenAIOperationName,
	}
	for name, expect := range want {
		if got[name] != expect {
			t.Errorf("OTel GenAI semconv drift: %s = %q, pinned %q; "+
				"audit https://opentelemetry.io/docs/specs/semconv/gen-ai/ "+
				"and update both contract.go and this test together",
				name, got[name], expect)
		}
	}
}
