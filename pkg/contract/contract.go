// Package contract defines the shared constants for the Ingero Fleet interface
// contract. The agent defines what it emits; Fleet conforms to consume it.
//
// This is the single source of truth for metric names, attribute keys, response
// headers, and API paths used in agent-Fleet communication. Fleet imports this
// package to ensure compile-time consistency.
//
// Fleet repo usage:
//
//	import "github.com/ingero-io/ingero/pkg/contract"
//	name := contract.MetricHealthScore
//
// The contract_test.go in the Fleet repo validates these constants against
// docs/interface-contract.md.
package contract

// OTLP metric names.
const (
	MetricHealthScore        = "ingero.node.health_score"
	MetricThroughputRatio    = "ingero.node.throughput_ratio"
	MetricComputeEfficiency  = "ingero.node.compute_efficiency"
	MetricMemoryHeadroom     = "ingero.node.memory_headroom"
	MetricCPUAvailability    = "ingero.node.cpu_availability"
	MetricDegradationWarning = "ingero.node.degradation_warning"
	MetricDetectionMode      = "ingero.agent.detection_mode"
	MetricFleetReachable     = "ingero.agent.fleet_reachable"
	// MetricStragglerEvent is emitted by the agent when its self-
	// classification flips to straggler (value = 1) or recovers from
	// straggler (value = 0, once on the edge). See Story 3.4.
	MetricStragglerEvent = "ingero.node.straggler_event"
	// MetricNodeInfo is a value=1 Int64 gauge that carries the node's
	// hardware identity (gpu_model + gpu_count) as data-point
	// attributes. The cost-of-problem recording-rule layer joins it
	// against operator-supplied gpu_rates.yaml to compute event_cost.
	MetricNodeInfo = "ingero.node.info"
	// MetricNodeWorldSize is the per-job rank count, emitted as an
	// Int64 gauge so the cost-of-problem layer can multiply per-event
	// duration by affected_rank_count without introspecting
	// data-point attributes. Equals zero when the agent is not part
	// of a distributed-training group.
	MetricNodeWorldSize = "ingero.node.world_size"
)

// OTLP straggler-event data-point attribute keys (Story 3.4).
const (
	AttrThreshold      = "threshold"
	AttrScore          = "score"
	AttrDominantSignal = "dominant_signal"
	// AttrEventID is the agent-generated UUIDv4 that uniquely identifies
	// one detection event across both the OTLP push and the UDS NDJSON
	// straggler message. Consumers correlate the two channels by this
	// value. Generated at the agent at first emission. JSON field name on
	// the UDS wire is `event_id`.
	AttrEventID = "ingero.event.id"
)

// OTLP resource attribute keys (stable per agent, identify the source).
const (
	AttrNodeID    = "ingero.node.id"
	AttrClusterID = "ingero.cluster.id"
	// AttrProvider is added at Fleet ingest by the provider-lookup
	// processor, sourced from operator-supplied node_providers.yaml
	// (IP form or k8s node-label form). Free-form, but Lambda / EC2 /
	// GCP / Azure / CoreWeave are the seed values shipped in the
	// example config. Cost recording rules key on this attribute to
	// pick the right per-provider rate from gpu_rates.yaml.
	AttrProvider = "ingero.provider"
)

// AttrGPUModel and AttrGPUCount are data-point attributes on the
// MetricNodeInfo gauge. They identify the hardware class of the node
// for cost-rate lookup; values are sourced from `nvidia-smi --query-gpu=name`
// (GPU model normalized; e.g. "NVIDIA H100 80GB HBM3" -> "h100-80gb")
// or zero / "unknown" when nvidia-smi is unavailable.
const (
	AttrGPUModel = "ingero.gpu_model"
	AttrGPUCount = "ingero.gpu_count"
	// AttrCgroupPathHash is a stable hash of the traced process's
	// cgroup path (typically the K8s pod cgroup hierarchy). Emitted
	// as a data-point attribute on event metrics so cost-of-problem
	// queries can attribute events to a specific tenant pod without
	// leaking the raw path. The hash is SHA256 of the cgroup path,
	// truncated to 16 hex characters.
	AttrCgroupPathHash = "ingero.cgroup_path_hash"
)

// OTLP data point attribute keys (per-push, can change).
const (
	AttrNodeState     = "ingero.node.state"
	AttrWorkloadType  = "ingero.workload_type"
	AttrWorldSize     = "ingero.world_size"
	AttrNodeRank      = "ingero.node.rank"
	AttrDetectionMode = "mode"
)

// NCCL collective metric names. Emitted by the agent's ncclprobe and
// (for derived metrics) by Fleet's ncclprocessor. See bpf/nccl_trace.bpf.c
// for the underlying op set.
const (
	// MetricNCCLDuration is the agent-side queue time of a single
	// collective: uretprobe.exit_ts - uprobe.entry_ts. NOT the barrier
	// wait. Histogram in milliseconds.
	MetricNCCLDuration = "nccl.collective.duration_ms"
	// MetricNCCLBarrierWait is the per-rank time spent blocked at the
	// implicit NCCL barrier, derived Fleet-side by correlating with
	// the next cudaStreamSynchronize on the same (pid, stream_handle).
	MetricNCCLBarrierWait = "nccl.collective.barrier_wait_ms"
	// MetricNCCLPeerLag is per-rank deviation from peer median for the
	// same (cluster_id, comm_id_hash, op_id). Identifies the slowest
	// rank in a comm without needing absolute thresholds.
	MetricNCCLPeerLag = "nccl.collective.peer_lag_ms"
	// MetricNCCLBytes is element-count times datatype-size for a single
	// collective. Histogram (bytes).
	MetricNCCLBytes = "nccl.collective.bytes"
)

// NCCL collective attribute keys, present on every nccl.collective.* metric.
const (
	// AttrNCCLOpType is the collective name (allreduce / allgather /
	// reducescatter / bcast / send / recv). Stable string from the BPF
	// op code; downstream dashboards group by this.
	AttrNCCLOpType = "nccl.op_type"
	// AttrNCCLCommIDHash is a 64-bit hash of the ncclUniqueId taken at
	// ncclCommInitRank time. Same value across all ranks of the same
	// communicator, so Fleet can correlate the same op across the cluster
	// without exposing the full 128-byte unique ID on the wire.
	AttrNCCLCommIDHash = "nccl.comm_id_hash"
	// AttrNCCLRank / AttrNCCLNRanks come from the comm-handle map
	// populated at ncclCommInitRank uretprobe.
	AttrNCCLRank   = "nccl.rank"
	AttrNCCLNRanks = "nccl.nranks"
	// AttrNCCLDatatype maps the ncclDataType_t enum (FLOAT16 = 0, etc.).
	AttrNCCLDatatype = "nccl.datatype"
	// AttrNCCLReduceOp is the ncclRedOp_t enum (SUM / PROD / MIN / MAX
	// / AVG); only meaningful for AllReduce + ReduceScatter.
	AttrNCCLReduceOp = "nccl.reduce_op"
)

// Node states.
const (
	StateActive      = "active"
	StateCalibrating = "calibrating"
	StateIdle        = "idle"
)

// HTTP response header names for threshold piggyback.
const (
	HeaderThreshold = "X-Ingero-Threshold"
	HeaderQuorumMet = "X-Ingero-Quorum-Met"
)

// gRPC response metadata keys (lowercase per gRPC convention).
const (
	GRPCMetaThreshold = "ingero-threshold"
	GRPCMetaQuorumMet = "ingero-quorum-met"
)

// API endpoint path.
const (
	ThresholdAPIPath = "/api/v1/threshold"
)

// Query parameter names.
const (
	ParamClusterID = "cluster_id"
)
