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
)

// OTLP straggler-event data-point attribute keys (Story 3.4).
const (
	AttrThreshold      = "threshold"
	AttrScore          = "score"
	AttrDominantSignal = "dominant_signal"
)

// OTLP resource attribute keys (stable per agent, identify the source).
const (
	AttrNodeID    = "ingero.node.id"
	AttrClusterID = "ingero.cluster.id"
)

// OTLP data point attribute keys (per-push, can change).
const (
	AttrNodeState     = "ingero.node.state"
	AttrWorkloadType  = "ingero.workload_type"
	AttrWorldSize     = "ingero.world_size"
	AttrNodeRank      = "ingero.node.rank"
	AttrDetectionMode = "mode"
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
