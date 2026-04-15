package contract

import (
	"strings"
	"testing"
)

func TestMetricNames_NotEmpty(t *testing.T) {
	metrics := []struct {
		name  string
		value string
	}{
		{"MetricHealthScore", MetricHealthScore},
		{"MetricThroughputRatio", MetricThroughputRatio},
		{"MetricComputeEfficiency", MetricComputeEfficiency},
		{"MetricMemoryHeadroom", MetricMemoryHeadroom},
		{"MetricCPUAvailability", MetricCPUAvailability},
		{"MetricDegradationWarning", MetricDegradationWarning},
		{"MetricDetectionMode", MetricDetectionMode},
		{"MetricFleetReachable", MetricFleetReachable},
	}

	for _, m := range metrics {
		if m.value == "" {
			t.Errorf("%s is empty", m.name)
		}
		if !strings.HasPrefix(m.value, "ingero.") {
			t.Errorf("%s = %q: expected 'ingero.' prefix", m.name, m.value)
		}
	}
}

func TestAttributes_NotEmpty(t *testing.T) {
	attrs := []struct {
		name  string
		value string
	}{
		{"AttrNodeID", AttrNodeID},
		{"AttrClusterID", AttrClusterID},
		{"AttrNodeState", AttrNodeState},
		{"AttrWorkloadType", AttrWorkloadType},
		{"AttrWorldSize", AttrWorldSize},
		{"AttrNodeRank", AttrNodeRank},
	}

	for _, a := range attrs {
		if a.value == "" {
			t.Errorf("%s is empty", a.name)
		}
		if !strings.HasPrefix(a.value, "ingero.") {
			t.Errorf("%s = %q: expected 'ingero.' prefix", a.name, a.value)
		}
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
