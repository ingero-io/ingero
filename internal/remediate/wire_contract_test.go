package remediate

import (
	"reflect"
	"sort"
	"strings"
	"testing"
)

// Stability classifies whether a wire type can gain or lose fields
// between releases.
//
// Stable types carry the strongest contract: once a field is listed as
// required in the Contract below, it may not be removed or renamed until
// the contract version bumps. That guarantee is what lets external
// consumers (orchestrator, straggler-sink, third-party tooling) pin to
// a specific ingero release and know exactly what they will receive.
//
// Experimental types declare the current shape but do NOT bind future
// releases to it — consumers must tolerate additions and removals.
type Stability int

const (
	Stable Stability = iota
	Experimental
)

func (s Stability) String() string {
	switch s {
	case Stable:
		return "stable"
	case Experimental:
		return "experimental"
	}
	return "unknown"
}

// TypeContract describes the JSON field-set expected on the wire for
// one message type, independent of the Go struct that produces it. The
// reflection test below verifies that the struct's json tags cover every
// required field.
type TypeContract struct {
	Name      string
	Stability Stability
	// Required fields MUST appear as json-tagged fields on the struct.
	// Removing one is a contract break.
	Required []string
	// Optional fields are allowed on the wire; callers tolerate their
	// absence. A struct may have a json tag for an optional field without
	// requiring it (the common omitempty pattern).
	Optional []string
	// goType is the reflect.Type of the Go struct that marshals this
	// message. Kept private so external code reads the contract via
	// Contract() below rather than coupling to implementation types.
	goType reflect.Type
}

// Contract is the single source of truth enforced by
// TestWireContract_FieldsDeclared. Changes to this slice are the
// code-review seam for "are we about to break downstream consumers?".
//
// The human-readable counterpart is docs/remediation-protocol.md — keep
// them in sync.
var Contract = []TypeContract{
	{
		Name:      "memory",
		Stability: Stable,
		Required: []string{
			"type", "pid", "gpu_id", "allocated_bytes", "total_vram",
			"utilization_pct", "last_alloc_size", "timestamp_ns",
		},
		Optional: []string{"comm"},
		goType:   reflect.TypeOf(typedMessage{}),
	},
	{
		Name:      "straggle",
		Stability: Stable,
		Required: []string{
			"type", "pid", "throughput_drop_pct", "sched_switch_count",
			"preempting_pids", "timestamp_ns", "sustained",
		},
		Optional: []string{"comm"},
		goType:   reflect.TypeOf(straggleMessage{}),
	},
	{
		Name:      "straggler_state",
		Stability: Experimental,
		Required: []string{
			"type", "node_id", "cluster_id", "score", "threshold",
			"detection_mode", "dominant_signal", "timestamp",
		},
		goType: reflect.TypeOf(fleetStragglerStateMessage{}),
	},
	{
		Name:      "straggler_resolved",
		Stability: Experimental,
		Required: []string{"type", "node_id", "cluster_id", "timestamp"},
		goType:   reflect.TypeOf(fleetStragglerResolvedMessage{}),
	},
}

// TestWireContract_FieldsDeclared fails the build if any type's struct
// has lost a field that Contract marks as required. This is the CI
// guard against accidental wire breaks during refactors. Reflection
// instead of go/ast parsing because we want the check to run on the
// actual in-use types, not a parallel static description.
func TestWireContract_FieldsDeclared(t *testing.T) {
	for _, c := range Contract {
		t.Run(c.Name, func(t *testing.T) {
			present := collectJSONTags(c.goType)

			for _, req := range c.Required {
				if !present[req] {
					t.Errorf(
						"wire contract %q (stability=%s) declares required field %q, but struct %s has no json-tagged field with that name. "+
							"Either add the field back (contract break) or bump the contract version and update the human-readable doc.",
						c.Name, c.Stability, req, c.goType.Name())
				}
			}

			// Surface any struct field whose json tag is neither in
			// Required nor Optional. That's usually fine (new optional
			// field) but makes the undocumented-on-wire surface visible
			// during code review.
			known := make(map[string]bool, len(c.Required)+len(c.Optional))
			for _, f := range c.Required {
				known[f] = true
			}
			for _, f := range c.Optional {
				known[f] = true
			}
			var undocumented []string
			for tag := range present {
				if !known[tag] {
					undocumented = append(undocumented, tag)
				}
			}
			if len(undocumented) > 0 {
				sort.Strings(undocumented)
				t.Logf("note: struct %s has json fields not listed in the contract: %v. Consider adding them to the contract doc.",
					c.goType.Name(), undocumented)
			}
		})
	}
}

// collectJSONTags walks a struct type and returns the set of json field
// names (the value before `,omitempty` and friends) with a bool value
// for fast membership testing. Fields tagged json:"-" are excluded.
func collectJSONTags(t reflect.Type) map[string]bool {
	if t.Kind() != reflect.Struct {
		return nil
	}
	out := make(map[string]bool, t.NumField())
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag, ok := f.Tag.Lookup("json")
		if !ok {
			continue
		}
		name := tag
		if comma := strings.Index(tag, ","); comma >= 0 {
			name = tag[:comma]
		}
		if name == "" || name == "-" {
			continue
		}
		out[name] = true
	}
	return out
}
