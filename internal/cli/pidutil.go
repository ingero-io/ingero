package cli

// PID conversion utilities for multi-PID support across trace/explain/query.
//
// These helpers bridge cobra's []int flag type and the internal uint32/map types:
//   - toUint32Slice: []int → []uint32 (for store.QueryParams.PIDs)
//   - singlePIDOrZero: []int → uint32 (for correlator, which takes a single PID where 0 = aggregate)
//   - pidSetFromInts: []int → map[uint32]bool (for event-loop PID filtering, nil = no filter)

// toUint32Slice converts a cobra IntSlice to []uint32, filtering out zeros.
// Returns nil if input is nil or empty (nil = no filter in QueryParams).
func toUint32Slice(pids []int) []uint32 {
	if len(pids) == 0 {
		return nil
	}
	out := make([]uint32, 0, len(pids))
	for _, p := range pids {
		if p > 0 {
			out = append(out, uint32(p))
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// singlePIDOrZero returns the PID if exactly one is specified, else 0 (= aggregate all).
// Used for the correlator which takes a single uint32 PID — 0 means "all processes".
// For multi-PID, events are already pre-filtered, so aggregating (0) is correct.
func singlePIDOrZero(pids []int) uint32 {
	if len(pids) == 1 && pids[0] > 0 {
		return uint32(pids[0])
	}
	return 0
}

// pidSetFromInts builds a PID lookup set for event-loop filtering.
// Returns nil if input is nil or empty — nil means "accept all events" (no filter).
// The caller checks `if pidFilter != nil && !pidFilter[evt.PID]` to skip non-matching events.
func pidSetFromInts(pids []int) map[uint32]bool {
	if len(pids) == 0 {
		return nil
	}
	m := make(map[uint32]bool, len(pids))
	for _, p := range pids {
		if p > 0 {
			m[uint32(p)] = true
		}
	}
	if len(m) == 0 {
		return nil
	}
	return m
}
