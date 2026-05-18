package annotate

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// ResolveStartTime reads /proc/<pid>/stat and returns field 22, the
// process start time in clock ticks after system boot. It is used to
// turn a bare PID into a full ProcessIncarnation at ingest time when
// the writer supplied a PID but no start time.
//
// Field 22 is positionally tricky: field 2 (comm) is wrapped in
// parentheses and may itself contain spaces and parentheses. The robust
// parse splits at the LAST ')' and counts space-separated fields from
// there: after the last ')' the remaining fields are field 3 onward, so
// field 22 is index 19 in that tail slice.
//
// Returns an error when the process does not exist or /proc is
// unreadable. Callers treat a failure as "unknown start time" and keep
// the PID-only scope.
func ResolveStartTime(pid uint32) (uint64, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0, fmt.Errorf("reading /proc/%d/stat: %w", pid, err)
	}
	return parseStartTimeFromStat(string(data))
}

// parseStartTimeFromStat extracts field 22 (starttime) from the raw
// contents of /proc/<pid>/stat. Split out from ResolveStartTime so the
// parse is unit-testable without a live process.
func parseStartTimeFromStat(stat string) (uint64, error) {
	// The comm field (field 2) is parenthesized and can contain any
	// byte including spaces and ')'. Everything after the LAST ')' is
	// field 3 onward, space-separated and well-behaved.
	lastParen := strings.LastIndexByte(stat, ')')
	if lastParen < 0 || lastParen+1 >= len(stat) {
		return 0, fmt.Errorf("malformed stat line: no comm terminator")
	}
	tail := strings.Fields(stat[lastParen+1:])
	// tail[0] is field 3 (state). Field 22 (starttime) is tail index 19.
	const startTimeTailIndex = 19
	if len(tail) <= startTimeTailIndex {
		return 0, fmt.Errorf("malformed stat line: only %d fields after comm", len(tail))
	}
	v, err := strconv.ParseUint(tail[startTimeTailIndex], 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing starttime field: %w", err)
	}
	return v, nil
}

// ResolveIncarnation turns a bare PID into a ProcessIncarnation by
// reading the process start time. A pid of 0 returns the unscoped
// zero value. When the start time cannot be read (process already
// exited, /proc restricted) the incarnation still carries the PID with
// StartTime 0 so the row is recorded; the join for that row degrades
// to PID + time-window matching.
func ResolveIncarnation(pid uint32) ProcessIncarnation {
	if pid == 0 {
		return ProcessIncarnation{}
	}
	st, err := ResolveStartTime(pid)
	if err != nil {
		return ProcessIncarnation{PID: pid}
	}
	return ProcessIncarnation{PID: pid, StartTime: st}
}
