package annotate

import (
	"os"
	"testing"
)

func TestParseStartTimeFromStat(t *testing.T) {
	// A realistic /proc/<pid>/stat line. Field 22 (starttime) is 8943215.
	// Built so the comm field contains spaces and a ')' to exercise the
	// last-paren split.
	line := "1234 (weird )comm name) S 1 1234 1234 0 -1 4194304 100 0 0 0 " +
		"5 6 0 0 20 0 1 0 8943215 12345678 999 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 2 0 0\n"
	got, err := parseStartTimeFromStat(line)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if got != 8943215 {
		t.Errorf("starttime = %d, want 8943215", got)
	}
}

func TestParseStartTimeFromStat_SimpleComm(t *testing.T) {
	line := "42 (bash) S 1 42 42 34816 42 4194304 500 0 0 0 " +
		"10 20 0 0 20 0 1 0 777777 9999 100 0 0 0 0 0 0 0 0 0 0 0 0 0 17 0 0 0\n"
	got, err := parseStartTimeFromStat(line)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if got != 777777 {
		t.Errorf("starttime = %d, want 777777", got)
	}
}

func TestParseStartTimeFromStat_Malformed(t *testing.T) {
	for _, bad := range []string{
		"",
		"no parens here",
		"1 (comm) S 1 2 3",     // too few fields after comm
		"1 (comm) S a b c d e", // also too few
	} {
		if _, err := parseStartTimeFromStat(bad); err == nil {
			t.Errorf("expected %q to fail parse", bad)
		}
	}
}

func TestResolveIncarnation_Unscoped(t *testing.T) {
	if (ResolveIncarnation(0)) != (ProcessIncarnation{}) {
		t.Error("pid 0 should resolve to the unscoped zero value")
	}
}

func TestResolveIncarnation_Self(t *testing.T) {
	self := uint32(os.Getpid())
	inc := ResolveIncarnation(self)
	if inc.PID != self {
		t.Errorf("PID = %d, want %d", inc.PID, self)
	}
	if inc.StartTime == 0 {
		t.Error("expected a non-zero start time for the running test process")
	}
}

func TestResolveIncarnation_DeadPID(t *testing.T) {
	// PID 0x7FFFFFFE is almost certainly not a live process. The
	// incarnation should still carry the PID with StartTime 0.
	inc := ResolveIncarnation(0x7FFFFFFE)
	if inc.PID != 0x7FFFFFFE {
		t.Errorf("PID = %d, want %d", inc.PID, uint32(0x7FFFFFFE))
	}
	if inc.StartTime != 0 {
		t.Errorf("StartTime = %d, want 0 for a non-existent PID", inc.StartTime)
	}
}
