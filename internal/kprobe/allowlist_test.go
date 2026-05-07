package kprobe

import "testing"

func TestParseNVIDIADriverVersion(t *testing.T) {
	cases := []struct {
		banner string
		want   string
	}{
		{
			banner: "NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.86.10  Wed Jul 26 ...\nGCC version: gcc version ...\n",
			want:   "535.86.10",
		},
		{
			banner: "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  535.104.05  Tue Aug 22 ...",
			want:   "535.104.05",
		},
		{
			banner: "NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54  Mon Jan 15 ...",
			want:   "550.54",
		},
		{
			banner: "",
			want:   "",
		},
		{
			banner: "garbage line\nno version here",
			want:   "",
		},
	}
	for _, c := range cases {
		if got := parseNVIDIADriverVersion(c.banner); got != c.want {
			t.Errorf("parseNVIDIADriverVersion(%q) = %q, want %q", c.banner, got, c.want)
		}
	}
}

func TestMatchVersion(t *testing.T) {
	cases := []struct {
		got, want string
		match     bool
	}{
		{"535.86.10", "535.", true},
		{"535.86.10", "535.86.", true},
		{"5.15.0-89-generic", "5.15.", true},
		{"6.5.0-15-generic", "5.15.", false},
		{"5.150.0", "5.15.", false}, // boundary check: trailing dot makes 5.15. distinct from 5.150
		{"5.15", "5.15.", false},     // exact "5.15" without trailing dot doesn't match prefix "5.15."
		{"", "535.", false},
	}
	for _, c := range cases {
		if got := MatchVersion(c.got, c.want); got != c.match {
			t.Errorf("MatchVersion(%q, %q) = %v, want %v", c.got, c.want, got, c.match)
		}
	}
}

func TestIsAllowed_DefaultMatchesLambdaA10(t *testing.T) {
	v := Versions{NVIDIADriver: "535.86.10", LinuxKernel: "5.15.0-89-generic"}
	if !IsAllowed(v, DefaultAllowlist) {
		t.Errorf("Lambda A10 baseline should be on default allowlist; got %+v", v)
	}
}

func TestIsAllowed_DefaultMatchesGH200(t *testing.T) {
	v := Versions{NVIDIADriver: "535.104.05", LinuxKernel: "6.5.0-1015-aws"}
	if !IsAllowed(v, DefaultAllowlist) {
		t.Errorf("Lambda GH200 baseline should be on default allowlist; got %+v", v)
	}
}

// 2026-05-07: Lambda A10 image now ships driver 570.x + kernel 6.8;
// validated end-to-end on a fresh provision.
func TestIsAllowed_DefaultMatchesA10Current(t *testing.T) {
	v := Versions{NVIDIADriver: "570.148.08", LinuxKernel: "6.8.0-60-generic"}
	if !IsAllowed(v, DefaultAllowlist) {
		t.Errorf("Lambda A10 current image should be on default allowlist; got %+v", v)
	}
}

func TestIsAllowed_RejectsUntestedDriver(t *testing.T) {
	v := Versions{NVIDIADriver: "470.183.01", LinuxKernel: "5.15.0-89-generic"}
	if IsAllowed(v, DefaultAllowlist) {
		t.Errorf("470 driver should NOT be on default allowlist; got %+v", v)
	}
}

func TestIsAllowed_RejectsUntestedKernel(t *testing.T) {
	v := Versions{NVIDIADriver: "535.86.10", LinuxKernel: "4.18.0-477.el8.x86_64"}
	if IsAllowed(v, DefaultAllowlist) {
		t.Errorf("RHEL 4.18 kernel should NOT be on default allowlist; got %+v", v)
	}
}

func TestIsAllowed_EmptyDriverRejected(t *testing.T) {
	v := Versions{NVIDIADriver: "", LinuxKernel: "5.15.0"}
	if IsAllowed(v, DefaultAllowlist) {
		t.Errorf("empty driver should never be allowed")
	}
}

func TestIsAllowed_EmptyAllowlistRejectsAll(t *testing.T) {
	v := Versions{NVIDIADriver: "535.86.10", LinuxKernel: "5.15.0"}
	if IsAllowed(v, nil) {
		t.Errorf("empty allowlist should reject all")
	}
}

func TestParseDriverMajor(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"535.86.10", 535},
		{"550.54", 550},
		{"470.183.01", 470},
		{"", 0},
		{"badinput", 0},
		{"535", 0}, // no dot
	}
	for _, c := range cases {
		if got := ParseDriverMajor(c.in); got != c.want {
			t.Errorf("ParseDriverMajor(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestDescribeStatus(t *testing.T) {
	v := Versions{NVIDIADriver: "535.86.10", LinuxKernel: "5.15.0-89-generic"}
	allowed := DescribeStatus(v, true, DefaultAllowlist)
	if !contains(allowed, "535.86.10") || !contains(allowed, "will load") {
		t.Errorf("allowed status missing expected fields: %q", allowed)
	}
	denied := DescribeStatus(v, false, DefaultAllowlist)
	if !contains(denied, "NOT load") {
		t.Errorf("denied status should say NOT load: %q", denied)
	}
	emptyV := Versions{}
	denyEmpty := DescribeStatus(emptyV, false, DefaultAllowlist)
	if !contains(denyEmpty, "(unknown)") {
		t.Errorf("empty version should display (unknown): %q", denyEmpty)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
