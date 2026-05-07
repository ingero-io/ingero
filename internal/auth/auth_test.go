package auth

import "testing"

func TestParseBearer(t *testing.T) {
	cases := []struct {
		in       string
		wantTok  string
		wantOK   bool
		describe string
	}{
		{"Bearer abc123", "abc123", true, "canonical case"},
		{"bearer abc123", "abc123", true, "lowercase scheme"},
		{"BEARER abc123", "abc123", true, "uppercase scheme"},
		{"Bearer  abc123  ", "abc123", true, "trims surrounding whitespace"},
		{"Bearer\tabc123", "abc123", true, "tab between scheme and token"},
		{"", "", false, "empty header"},
		{"Bear", "", false, "shorter than scheme literal"},
		{"Bearer", "", false, "scheme only"},
		{"Bearer ", "", false, "empty token after scheme"},
		{"Bearer    ", "", false, "all-whitespace token"},
		{"Basic abc123", "", false, "wrong scheme"},
		{"BeerAbc123", "", false, "scheme letters but not Bearer"},
	}
	for _, c := range cases {
		t.Run(c.describe, func(t *testing.T) {
			got, ok := ParseBearer(c.in)
			if got != c.wantTok || ok != c.wantOK {
				t.Errorf("ParseBearer(%q) = (%q, %v), want (%q, %v)",
					c.in, got, ok, c.wantTok, c.wantOK)
			}
		})
	}
}

func TestTokensEqual_Match(t *testing.T) {
	if !TokensEqual("hunter2", "hunter2") {
		t.Errorf("equal tokens should compare equal")
	}
}

func TestTokensEqual_Mismatch(t *testing.T) {
	if TokensEqual("hunter2", "hunter3") {
		t.Errorf("differing tokens should compare unequal")
	}
}

func TestTokensEqual_DifferentLengths(t *testing.T) {
	if TokensEqual("short", "a-much-longer-token-string-here") {
		t.Errorf("different-length tokens should compare unequal")
	}
}

func TestTokensEqual_EmptyWant(t *testing.T) {
	if TokensEqual("anything", "") {
		t.Errorf("empty want must always reject (defense-in-depth guard)")
	}
	if TokensEqual("", "") {
		t.Errorf("empty/empty must reject (the empty-want guard fires first)")
	}
}

func TestTokensEqual_EmptyGot(t *testing.T) {
	if TokensEqual("", "want-something") {
		t.Errorf("empty got vs non-empty want must reject")
	}
}
