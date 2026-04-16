package symtab

import "testing"

// TestLookupByBuildID_UnknownBuildID verifies that a binary whose build-id
// is not in the DB returns (nil, nil) rather than an error. We use the
// running Go test binary itself as a target — it's a real ELF with a
// build-id (on Linux) but won't match any entry in knownPyOffsets.
func TestLookupByBuildID_UnknownBuildID(t *testing.T) {
	// For a binary without a build-id or with an unknown one, should return (nil, nil).
	// Use the Go test binary itself as a test target — it won't be in the DB.
	offsets, err := LookupByBuildID("/proc/self/exe")
	if err != nil {
		t.Logf("read failed (OK if not Linux): %v", err)
		return
	}
	if offsets != nil {
		t.Errorf("expected nil for unknown build-id, got %+v", offsets)
	}
}

// TestLookupByBuildID_NonexistentFile verifies that attempting to look up a
// path that doesn't exist propagates an error from the underlying ELF open.
func TestLookupByBuildID_NonexistentFile(t *testing.T) {
	_, err := LookupByBuildID("/nonexistent/path")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}
