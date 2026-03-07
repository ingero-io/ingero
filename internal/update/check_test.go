package update

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestIsNewer(t *testing.T) {
	tests := []struct {
		latest  string
		current string
		want    bool
	}{
		// Basic cases
		{"0.7.0", "0.6", true},
		{"0.7.0", "0.6.0", true},
		{"0.6", "0.6", false},
		{"0.6.0", "0.6.0", false},
		{"0.5", "0.6", false},
		{"0.5.0", "0.6.0", false},

		// Same minor, different patch
		{"0.6.1", "0.6.0", true},
		{"0.6.0", "0.6.1", false},
		{"0.6.1", "0.6", true},

		// Major version bumps
		{"1.0.0", "0.9.9", true},
		{"2.0.0", "1.99.99", true},

		// With v prefix
		{"v0.7.0", "v0.6.0", true},
		{"v0.7.0", "0.6.0", true},
		{"0.7.0", "v0.6.0", true},

		// Pre-release stripped — both parse to 0.7.0, so not newer.
		// (In practice, versions with "-" are skipped by shouldSkip() before isNewer runs.)
		{"0.7.0", "0.7.0-22-g0763640", false},

		// Equal after stripping
		{"0.6.0", "0.6", false},
		{"0.6", "0.6.0", false},

		// Single-component versions
		{"1", "0", true},
		{"0", "0", false},
		{"0", "1", false},

		// Non-numeric (Atoi returns 0 on error)
		{"abc", "0.0.0", false},
		{"abc", "def", false},
	}

	for _, tt := range tests {
		t.Run(tt.latest+"_vs_"+tt.current, func(t *testing.T) {
			got := isNewer(tt.latest, tt.current)
			if got != tt.want {
				t.Errorf("isNewer(%q, %q) = %v, want %v", tt.latest, tt.current, got, tt.want)
			}
		})
	}
}

func TestParseSemver(t *testing.T) {
	tests := []struct {
		input               string
		major, minor, patch int
	}{
		{"0.6", 0, 6, 0},
		{"0.6.0", 0, 6, 0},
		{"0.7.1", 0, 7, 1},
		{"1.2.3", 1, 2, 3},
		{"v1.2.3", 1, 2, 3},
		{"0.7.0-22-g0763640", 0, 7, 0},
		{"dev", 0, 0, 0},
		{"", 0, 0, 0},

		// Edge cases
		{"1.2.3.4", 1, 2, 3},       // 4th component silently ignored
		{"01.02.03", 1, 2, 3},      // leading zeros: Atoi normalizes
		{"abc.def.ghi", 0, 0, 0},   // non-numeric → Atoi returns 0
		{"-1.-2.-3", 0, 0, 0},      // leading "-" triggers pre-release strip → ""
		{"7", 7, 0, 0},             // single-component
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			major, minor, patch := parseSemver(tt.input)
			if major != tt.major || minor != tt.minor || patch != tt.patch {
				t.Errorf("parseSemver(%q) = (%d, %d, %d), want (%d, %d, %d)",
					tt.input, major, minor, patch, tt.major, tt.minor, tt.patch)
			}
		})
	}
}

func TestShouldSkip(t *testing.T) {
	tests := []struct {
		version string
		envVar  bool
		want    bool
	}{
		{"dev", false, true},
		{"", false, true},
		{"0.7.0-22-g0763640", false, false}, // git describe — parseSemver strips suffix
		{"0.7.0-dirty", false, false},        // dirty build — still check for updates
		{"v0.6-32-g6efdbb5", false, false},   // typical dev build
		{"0.6", false, false},
		{"0.7.0", false, false},
	}

	for _, tt := range tests {
		t.Run(tt.version, func(t *testing.T) {
			if tt.envVar {
				t.Setenv("INGERO_NO_UPDATE_NOTIFIER", "1")
			}
			got := shouldSkip(tt.version)
			if got != tt.want {
				t.Errorf("shouldSkip(%q) = %v, want %v", tt.version, got, tt.want)
			}
		})
	}
}

func TestShouldSkipEnvVar(t *testing.T) {
	t.Setenv("INGERO_NO_UPDATE_NOTIFIER", "1")
	if !shouldSkip("0.7.0") {
		t.Error("shouldSkip should return true when INGERO_NO_UPDATE_NOTIFIER is set")
	}
}

func TestCheckInBackgroundSkipsDev(t *testing.T) {
	ch := CheckInBackground("dev")
	// Channel should be closed immediately (no goroutine).
	select {
	case _, ok := <-ch:
		if ok {
			t.Error("expected channel to be closed for dev version")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("channel not closed within 100ms")
	}
}

func TestCheckInBackgroundSkipsEnvVar(t *testing.T) {
	t.Setenv("INGERO_NO_UPDATE_NOTIFIER", "1")
	ch := CheckInBackground("0.7.0")
	select {
	case _, ok := <-ch:
		if ok {
			t.Error("expected channel to be closed when env var is set")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("channel not closed within 100ms")
	}
}

func TestStateFileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "update-check")

	now := time.Now().Truncate(time.Second)
	want := stateFile{
		CheckedAt:     now,
		LatestVersion: "0.7.0",
	}

	writeState(path, want)

	got, err := readState(path)
	if err != nil {
		t.Fatalf("readState: %v", err)
	}

	if !got.CheckedAt.Equal(want.CheckedAt) {
		t.Errorf("CheckedAt = %v, want %v", got.CheckedAt, want.CheckedAt)
	}
	if got.LatestVersion != want.LatestVersion {
		t.Errorf("LatestVersion = %q, want %q", got.LatestVersion, want.LatestVersion)
	}
}

func TestWriteStateAtomic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, stateFileName)

	// Write state and verify the temp file is cleaned up.
	writeState(path, stateFile{
		CheckedAt:     time.Now(),
		LatestVersion: "0.7.0",
	})

	// State file should exist.
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("state file not created: %v", err)
	}

	// Temp file should NOT exist (renamed away).
	if _, err := os.Stat(path + ".tmp"); err == nil {
		t.Error("temp file should have been renamed away")
	}
}

func TestReadStateMissing(t *testing.T) {
	_, err := readState("/nonexistent/path")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestReadStateCorrupt(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "update-check")
	os.WriteFile(path, []byte("not json"), 0o644)

	_, err := readState(path)
	if err == nil {
		t.Error("expected error for corrupt file")
	}
}

func TestPrintNoticeWithUpdate(t *testing.T) {
	ch := make(chan Result, 1)
	ch <- Result{
		UpdateAvailable: true,
		LatestVersion:   "0.7.0",
		CurrentVersion:  "0.6",
	}

	var buf bytes.Buffer
	printNoticeTo(&buf, ch)

	output := buf.String()
	if output == "" {
		t.Error("expected output")
	}
	if !strings.Contains(output, "0.6") || !strings.Contains(output, "0.7.0") ||
		!strings.Contains(output, "github.com/ingero-io/ingero/releases") {
		t.Errorf("unexpected output: %s", output)
	}
}

func TestPrintNoticeNoUpdate(t *testing.T) {
	ch := make(chan Result, 1)
	close(ch)

	var buf bytes.Buffer
	printNoticeTo(&buf, ch)

	if buf.Len() > 0 {
		t.Errorf("expected no output, got: %s", buf.String())
	}
}

func TestPrintNoticeNotReady(t *testing.T) {
	ch := make(chan Result, 1)
	// Don't send anything — goroutine "still running".

	var buf bytes.Buffer
	printNoticeTo(&buf, ch)

	if buf.Len() > 0 {
		t.Errorf("expected no output, got: %s", buf.String())
	}
}

func TestStateCacheTTL(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, stateFileName)

	// Write a state file with a recent check (within TTL).
	st := stateFile{
		CheckedAt:     time.Now(),
		LatestVersion: "0.7.0",
	}
	data, _ := json.Marshal(st)
	os.WriteFile(path, data, 0o644)

	// Read it back — should be valid.
	got, err := readState(path)
	if err != nil {
		t.Fatalf("readState: %v", err)
	}
	if time.Since(got.CheckedAt) >= cacheTTL {
		t.Error("state file should be within TTL")
	}
}

// --- Integration tests for fetchLatest (via httptest) ---

func TestFetchLatestSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(githubRelease{TagName: "v0.9.0"})
	}))
	defer srv.Close()

	old := releasesURL
	releasesURL = srv.URL
	defer func() { releasesURL = old }()

	got := fetchLatest()
	if got != "0.9.0" {
		t.Errorf("fetchLatest() = %q, want %q", got, "0.9.0")
	}
}

func TestFetchLatestNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	}))
	defer srv.Close()

	old := releasesURL
	releasesURL = srv.URL
	defer func() { releasesURL = old }()

	if got := fetchLatest(); got != "" {
		t.Errorf("fetchLatest() = %q, want empty on 403", got)
	}
}

func TestFetchLatestMalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("not json at all"))
	}))
	defer srv.Close()

	old := releasesURL
	releasesURL = srv.URL
	defer func() { releasesURL = old }()

	if got := fetchLatest(); got != "" {
		t.Errorf("fetchLatest() = %q, want empty on malformed JSON", got)
	}
}

func TestFetchLatestMissingTagName(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"name": "Release 0.9.0"}`))
	}))
	defer srv.Close()

	old := releasesURL
	releasesURL = srv.URL
	defer func() { releasesURL = old }()

	if got := fetchLatest(); got != "" {
		t.Errorf("fetchLatest() = %q, want empty when tag_name missing", got)
	}
}

func TestFetchLatestStripsVPrefix(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(githubRelease{TagName: "v1.2.3"})
	}))
	defer srv.Close()

	old := releasesURL
	releasesURL = srv.URL
	defer func() { releasesURL = old }()

	got := fetchLatest()
	if got != "1.2.3" {
		t.Errorf("fetchLatest() = %q, want %q (v prefix stripped)", got, "1.2.3")
	}
}

// --- Integration tests for check() (cache hit/miss) ---

func TestCheckCacheHitUpdateAvailable(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	// Pre-create ~/.ingero/update-check with a fresh cache saying latest is 0.8.0.
	ingeroPath := filepath.Join(dir, ".ingero")
	os.MkdirAll(ingeroPath, 0o755)
	writeState(filepath.Join(ingeroPath, stateFileName), stateFile{
		CheckedAt:     time.Now(),
		LatestVersion: "0.8.0",
	})

	r, ok := check("0.6.0")
	if !ok || !r.UpdateAvailable {
		t.Error("expected update available from cached state")
	}
	if r.LatestVersion != "0.8.0" {
		t.Errorf("LatestVersion = %q, want %q", r.LatestVersion, "0.8.0")
	}
}

func TestCheckCacheHitNoUpdate(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	ingeroPath := filepath.Join(dir, ".ingero")
	os.MkdirAll(ingeroPath, 0o755)
	writeState(filepath.Join(ingeroPath, stateFileName), stateFile{
		CheckedAt:     time.Now(),
		LatestVersion: "0.6.0",
	})

	_, ok := check("0.6.0")
	if ok {
		t.Error("expected no update when cached version equals current")
	}
}

func TestCheckCacheExpiredFetches(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	// Write an expired cache (25 hours ago).
	ingeroPath := filepath.Join(dir, ".ingero")
	os.MkdirAll(ingeroPath, 0o755)
	writeState(filepath.Join(ingeroPath, stateFileName), stateFile{
		CheckedAt:     time.Now().Add(-25 * time.Hour),
		LatestVersion: "0.6.0",
	})

	// Mock GitHub to return a newer version.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(githubRelease{TagName: "v0.9.0"})
	}))
	defer srv.Close()

	old := releasesURL
	releasesURL = srv.URL
	defer func() { releasesURL = old }()

	r, ok := check("0.6.0")
	if !ok || !r.UpdateAvailable {
		t.Error("expected update available after cache expired")
	}
	if r.LatestVersion != "0.9.0" {
		t.Errorf("LatestVersion = %q, want %q", r.LatestVersion, "0.9.0")
	}
}

func TestCheckNoStateFetches(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	// No pre-existing state file. Mock GitHub.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(githubRelease{TagName: "v0.8.0"})
	}))
	defer srv.Close()

	old := releasesURL
	releasesURL = srv.URL
	defer func() { releasesURL = old }()

	r, ok := check("0.6.0")
	if !ok || !r.UpdateAvailable {
		t.Error("expected update available from fresh fetch")
	}
	if r.LatestVersion != "0.8.0" {
		t.Errorf("LatestVersion = %q, want %q", r.LatestVersion, "0.8.0")
	}

	// Verify state file was written.
	st, err := readState(filepath.Join(dir, ".ingero", stateFileName))
	if err != nil {
		t.Fatalf("state file not written: %v", err)
	}
	if st.LatestVersion != "0.8.0" {
		t.Errorf("cached LatestVersion = %q, want %q", st.LatestVersion, "0.8.0")
	}
}

// --- ingeroDir tests ---

// --- WaitNotice tests ---

func TestWaitNoticeWithUpdate(t *testing.T) {
	ch := make(chan Result, 1)
	ch <- Result{
		UpdateAvailable: true,
		LatestVersion:   "0.8.0",
		CurrentVersion:  "0.6",
	}

	var buf bytes.Buffer
	waitNoticeTo(&buf, ch, time.Second)

	output := buf.String()
	if !strings.Contains(output, "0.8.0") || !strings.Contains(output, "0.6") {
		t.Errorf("expected update notice, got: %s", output)
	}
}

func TestWaitNoticeTimeout(t *testing.T) {
	ch := make(chan Result) // unbuffered, never sent to

	var buf bytes.Buffer
	start := time.Now()
	waitNoticeTo(&buf, ch, 50*time.Millisecond)
	elapsed := time.Since(start)

	if elapsed > 500*time.Millisecond {
		t.Errorf("WaitNotice took %v, expected ~50ms timeout", elapsed)
	}
	if buf.Len() > 0 {
		t.Errorf("expected no output on timeout, got: %s", buf.String())
	}
}

func TestWaitNoticeNoUpdate(t *testing.T) {
	ch := make(chan Result, 1)
	close(ch)

	var buf bytes.Buffer
	waitNoticeTo(&buf, ch, time.Second)

	if buf.Len() > 0 {
		t.Errorf("expected no output, got: %s", buf.String())
	}
}

func TestIngeroDir(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	got := ingeroDir()
	want := filepath.Join(dir, ".ingero")
	if got != want {
		t.Errorf("ingeroDir() = %q, want %q", got, want)
	}

	// Verify directory was created.
	info, err := os.Stat(got)
	if err != nil {
		t.Fatalf("directory not created: %v", err)
	}
	if !info.IsDir() {
		t.Error("expected directory, got file")
	}
}
