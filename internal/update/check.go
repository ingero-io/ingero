// Package update provides a background version update checker.
//
// On every ingero invocation (except version/query/mcp), a goroutine checks
// GitHub Releases for a newer version. The check is cached for 24 hours in
// ~/.ingero/update-check. All errors are silently swallowed — the update
// check never interferes with normal operation.
package update

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Result is the outcome of an update check.
type Result struct {
	UpdateAvailable bool
	LatestVersion   string
	CurrentVersion  string
}

// stateFile persists the last check timestamp and result.
type stateFile struct {
	CheckedAt     time.Time `json:"checked_at"`
	LatestVersion string    `json:"latest_version"`
}

const (
	cacheTTL      = 24 * time.Hour
	httpTimeout   = 5 * time.Second
	stateFileName = "update-check"
)

// releasesURL is a var (not const) so tests can override it with httptest.
var releasesURL = "https://api.github.com/repos/ingero-io/ingero/releases/latest"

// CheckInBackground starts a non-blocking update check goroutine.
// Returns a channel that receives the result (or is closed on skip/error).
//
// Skip conditions (no goroutine spawned):
//   - currentVersion is "dev" or empty
//   - currentVersion contains "-" (pre-release / dirty build)
//   - INGERO_NO_UPDATE_NOTIFIER env var is set
func CheckInBackground(currentVersion string) <-chan Result {
	ch := make(chan Result, 1)

	if shouldSkip(currentVersion) {
		close(ch)
		return ch
	}

	go func() {
		defer close(ch)
		if r, ok := check(currentVersion); ok {
			ch <- r
		}
	}()

	return ch
}

// PrintNotice reads from the channel (non-blocking) and prints to stderr
// if an update is available. If the goroutine hasn't finished, skip silently.
func PrintNotice(ch <-chan Result) {
	printNoticeTo(os.Stderr, ch)
}

// printNoticeTo writes the update notice to w. Extracted for testability
// (avoids mutating os.Stderr in tests, which is not goroutine-safe).
func printNoticeTo(w io.Writer, ch <-chan Result) {
	select {
	case r, ok := <-ch:
		if ok && r.UpdateAvailable {
			fmt.Fprintf(w, "\nA new version of ingero is available: %s → %s\nhttps://github.com/ingero-io/ingero/releases/latest\n",
				r.CurrentVersion, r.LatestVersion)
		}
	default:
		// Goroutine still running — don't block the CLI exit.
	}
}

func shouldSkip(v string) bool {
	if v == "" || v == "dev" {
		return true
	}
	if strings.Contains(v, "-") {
		return true
	}
	if os.Getenv("INGERO_NO_UPDATE_NOTIFIER") != "" {
		return true
	}
	return false
}

func check(currentVersion string) (Result, bool) {
	dir := ingeroDir()
	if dir == "" {
		return Result{}, false
	}

	path := filepath.Join(dir, stateFileName)

	// Try cached state first.
	if st, err := readState(path); err == nil {
		if time.Since(st.CheckedAt) < cacheTTL {
			if isNewer(st.LatestVersion, currentVersion) {
				return Result{
					UpdateAvailable: true,
					LatestVersion:   st.LatestVersion,
					CurrentVersion:  currentVersion,
				}, true
			}
			return Result{}, false
		}
	}

	// Fetch latest from GitHub.
	latest := fetchLatest()
	if latest == "" {
		return Result{}, false
	}

	// Write state (best-effort).
	writeState(path, stateFile{
		CheckedAt:     time.Now(),
		LatestVersion: latest,
	})

	if isNewer(latest, currentVersion) {
		return Result{
			UpdateAvailable: true,
			LatestVersion:   latest,
			CurrentVersion:  currentVersion,
		}, true
	}
	return Result{}, false
}

// ingeroDir returns ~/.ingero/, respecting SUDO_USER when running as root.
// Creates the directory if it doesn't exist. Returns "" on failure.
func ingeroDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}

	// If running as root via sudo, use the invoking user's home.
	if sudoUser := os.Getenv("SUDO_USER"); sudoUser != "" && os.Getuid() == 0 {
		if h := lookupHome(sudoUser); h != "" {
			home = h
		}
	}

	dir := filepath.Join(home, ".ingero")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return ""
	}
	return dir
}

// lookupHome reads /etc/passwd to find a user's home directory.
// Same approach as store.lookupHome — duplicated to avoid import cycle.
func lookupHome(username string) string {
	data, err := os.ReadFile("/etc/passwd")
	if err != nil {
		return ""
	}
	prefix := username + ":"
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		// Format: user:x:uid:gid:gecos:home:shell
		fields := strings.SplitN(line, ":", 7)
		if len(fields) >= 6 {
			return fields[5]
		}
	}
	return ""
}

func readState(path string) (stateFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return stateFile{}, err
	}
	var st stateFile
	if err := json.Unmarshal(data, &st); err != nil {
		return stateFile{}, err
	}
	return st, nil
}

func writeState(path string, st stateFile) {
	data, err := json.Marshal(st)
	if err != nil {
		return
	}
	// Atomic write: write to temp file, then rename. Prevents corruption
	// when two ingero invocations race on the state file.
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return
	}
	os.Rename(tmp, path)
}

// githubRelease is the minimal GitHub API response we need.
type githubRelease struct {
	TagName string `json:"tag_name"`
}

func fetchLatest() string {
	client := &http.Client{Timeout: httpTimeout}
	resp, err := client.Get(releasesURL)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return ""
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if err != nil {
		return ""
	}

	var rel githubRelease
	if err := json.Unmarshal(body, &rel); err != nil {
		return ""
	}

	// Strip "v" prefix: tag "v0.7.0" → "0.7.0"
	return strings.TrimPrefix(rel.TagName, "v")
}

// isNewer returns true if latest is a higher semver than current.
// Handles 2-part ("0.6") and 3-part ("0.7.0") versions.
// Strips "v" prefix and ignores anything after "-" (pre-release).
func isNewer(latest, current string) bool {
	lMajor, lMinor, lPatch := parseSemver(latest)
	cMajor, cMinor, cPatch := parseSemver(current)

	if lMajor != cMajor {
		return lMajor > cMajor
	}
	if lMinor != cMinor {
		return lMinor > cMinor
	}
	return lPatch > cPatch
}

// parseSemver extracts major.minor.patch from a version string.
// Strips "v" prefix and everything after "-". Missing patch defaults to 0.
func parseSemver(v string) (int, int, int) {
	v = strings.TrimPrefix(v, "v")
	if idx := strings.Index(v, "-"); idx >= 0 {
		v = v[:idx]
	}

	parts := strings.SplitN(v, ".", 4)
	major, _ := strconv.Atoi(safeIndex(parts, 0))
	minor, _ := strconv.Atoi(safeIndex(parts, 1))
	patch, _ := strconv.Atoi(safeIndex(parts, 2))
	return major, minor, patch
}

func safeIndex(s []string, i int) string {
	if i < len(s) {
		return s[i]
	}
	return "0"
}
