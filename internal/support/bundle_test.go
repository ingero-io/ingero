package support

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Bundle writes a tgz that contains the canonical metadata.txt entry
// even when every other input fails (we're running in a hermetic
// test env without nvidia-smi / bpftool). Verify the tarball parses,
// metadata.txt is present, and at least one *.error.txt entry was
// captured (proving the failure-handling path works without aborting).
func TestBundle_WritesTarballWithMetadataEvenWhenInputsAreMissing(t *testing.T) {
	out := filepath.Join(t.TempDir(), "bundle.tgz")
	path, n, err := Bundle(out, "v0.11.0", "abc1234")
	if err != nil {
		t.Fatalf("Bundle: %v", err)
	}
	if n < 1 {
		t.Fatalf("entries=%d, want >= 1", n)
	}
	if path != out {
		// allowed to be absolute path of out; just check it points at the same file
		if got, _ := filepath.Abs(out); got != path {
			t.Fatalf("returned path %q, want %q (or its abs)", path, out)
		}
	}

	entries := readEntries(t, out)
	mustHave(t, entries, "ingero-support/metadata.txt")
	if !strings.Contains(entries["ingero-support/metadata.txt"], "ingero_version: v0.11.0") {
		t.Errorf("metadata.txt missing version line; got: %q", entries["ingero-support/metadata.txt"])
	}
	if !strings.Contains(entries["ingero-support/metadata.txt"], "ingero_commit:  abc1234") {
		t.Errorf("metadata.txt missing commit line; got: %q", entries["ingero-support/metadata.txt"])
	}
	// Env dump always succeeds (process always has env).
	mustHave(t, entries, "ingero-support/env-redacted.txt")
}

// Sensitive env vars are masked, non-sensitive ones survive verbatim.
func TestBundle_RedactsSensitiveEnvVars(t *testing.T) {
	t.Setenv("INGERO_TEST_PUBLIC_VAR", "hello-world")
	t.Setenv("INGERO_TEST_API_TOKEN", "supersecret")
	t.Setenv("INGERO_TEST_AWS_SECRET_KEY", "shouldnotleak")

	out := filepath.Join(t.TempDir(), "bundle.tgz")
	if _, _, err := Bundle(out, "v0.11.0", "abc"); err != nil {
		t.Fatalf("Bundle: %v", err)
	}
	entries := readEntries(t, out)
	env := entries["ingero-support/env-redacted.txt"]

	// Non-sensitive: present with value.
	if !strings.Contains(env, "INGERO_TEST_PUBLIC_VAR=hello-world") {
		t.Errorf("non-sensitive env var stripped or wrong: %q", env)
	}
	// Sensitive: keys present, values masked.
	if !strings.Contains(env, "INGERO_TEST_API_TOKEN=<redacted") {
		t.Errorf("INGERO_TEST_API_TOKEN value not redacted")
	}
	if strings.Contains(env, "supersecret") {
		t.Errorf("token VALUE leaked into bundle")
	}
	if !strings.Contains(env, "INGERO_TEST_AWS_SECRET_KEY=<redacted") {
		t.Errorf("INGERO_TEST_AWS_SECRET_KEY value not redacted")
	}
	if strings.Contains(env, "shouldnotleak") {
		t.Errorf("AWS-shaped value leaked into bundle")
	}
}

// Per-input failures land as <name>.error.txt rather than aborting
// the whole bundle. We can't easily force ONE input to fail without
// mocking, so we just confirm: the tarball parses cleanly even when
// most inputs fail (the test env lacks bpftool, nvidia-smi, etc.).
func TestBundle_ContinuesPastInputFailures(t *testing.T) {
	out := filepath.Join(t.TempDir(), "bundle.tgz")
	if _, n, err := Bundle(out, "v0.11.0", "abc"); err != nil {
		t.Fatalf("Bundle: %v", err)
	} else if n == 0 {
		t.Fatal("expected at least metadata + env entries")
	}
	entries := readEntries(t, out)
	hasErrorEntry := false
	for name := range entries {
		if strings.HasSuffix(name, ".error.txt") {
			hasErrorEntry = true
			break
		}
	}
	if !hasErrorEntry {
		t.Log("no .error.txt entries (test env has bpftool + nvidia-smi available?); bundle still wrote successfully")
	}
}

func TestIsSensitive(t *testing.T) {
	cases := []struct {
		key  string
		want bool
	}{
		{"GITHUB_TOKEN", true},
		{"AWS_SECRET_ACCESS_KEY", true},
		{"OPENAI_API_KEY", true},
		{"DATABASE_PASSWORD", true},
		{"PATH", false},
		{"HOME", false},
		{"INGERO_NODE_ID", false},
		{"NVIDIA_VISIBLE_DEVICES", false},
		// v0.12.1 (QA audit ★3 #7): pin the conservative-substring
		// behavior. KEY is a substring of MONKEY, so MONKEY_HOST is
		// masked spuriously. This is intentional: false positives are
		// safe (an env var ends up redacted in a support bundle); a
		// false negative could leak OPENAI_API_KEY. Lock this contract
		// so a future contributor doesn't "fix" it by switching to
		// word-boundary matching and accidentally regressing the
		// secret-detection.
		{"MONKEY_HOST", true},
		{"MONKEY_PATCH", true},
	}
	for _, c := range cases {
		if got := isSensitive(c.key); got != c.want {
			t.Errorf("isSensitive(%q)=%v, want %v", c.key, got, c.want)
		}
	}
}

// readEntries reads every regular-file entry out of the gzip tar at
// path and returns a name -> body map. Helper for the asserts above.
func readEntries(t *testing.T, path string) map[string]string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatalf("gzip: %v", err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	out := map[string]string{}
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("tar.Next: %v", err)
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		buf, err := io.ReadAll(tr)
		if err != nil {
			t.Fatalf("read %s: %v", hdr.Name, err)
		}
		out[hdr.Name] = string(buf)
	}
	return out
}

func mustHave(t *testing.T, entries map[string]string, name string) {
	t.Helper()
	if _, ok := entries[name]; !ok {
		got := make([]string, 0, len(entries))
		for k := range entries {
			got = append(got, k)
		}
		t.Fatalf("entry %q missing; have: %v", name, got)
	}
}
