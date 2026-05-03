package cli

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ingero-io/ingero/docs"
)

const seedYAML = `currency_name: USD
currency_symbol: "$"
providers:
  ec2:
    "NVIDIA H100 80GB HBM3": 12.29
    "NVIDIA L4 24GB": 0.85
  lambda:
    "NVIDIA H100 80GB HBM3": 2.49
fallback_rate: 0.0
`

// httpsTestServer wraps httptest.NewTLSServer so we can swap the URL scheme
// for tests that exercise non-HTTPS rejection without actually opening
// plain HTTP listeners.
func httpsTestServer(t *testing.T, h http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewTLSServer(h)
	t.Cleanup(srv.Close)
	return srv
}

// clientWithTLS returns a client trusting the test server's self-signed cert.
// Used only by tests; production CLI uses default TLS verification.
func clientWithTLS(srv *httptest.Server) *http.Client {
	return srv.Client()
}

func TestParseAndValidate_HappyPath(t *testing.T) {
	rates, err := parseAndValidate([]byte(seedYAML))
	if err != nil {
		t.Fatalf("parseAndValidate: %v", err)
	}
	if rates.CurrencyName != "USD" {
		t.Errorf("currency_name: %q", rates.CurrencyName)
	}
	if len(rates.Providers) != 2 {
		t.Errorf("providers: got %d, want 2", len(rates.Providers))
	}
}

func TestParseAndValidate_Errors(t *testing.T) {
	cases := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name:    "malformed",
			body:    "::: not yaml :::",
			wantErr: "parse:",
		},
		{
			name: "missing currency_name",
			body: `currency_symbol: "$"
providers:
  ec2:
    "NVIDIA H100 80GB HBM3": 12.29
fallback_rate: 0.0
`,
			wantErr: "currency_name required",
		},
		{
			name: "missing currency_symbol",
			body: `currency_name: USD
providers:
  ec2:
    "NVIDIA H100 80GB HBM3": 12.29
fallback_rate: 0.0
`,
			wantErr: "currency_symbol required",
		},
		{
			name: "empty providers map",
			body: `currency_name: USD
currency_symbol: "$"
providers: {}
fallback_rate: 0.0
`,
			wantErr: "at least one provider required",
		},
		{
			name: "missing providers",
			body: `currency_name: USD
currency_symbol: "$"
fallback_rate: 0.0
`,
			wantErr: "at least one provider required",
		},
		{
			name: "missing fallback_rate",
			body: `currency_name: USD
currency_symbol: "$"
providers:
  ec2:
    "NVIDIA H100 80GB HBM3": 12.29
`,
			wantErr: "fallback_rate required",
		},
		{
			name: "negative fallback_rate",
			body: `currency_name: USD
currency_symbol: "$"
providers:
  ec2:
    "NVIDIA H100 80GB HBM3": 12.29
fallback_rate: -1.5
`,
			wantErr: "fallback_rate must be >= 0",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseAndValidate([]byte(tc.body))
			if err == nil {
				t.Fatal("want error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("err = %q, want substring %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestFetchRates_HappyPath(t *testing.T) {
	srv := httpsTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(seedYAML))
	})

	body, err := fetchRatesWithClient(context.Background(), srv.URL, 5*time.Second, clientWithTLS(srv))
	if err != nil {
		t.Fatalf("fetchRates: %v", err)
	}
	if string(body) != seedYAML {
		t.Errorf("body mismatch")
	}
}

func TestFetchRates_404(t *testing.T) {
	srv := httpsTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	})

	_, err := fetchRatesWithClient(context.Background(), srv.URL, 5*time.Second, clientWithTLS(srv))
	if err == nil || !strings.Contains(err.Error(), "HTTP 404") {
		t.Fatalf("err = %v, want HTTP 404", err)
	}
}

func TestFetchRates_500(t *testing.T) {
	srv := httpsTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})

	_, err := fetchRatesWithClient(context.Background(), srv.URL, 5*time.Second, clientWithTLS(srv))
	if err == nil || !strings.Contains(err.Error(), "HTTP 500") {
		t.Fatalf("err = %v, want HTTP 500", err)
	}
}

func TestFetchRates_Timeout(t *testing.T) {
	srv := httpsTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(500 * time.Millisecond)
		_, _ = w.Write([]byte(seedYAML))
	})

	_, err := fetchRatesWithClient(context.Background(), srv.URL, 50*time.Millisecond, clientWithTLS(srv))
	if err == nil || !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("err = %v, want timeout", err)
	}
}

func TestFetchRates_RedirectToHTTP_Rejected(t *testing.T) {
	httpSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(seedYAML))
	}))
	t.Cleanup(httpSrv.Close)

	httpsSrv := httpsTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, httpSrv.URL, http.StatusFound)
	})

	_, err := fetchRatesWithClient(context.Background(), httpsSrv.URL, 5*time.Second, clientWithTLS(httpsSrv))
	if err == nil || !strings.Contains(err.Error(), "non-HTTPS") {
		t.Fatalf("err = %v, want non-HTTPS rejection", err)
	}
}

// withExplicitURL sets the --url flag on ratesUpdateCmd so that
// cmd.Flags().Changed("url") returns true inside runRatesUpdate.
// Required to opt out of the embedded-md fallback path. Cleans up
// after the test.
func withExplicitURL(t *testing.T, u string) {
	t.Helper()
	if err := ratesUpdateCmd.Flags().Set("url", u); err != nil {
		t.Fatalf("flag set: %v", err)
	}
	t.Cleanup(func() {
		// Reset Changed by setting back to the default value, then
		// re-running flag.Changed bookkeeping requires direct
		// access; the simplest reset is to clear the flag's
		// Changed bit by hand via the lookup.
		f := ratesUpdateCmd.Flags().Lookup("url")
		if f != nil {
			f.Changed = false
			f.Value.Set(defaultRatesURL)
		}
	})
}

func TestRunRatesUpdate_RejectsHTTP(t *testing.T) {
	prev := ratesURL
	t.Cleanup(func() { ratesURL = prev })
	ratesURL = "http://example.com/gpu_rates.yaml"
	withExplicitURL(t, ratesURL)

	err := runRatesUpdate(ratesUpdateCmd, nil)
	if err == nil || !strings.Contains(err.Error(), "https") {
		t.Fatalf("err = %v, want https rejection", err)
	}
}

func TestRunRatesUpdate_RejectsBadURL(t *testing.T) {
	prev := ratesURL
	t.Cleanup(func() { ratesURL = prev })
	ratesURL = "://not-a-url"
	withExplicitURL(t, ratesURL)

	err := runRatesUpdate(ratesUpdateCmd, nil)
	if err == nil {
		t.Fatal("want error for malformed URL")
	}
}

func TestRunRatesUpdate_OutputDirMissing(t *testing.T) {
	prevURL, prevOut := ratesURL, ratesOutput
	t.Cleanup(func() { ratesURL, ratesOutput = prevURL, prevOut })

	srv := httpsTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(seedYAML))
	})
	ratesURL = srv.URL
	ratesOutput = filepath.Join(t.TempDir(), "missing-subdir", "gpu_rates.yaml")
	withExplicitURL(t, ratesURL)

	err := runRatesUpdate(ratesUpdateCmd, nil)
	if err == nil || !strings.Contains(err.Error(), "directory does not exist") {
		t.Fatalf("err = %v, want directory does not exist", err)
	}
}

func TestAtomicWrite_ReplacesExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "gpu_rates.yaml")

	if err := os.WriteFile(path, []byte("OLD"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := atomicWrite(path, []byte(seedYAML)); err != nil {
		t.Fatalf("atomicWrite: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != seedYAML {
		t.Errorf("file content mismatch")
	}
	// Confirm no .tmp leftovers.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.Contains(e.Name(), ".tmp") {
			t.Errorf("leftover tmp file: %s", e.Name())
		}
	}
}

func TestAtomicWrite_OriginalIntactOnExplicitURLFailure(t *testing.T) {
	// With --url set explicitly, fallback is disabled. A bogus URL
	// makes fetch fail and the original output file must survive.
	dir := t.TempDir()
	path := filepath.Join(dir, "gpu_rates.yaml")
	if err := os.WriteFile(path, []byte("OLD"), 0o644); err != nil {
		t.Fatal(err)
	}

	prevURL, prevOut := ratesURL, ratesOutput
	t.Cleanup(func() { ratesURL, ratesOutput = prevURL, prevOut })
	ratesURL = "https://127.0.0.1:1/nope.yaml" // unreachable
	ratesOutput = path
	prevTimeout := ratesTimeout
	t.Cleanup(func() { ratesTimeout = prevTimeout })
	ratesTimeout = 100 * time.Millisecond
	withExplicitURL(t, ratesURL)

	err := runRatesUpdate(ratesUpdateCmd, nil)
	if err == nil {
		t.Fatal("want fetch failure")
	}
	got, _ := os.ReadFile(path)
	if string(got) != "OLD" {
		t.Errorf("original file modified on fetch failure: %q", got)
	}
}

func TestRunRatesUpdate_FallbackOnDefaultURLFailure(t *testing.T) {
	// With --url left at default, a fetch failure triggers the
	// embedded md fallback. The output file is written with the
	// fallback YAML and the success line names "embedded-fallback".
	srv := httpsTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})

	prevURL, prevOut, prevTimeout := ratesURL, ratesOutput, ratesTimeout
	t.Cleanup(func() { ratesURL, ratesOutput, ratesTimeout = prevURL, prevOut, prevTimeout })

	dir := t.TempDir()
	ratesURL = srv.URL // package var only; flag.Changed stays false → fallback allowed
	ratesOutput = filepath.Join(dir, "gpu_rates.yaml")
	ratesTimeout = 2 * time.Second

	// Capture stdout via cmd.SetOut on ratesUpdateCmd.
	var buf strings.Builder
	prevOutW := ratesUpdateCmd.OutOrStdout()
	ratesUpdateCmd.SetOut(&buf)
	t.Cleanup(func() { ratesUpdateCmd.SetOut(prevOutW) })

	if err := runRatesUpdate(ratesUpdateCmd, nil); err != nil {
		t.Fatalf("runRatesUpdate: %v", err)
	}

	body, err := os.ReadFile(ratesOutput)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if !strings.Contains(string(body), "currency_name: USD") {
		t.Errorf("output missing fallback YAML")
	}
	if !strings.Contains(buf.String(), "source=embedded-fallback") {
		t.Errorf("stdout missing source label: %q", buf.String())
	}
}

func TestExtractYAMLFromMarkdown(t *testing.T) {
	cases := []struct {
		name    string
		md      string
		want    string
		wantErr string
	}{
		{
			name: "happy",
			md: "prefix\n```yaml\ncurrency_name: USD\nproviders:\n  ec2: {}\n```\nsuffix\n",
			want: "currency_name: USD\nproviders:\n  ec2: {}\n",
		},
		{
			name:    "no block",
			md:      "no fences here, just words\n",
			wantErr: "no ```yaml block found",
		},
		{
			name:    "unterminated",
			md:      "prefix\n```yaml\ncurrency_name: USD\n", // no closing
			wantErr: "unterminated",
		},
		{
			name: "ignores non-yaml fences",
			md:   "```python\nx=1\n```\n```yaml\nfoo: bar\n```\n",
			want: "foo: bar\n",
		},
		{
			name: "CRLF line endings",
			md:   "prefix\r\n```yaml\r\ncurrency_name: USD\r\nproviders:\r\n  ec2: {}\r\n```\r\nsuffix\r\n",
			want: "currency_name: USD\nproviders:\n  ec2: {}\n",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := extractYAMLFromMarkdown([]byte(tc.md))
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want substring %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if string(got) != tc.want {
				t.Errorf("yaml = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestEmbeddedFallback_ParsesAndValidates(t *testing.T) {
	// Sanity check the embedded md ships valid rates. Catches regressions
	// where the fallback YAML in docs/gpu_rates.md drifts from a parseable
	// shape (forgetting to close a fence, breaking indentation, etc.).
	body, err := extractYAMLFromMarkdown(docs.GPURatesMD)
	if err != nil {
		t.Fatalf("extract: %v", err)
	}
	if _, err := parseAndValidate(body); err != nil {
		t.Fatalf("parse+validate: %v", err)
	}
}

// fetchRatesWithClient is the test-only seam that injects a pre-configured
// http.Client (with TLS roots for httptest's self-signed cert). The
// production code path in fetchRates constructs its own client; this
// function mirrors fetchRates exactly except for client construction.
func fetchRatesWithClient(ctx context.Context, src string, timeout time.Duration, client *http.Client) ([]byte, error) {
	parsed, err := url.Parse(src)
	if err != nil {
		return nil, err
	}
	if parsed.Scheme != "https" {
		return nil, errors.New("--url must be https://")
	}

	// Wrap the caller-supplied client with our redirect policy so that
	// the redirect test exercises the exact rejection path used in
	// production fetchRates.
	wrapped := *client
	wrapped.Timeout = timeout
	wrapped.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if req.URL.Scheme != "https" {
			return errors.New("redirect to non-HTTPS rejected")
		}
		if len(via) >= 10 {
			return errors.New("stopped after 10 redirects")
		}
		return nil
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, src, nil)
	if err != nil {
		return nil, err
	}
	resp, err := wrapped.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || isTimeout(err) {
			return nil, errors.New("timeout after " + timeout.String())
		}
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	const maxBytes = 1 << 20
	body := make([]byte, 0, 4096)
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			body = append(body, buf[:n]...)
			if len(body) > maxBytes {
				return nil, errors.New("body too large")
			}
		}
		if err != nil {
			break
		}
	}
	return body, nil
}
