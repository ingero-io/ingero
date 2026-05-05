package cli

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ingero-io/ingero/docs"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

// Default fetch URL for the canonical gpu_rates.yaml. Override with --url.
// Today this points at the example file in the ingero-fleet repo;
// when a dedicated catalog repo lands the URL will move and the
// embedded fallback in docs/gpu_rates.md will be retired.
//
// TRUST BOUNDARY (v0.14 R3 ★4): the URL points at `main` on a public
// repo. Anyone with merge rights to `ingero-fleet:main` can change
// the rate values, and every operator running `ingero rates update`
// will silently pick up the new file. The schema is validated; the
// content (USD/sec values) is NOT. A bad-faith or buggy PR landing
// `fallback_rate: 0.0001` flows into operator dashboards as silently-
// wrong cost telemetry. Operators who need stronger supply-chain
// integrity should pin --url to a specific commit SHA (paste the
// commit-pinned raw.githubusercontent URL) and review the diff
// before each update. v0.15 will move the catalog to a dedicated,
// release-signed repository (github.com/ingero-io/gpu-rates) and
// update this default; the trust posture will tighten then.
const defaultRatesURL = "https://raw.githubusercontent.com/ingero-io/ingero-fleet/main/examples/gpu_rates.yaml"

var (
	ratesURL     string
	ratesOutput  string
	ratesTimeout time.Duration
)

var ratesCmd = &cobra.Command{
	Use:   "rates",
	Short: "Manage the gpu_rates.yaml cost-attribution catalog",
	Long: `Operations on the gpu_rates.yaml file consumed by the v0.11+
cost-of-problem dashboard. The canonical catalog lives at
github.com/ingero-io/gpu-rates; ` + "`ingero rates update`" + ` fetches it.`,
}

var ratesUpdateCmd = &cobra.Command{
	Use:   "update",
	Short: "Fetch the latest gpu_rates.yaml from the public catalog",
	Long: `Download gpu_rates.yaml from the canonical catalog (default
github.com/ingero-io/gpu-rates), validate it, and write it atomically
to the output path.

The fetch is HTTPS-only. Plain HTTP is rejected before any network
call. The downloaded file is parsed as YAML and validated against
the expected schema before the output file is touched; on any error
the existing file (if present) is left unchanged.`,
	RunE: runRatesUpdate,
}

func init() {
	ratesUpdateCmd.Flags().StringVar(&ratesURL, "url", defaultRatesURL, "source URL (must be https://)")
	ratesUpdateCmd.Flags().StringVar(&ratesOutput, "output", "./gpu_rates.yaml", "output path")
	ratesUpdateCmd.Flags().DurationVar(&ratesTimeout, "timeout", 30*time.Second, "fetch timeout")

	ratesCmd.AddCommand(ratesUpdateCmd)
	rootCmd.AddCommand(ratesCmd)
}

func runRatesUpdate(cmd *cobra.Command, args []string) error {
	parsed, err := url.Parse(ratesURL)
	if err != nil {
		return fmt.Errorf("--url: %w", err)
	}
	if parsed.Scheme != "https" {
		return fmt.Errorf("--url must be https:// (got %q)", parsed.Scheme)
	}

	outDir := filepath.Dir(ratesOutput)
	if info, err := os.Stat(outDir); err != nil || !info.IsDir() {
		return fmt.Errorf("output: directory does not exist: %s", outDir)
	}

	// Fallback applies only when --url is left at the default. Explicit
	// --url overrides indicate the operator is targeting a specific
	// mirror or fork; failing loudly there is the right behavior.
	urlIsDefault := !cmd.Flags().Changed("url")

	body, source, err := loadRatesBody(cmd.Context(), ratesURL, ratesTimeout, urlIsDefault)
	if err != nil {
		return err
	}

	rates, err := parseAndValidate(body)
	if err != nil {
		return err
	}

	if err := atomicWrite(ratesOutput, body); err != nil {
		return fmt.Errorf("write: %w", err)
	}

	models := 0
	for _, m := range rates.Providers {
		models += len(m)
	}
	fmt.Fprintf(cmd.OutOrStdout(),
		"updated: %d providers, %d models, currency=%s (source=%s)\n",
		len(rates.Providers), models, rates.CurrencyName, source,
	)
	return nil
}

// loadRatesBody returns the YAML bytes plus a source label. It first
// tries the canonical URL; on any fetch error and only when fallback
// is allowed, it returns the embedded md fallback. Source is one of
// "url", "embedded-fallback".
func loadRatesBody(ctx context.Context, src string, timeout time.Duration, allowFallback bool) ([]byte, string, error) {
	body, fetchErr := fetchRates(ctx, src, timeout)
	if fetchErr == nil {
		return body, "url", nil
	}
	if !allowFallback {
		return nil, "", fmt.Errorf("fetch: %w", fetchErr)
	}
	fallback, fbErr := extractYAMLFromMarkdown(docs.GPURatesMD)
	if fbErr != nil {
		return nil, "", fmt.Errorf("fetch: %w (fallback: %v)", fetchErr, fbErr)
	}
	return fallback, "embedded-fallback", nil
}

// extractYAMLFromMarkdown returns the contents of the first ```yaml fenced
// code block in md. Anchored on \n```yaml\n / \n```\n so that arbitrary
// adjacent prose cannot accidentally produce a partial match.
func extractYAMLFromMarkdown(md []byte) ([]byte, error) {
	const open = "\n```yaml\n"
	const close = "\n```"
	// Normalize CRLF -> LF so a Windows checkout (or one produced by a
	// tool that ignores .gitattributes) still matches the LF-anchored
	// fence patterns above.
	s := strings.ReplaceAll(string(md), "\r\n", "\n")
	i := strings.Index(s, open)
	if i < 0 {
		return nil, errors.New("no ```yaml block found in fallback md")
	}
	start := i + len(open)
	rest := s[start:]
	j := strings.Index(rest, close)
	if j < 0 {
		return nil, errors.New("unterminated ```yaml block in fallback md")
	}
	// Include trailing newline so the YAML decoder sees a clean buffer.
	return []byte(rest[:j+1]), nil
}

// fetchRates issues the GET, enforcing HTTPS even across redirects.
// Returns the response body or an error tagged for the caller to wrap.
func fetchRates(ctx context.Context, src string, timeout time.Duration) ([]byte, error) {
	client := &http.Client{
		Timeout: timeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if req.URL.Scheme != "https" {
				return errors.New("redirect to non-HTTPS rejected")
			}
			if len(via) >= 10 {
				return errors.New("stopped after 10 redirects")
			}
			return nil
		},
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, src, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "ingero-rates-update")

	resp, err := client.Do(req)
	if err != nil {
		// Surface "context deadline exceeded" as "timeout after Xs" for clarity.
		if errors.Is(err, context.DeadlineExceeded) || isTimeout(err) {
			return nil, fmt.Errorf("timeout after %s", timeout)
		}
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	// Cap body read at 1 MiB. The seed file is ~1 KiB; a malicious or
	// misconfigured origin should not exhaust memory.
	const maxBytes = 1 << 20
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes))
	if err != nil {
		return nil, err
	}
	return body, nil
}

func isTimeout(err error) bool {
	var ne interface{ Timeout() bool }
	return errors.As(err, &ne) && ne.Timeout()
}

// gpuRatesFile mirrors the YAML schema operators consume.
type gpuRatesFile struct {
	CurrencyName   string                        `yaml:"currency_name"`
	CurrencySymbol string                        `yaml:"currency_symbol"`
	Providers      map[string]map[string]float64 `yaml:"providers"`
	FallbackRate   *float64                      `yaml:"fallback_rate"`
}

func parseAndValidate(body []byte) (*gpuRatesFile, error) {
	var rates gpuRatesFile
	dec := yaml.NewDecoder(bytes.NewReader(body))
	dec.KnownFields(false)
	if err := dec.Decode(&rates); err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}

	if rates.CurrencyName == "" {
		return nil, errors.New("validate: currency_name required")
	}
	if rates.CurrencySymbol == "" {
		return nil, errors.New("validate: currency_symbol required")
	}
	if len(rates.Providers) == 0 {
		return nil, errors.New("validate: at least one provider required")
	}
	if rates.FallbackRate == nil {
		return nil, errors.New("validate: fallback_rate required")
	}
	if *rates.FallbackRate < 0 {
		return nil, fmt.Errorf("validate: fallback_rate must be >= 0 (got %g)", *rates.FallbackRate)
	}
	return &rates, nil
}

// atomicWrite stages bytes to a sibling tmp file in the target's directory
// and renames into place. On any failure the target is left unchanged and
// the tmp file is removed.
func atomicWrite(path string, data []byte) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".gpu_rates.*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}
