// Package provider auto-detects the cloud provider an agent is running
// on, by probing the standard instance-metadata endpoints. Used at
// agent startup to seed the ingero.provider resource attribute when no
// operator-supplied value is present (Fleet-side providerlookup still
// wins via mapping rules; this is the agent-side floor).
//
// v0.12.3 (Roadmap §4.5.1 attribution-(b)): closes the v0.11 cost-of-
// stragglers story for environments that don't ship a node_providers.yaml.
//
// Probe order: AWS (IMDSv2) → GCP → Azure. Each probe has a short
// timeout so a non-cloud host doesn't pay a multi-second startup tax.
package provider

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"regexp"
	"time"
)

// Provider is the canonical short name for a cloud. Values intentionally
// match the seed entries in examples/node_providers.yaml so an
// operator-supplied YAML and an auto-detected attribute carry the same
// label space.
type Provider string

const (
	ProviderAWS   Provider = "aws"
	ProviderGCP   Provider = "gcp"
	ProviderAzure Provider = "azure"
)

// Default per-probe timeout. Tunable via Detect's ctx; this constant is
// the floor used by DetectDefault.
const defaultProbeTimeout = 500 * time.Millisecond

// canonical IMDS endpoints. Held as vars so probeAWS/GCP/Azure tests
// can swap in httptest URLs without re-implementing the probe logic.
// Production callers go through Detect() / DetectDefault() and never
// see these.
const (
	awsTokenURL   = "http://169.254.169.254/latest/api/token"
	awsIDURL      = "http://169.254.169.254/latest/meta-data/instance-id"
	gcpInstanceID = "http://metadata.google.internal/computeMetadata/v1/instance/id"
	azureMetadata = "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
)

// awsInstanceIDRE asserts the AWS-shaped instance-id (i-<17 hex>).
// v0.12.4 (Sys Arch ★2): Azure shares 169.254.169.254 with AWS; if a
// future Azure-IMDS variant returns 200 to AWS's PUT-then-GET dance,
// this body-shape check rules out the false-positive.
var awsInstanceIDRE = regexp.MustCompile(`^i-[0-9a-f]{17}$`)

// Detect probes IMDS endpoints in order and returns the first match.
// Returns "" (empty Provider) if no probe succeeds within ctx's
// deadline. Network errors, timeouts, and non-2xx responses are all
// "no match"; only an explicit positive identification returns a value.
func Detect(ctx context.Context) Provider {
	c := httpClient()
	if p := probeAWS(ctx, c, awsTokenURL, awsIDURL); p != "" {
		return p
	}
	if p := probeGCP(ctx, c, gcpInstanceID); p != "" {
		return p
	}
	if p := probeAzure(ctx, c, azureMetadata); p != "" {
		return p
	}
	return ""
}

// DetectDefault is Detect with a per-probe timeout floor (1.5 s total
// worst-case across the three probes). Use this from agent startup
// where an external context isn't available.
func DetectDefault() Provider {
	ctx, cancel := context.WithTimeout(context.Background(), 3*defaultProbeTimeout)
	defer cancel()
	return Detect(ctx)
}

// httpClient builds an http.Client with a tight timeout that does NOT
// follow redirects (IMDS endpoints don't redirect; a redirect to
// somewhere else means we got captured by a transparent proxy).
func httpClient() *http.Client {
	return &http.Client{
		Timeout: defaultProbeTimeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
		Transport: &http.Transport{
			// Explicit Proxy: nil so a future refactor to
			// http.DefaultTransport.Clone() (which honors HTTP_PROXY)
			// can't silently re-route IMDS to an attacker-controlled
			// proxy. v0.12.3 Sec audit ★2.
			Proxy: nil,
			DialContext: (&net.Dialer{
				Timeout: defaultProbeTimeout,
			}).DialContext,
			DisableKeepAlives: true,
		},
	}
}

// probeAWS uses IMDSv2: PUT a token, then GET instance-id with the token.
// Old IMDSv1-only images return the instance-id directly on the GET path,
// but every AWS region honors v2 so v2 is sufficient for detection.
//
// v0.12.4 (Sys Arch ★2): the GET response body must match the
// AWS instance-id shape (i-<17 hex>). Azure shares 169.254.169.254;
// without the body-shape check, an Azure-IMDS variant that returns
// 200 to the AWS-shaped GET would false-positive as AWS.
//
// v0.12.4 (Sys Arch ★3): probeAWS takes the URL pair so tests can
// drive it against an httptest server instead of reimplementing.
func probeAWS(ctx context.Context, c *http.Client, tokenURL, idURL string) Provider {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, tokenURL, nil)
	if err != nil {
		return ""
	}
	req.Header.Set("X-aws-ec2-metadata-token-ttl-seconds", "60")
	resp, err := c.Do(req)
	if err != nil {
		return ""
	}
	tok, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || len(tok) == 0 {
		return ""
	}

	req, err = http.NewRequestWithContext(ctx, http.MethodGet, idURL, nil)
	if err != nil {
		return ""
	}
	req.Header.Set("X-aws-ec2-metadata-token", string(tok))
	resp, err = c.Do(req)
	if err != nil {
		return ""
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ""
	}
	if !awsInstanceIDRE.Match(trimSpace(body)) {
		// Body did not match the AWS shape -- could be an Azure
		// virtual-IP appliance proxying a 200 OK to a GET it doesn't
		// understand. Refuse to claim AWS.
		return ""
	}
	return ProviderAWS
}

// probeGCP queries metadata.google.internal which only resolves on GCE.
// The Metadata-Flavor: Google header is required for IMDS responses;
// without it the endpoint 403s.
func probeGCP(ctx context.Context, c *http.Client, url string) Provider {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return ""
	}
	req.Header.Set("Metadata-Flavor", "Google")
	resp, err := c.Do(req)
	if err != nil {
		return ""
	}
	io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		return ProviderGCP
	}
	return ""
}

// probeAzure queries the Azure IMDS endpoint (shares the AWS magic IP
// but distinguished by the api-version query parameter and Metadata: true
// header).
func probeAzure(ctx context.Context, c *http.Client, url string) Provider {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return ""
	}
	req.Header.Set("Metadata", "true")
	resp, err := c.Do(req)
	if err != nil {
		return ""
	}
	io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		return ProviderAzure
	}
	return ""
}

// ErrNoProvider is returned by callers wrapping Detect that want to
// distinguish "didn't try" from "tried, nothing matched". Detect itself
// returns "" for the latter; callers can wrap as needed.
var ErrNoProvider = errors.New("no cloud provider detected")

// trimSpace strips leading/trailing ASCII whitespace from a byte
// slice without allocating. Used by the AWS instance-id body-shape
// check so a "i-12345...\n" response matches.
func trimSpace(b []byte) []byte {
	i, j := 0, len(b)
	for i < j && isSpace(b[i]) {
		i++
	}
	for j > i && isSpace(b[j-1]) {
		j--
	}
	return b[i:j]
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}
