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

// Detect probes IMDS endpoints in order and returns the first match.
// Returns "" (empty Provider) if no probe succeeds within ctx's
// deadline. Network errors, timeouts, and non-2xx responses are all
// "no match"; only an explicit positive identification returns a value.
func Detect(ctx context.Context) Provider {
	if p := detectAWS(ctx); p != "" {
		return p
	}
	if p := detectGCP(ctx); p != "" {
		return p
	}
	if p := detectAzure(ctx); p != "" {
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

// detectAWS uses IMDSv2: PUT a token, then GET instance-id with the token.
// Old IMDSv1-only images return the instance-id directly on the GET path,
// but every AWS region honors v2 so v2 is sufficient for detection.
func detectAWS(ctx context.Context) Provider {
	const tokenURL = "http://169.254.169.254/latest/api/token"
	const idURL = "http://169.254.169.254/latest/meta-data/instance-id"
	c := httpClient()

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
	io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		return ProviderAWS
	}
	return ""
}

// detectGCP queries metadata.google.internal which only resolves on GCE.
// The Metadata-Flavor: Google header is required for IMDS responses;
// without it the endpoint 403s.
func detectGCP(ctx context.Context) Provider {
	const url = "http://metadata.google.internal/computeMetadata/v1/instance/id"
	c := httpClient()
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

// detectAzure queries the Azure IMDS endpoint (shares the AWS magic IP
// but distinguished by the api-version query parameter and Metadata: true
// header).
func detectAzure(ctx context.Context) Provider {
	const url = "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
	c := httpClient()
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
