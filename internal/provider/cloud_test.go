package provider

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// The Detect probes hit hard-coded IMDS endpoints (169.254.169.254 +
// metadata.google.internal). On a non-cloud host both fail/time out and
// Detect returns "". The unit tests below exercise the per-probe
// helpers indirectly by spinning up local httptest servers and pointing
// the per-detector logic at them via internal-testing helpers below.

// v0.12.4 (Sys Arch ★3): tests now drive the production probeAWS /
// probeGCP / probeAzure functions directly via the URL-injection
// parameters. The previous probeAWSAt/probeGCPAt/probeAzureAt
// reimplementations have been removed -- they could drift away from
// the real probes silently. The fake handlers below remain as the
// httptest fixture; the actual probe code is exercised in production form.

func TestDetectDefault_NonCloud(t *testing.T) {
	// Real call against a sandbox / dev box should return "" within the
	// budget. Skip if we somehow happen to be on a real cloud (CI rarely is).
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	got := Detect(ctx)
	// We can't assert "must be empty" reliably (CI on AWS would lie), but
	// we can assert the function returns within a bounded time without
	// panicking. If the platform really is AWS/GCP/Azure, an empty
	// assertion would be wrong — so we just confirm a valid return.
	switch got {
	case "", ProviderAWS, ProviderGCP, ProviderAzure:
		// fine
	default:
		t.Fatalf("unexpected provider: %q", got)
	}
}

func TestDetect_TimeoutBudget(t *testing.T) {
	// Already-cancelled ctx must short-circuit and return "".
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if got := Detect(ctx); got != "" {
		t.Fatalf("expected empty on cancelled ctx, got %q", got)
	}
}

// fakeAWSHandler mimics the IMDSv2 dance: PUT /latest/api/token returns
// a token; GET /latest/meta-data/instance-id with the token returns 200.
func fakeAWSHandler(t *testing.T) http.Handler {
	t.Helper()
	const tok = "AQAB-fake-token"
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPut && strings.HasSuffix(r.URL.Path, "/latest/api/token"):
			if r.Header.Get("X-aws-ec2-metadata-token-ttl-seconds") == "" {
				http.Error(w, "missing TTL header", 400)
				return
			}
			w.WriteHeader(200)
			w.Write([]byte(tok))
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/latest/meta-data/instance-id"):
			if r.Header.Get("X-aws-ec2-metadata-token") != tok {
				http.Error(w, "bad token", http.StatusUnauthorized)
				return
			}
			w.WriteHeader(200)
			// v0.12.4 (Sys Arch ★2): canonical AWS instance-id shape
			// (i-<17 hex>) so probeAWS's body-shape check accepts.
			w.Write([]byte("i-0123456789abcdef0\n"))
		default:
			http.NotFound(w, r)
		}
	})
}

func fakeGCPHandler(t *testing.T) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Metadata-Flavor") != "Google" {
			http.Error(w, "missing flavor header", http.StatusForbidden)
			return
		}
		if !strings.HasSuffix(r.URL.Path, "/computeMetadata/v1/instance/id") {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(200)
		w.Write([]byte("12345"))
	})
}

func fakeAzureHandler(t *testing.T) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Metadata") != "true" {
			http.Error(w, "missing Metadata header", 400)
			return
		}
		if r.URL.Query().Get("api-version") == "" {
			http.Error(w, "missing api-version", 400)
			return
		}
		w.WriteHeader(200)
		w.Write([]byte("{\"compute\":{\"vmId\":\"fake\"}}"))
	})
}

func TestProbeAWS(t *testing.T) {
	srv := httptest.NewServer(fakeAWSHandler(t))
	defer srv.Close()
	c := httpClient()
	got := probeAWS(context.Background(), c, srv.URL+"/latest/api/token", srv.URL+"/latest/meta-data/instance-id")
	if got != ProviderAWS {
		t.Fatalf("AWS probe: got %q want %q", got, ProviderAWS)
	}
}

func TestProbeGCP(t *testing.T) {
	srv := httptest.NewServer(fakeGCPHandler(t))
	defer srv.Close()
	c := httpClient()
	if got := probeGCP(context.Background(), c, srv.URL+"/computeMetadata/v1/instance/id"); got != ProviderGCP {
		t.Fatalf("GCP probe: got %q want %q", got, ProviderGCP)
	}
}

func TestProbeAzure(t *testing.T) {
	srv := httptest.NewServer(fakeAzureHandler(t))
	defer srv.Close()
	c := httpClient()
	if got := probeAzure(context.Background(), c, srv.URL+"/metadata/instance?api-version=2021-02-01"); got != ProviderAzure {
		t.Fatalf("Azure probe: got %q want %q", got, ProviderAzure)
	}
}

func TestProbe_404IsEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()
	c := httpClient()
	if got := probeAWS(context.Background(), c, srv.URL+"/latest/api/token", srv.URL+"/latest/meta-data/instance-id"); got != "" {
		t.Fatalf("expected empty on 404, got %q", got)
	}
	if got := probeGCP(context.Background(), c, srv.URL); got != "" {
		t.Fatalf("expected empty on 404, got %q", got)
	}
	if got := probeAzure(context.Background(), c, srv.URL); got != "" {
		t.Fatalf("expected empty on 404, got %q", got)
	}
}

func TestAWSProbe_RejectsMissingHeader(t *testing.T) {
	// Server requires Metadata-Flavor header (mimicking GCP), so AWS
	// probe (which sends an AWS token header) should NOT match.
	srv := httptest.NewServer(fakeGCPHandler(t))
	defer srv.Close()
	c := httpClient()
	if got := probeAWS(context.Background(), c, srv.URL+"/latest/api/token", srv.URL+"/latest/meta-data/instance-id"); got != "" {
		t.Fatalf("AWS probe must not match a GCP server; got %q", got)
	}
}

func TestAWSProbe_RejectsWrongInstanceIDShape(t *testing.T) {
	// v0.12.4 (Sys Arch ★2): a server that returns 200 to both the
	// token PUT and the instance-id GET but responds with non-AWS
	// shape (e.g. an Azure proxy) must NOT match AWS.
	const tok = "AQAB-fake-token"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPut && strings.HasSuffix(r.URL.Path, "/latest/api/token"):
			w.WriteHeader(200)
			w.Write([]byte(tok))
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/latest/meta-data/instance-id"):
			w.WriteHeader(200)
			w.Write([]byte("not-an-aws-id\n")) // Azure-shape impostor
		}
	}))
	defer srv.Close()
	c := httpClient()
	if got := probeAWS(context.Background(), c, srv.URL+"/latest/api/token", srv.URL+"/latest/meta-data/instance-id"); got != "" {
		t.Fatalf("AWS probe must reject non-i-<17hex> body; got %q", got)
	}
}
