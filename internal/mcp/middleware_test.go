package mcp

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func okHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
}

func TestBearerAuth_EmptyTokenDisablesMiddleware(t *testing.T) {
	h := bearerAuth(okHandler(), "")
	req := httptest.NewRequest("POST", "/mcp", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("empty-token: code=%d, want 200", rec.Code)
	}
}

func TestBearerAuth_RejectsMissingHeader(t *testing.T) {
	h := bearerAuth(okHandler(), "secret-token")
	req := httptest.NewRequest("POST", "/mcp", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("missing-header: code=%d, want 401", rec.Code)
	}
	if got := rec.Header().Get("WWW-Authenticate"); !strings.Contains(got, "Bearer") {
		t.Errorf("missing WWW-Authenticate Bearer challenge: %q", got)
	}
}

func TestBearerAuth_RejectsWrongScheme(t *testing.T) {
	h := bearerAuth(okHandler(), "secret-token")
	req := httptest.NewRequest("POST", "/mcp", nil)
	req.Header.Set("Authorization", "Basic dXNlcjpwYXNz")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("wrong-scheme: code=%d, want 401", rec.Code)
	}
}

func TestBearerAuth_RejectsWrongToken(t *testing.T) {
	h := bearerAuth(okHandler(), "secret-token")
	req := httptest.NewRequest("POST", "/mcp", nil)
	req.Header.Set("Authorization", "Bearer wrong-token")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("wrong-token: code=%d, want 401", rec.Code)
	}
}

func TestBearerAuth_AcceptsCorrectToken(t *testing.T) {
	h := bearerAuth(okHandler(), "secret-token")
	req := httptest.NewRequest("POST", "/mcp", nil)
	req.Header.Set("Authorization", "Bearer secret-token")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("correct-token: code=%d, want 200", rec.Code)
	}
	body, _ := io.ReadAll(rec.Body)
	if string(body) != "ok" {
		t.Errorf("correct-token body=%q, want %q", body, "ok")
	}
}

func TestBearerAuth_AcceptsLowercaseScheme(t *testing.T) {
	h := bearerAuth(okHandler(), "secret-token")
	req := httptest.NewRequest("POST", "/mcp", nil)
	req.Header.Set("Authorization", "bearer secret-token")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("lowercase scheme: code=%d, want 200", rec.Code)
	}
}

// Reject-path body is a single short line (no internal detail
// echoed to the network).
func TestBearerAuth_RejectBodyIsSingleLine(t *testing.T) {
	h := bearerAuth(okHandler(), "secret-token")
	req := httptest.NewRequest("POST", "/mcp", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	body, _ := io.ReadAll(rec.Body)
	if strings.Count(string(body), "\n") > 1 {
		t.Errorf("reject body should be single line: %q", body)
	}
	if strings.Contains(strings.ToLower(string(body)), "secret") {
		t.Errorf("reject body must not echo wanted token: %q", body)
	}
}
