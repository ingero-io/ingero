package mcp

import (
	"net/http"

	"github.com/ingero-io/ingero/internal/auth"
)

// bearerAuth returns a middleware that requires every request carry
// `Authorization: Bearer <token>` matching the configured token
// (constant-time-compared after SHA-256 padding). Empty token disables
// the middleware (caller decides loopback vs http exposure).
//
// On reject:
//   - 401 with WWW-Authenticate: Bearer error="invalid_token"
//   - body is a single line so curl/jq output stays clean
//   - error reason ("missing", "malformed", "wrong") is NOT echoed; a
//     timing-channel-resistant compare answers all three the same
//     way to the network observer
func bearerAuth(next http.Handler, token string) http.Handler {
	if token == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hdr := r.Header.Get("Authorization")
		got, ok := auth.ParseBearer(hdr)
		if !ok || !auth.TokensEqual(got, token) {
			w.Header().Set("WWW-Authenticate", `Bearer error="invalid_token"`)
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte("unauthorized\n"))
			return
		}
		next.ServeHTTP(w, r)
	})
}
