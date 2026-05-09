// Package auth holds bearer-token parsing + constant-time compare.
// Agent-local copy; the fleet repo has the same shape under
// ingero-fleet/internal/auth. They are intentionally duplicated
// because the dependency rule keeps the ingero agent self-contained
// (no imports from ingero-fleet). Drift is policed by parity tests
// at release time.
package auth

import (
	"crypto/sha256"
	"crypto/subtle"
	"strings"
)

// ParseBearer extracts the token from an RFC 7235 Authorization
// header. Scheme match is case-insensitive ("Bearer" / "bearer" /
// "BEARER"); whitespace between scheme and token is tolerated.
// Returns the trimmed token and ok=true on match. Returns ok=false
// on missing scheme, wrong scheme, empty token, or header shorter
// than the scheme literal.
func ParseBearer(h string) (string, bool) {
	if len(h) < 6 {
		return "", false
	}
	if !strings.EqualFold(h[:6], "bearer") {
		return "", false
	}
	rest := strings.TrimSpace(h[6:])
	if rest == "" {
		return "", false
	}
	return rest, true
}

// TokensEqual reports whether got and want are the same bearer
// token in constant time. Both inputs are SHA-256-hashed before
// compare so the wall-clock cost does NOT depend on the input
// lengths. Plain subtle.ConstantTimeCompare returns 0 immediately
// on length mismatch, leaking the wanted-token's byte length to a
// timing attacker who can probe many requests; padding both sides
// to a 32-byte digest closes that side channel.
//
// Empty want rejected unconditionally (defense in depth against a
// future re-wiring footgun where a caller passes an empty token
// without the empty-check).
func TokensEqual(got, want string) bool {
	if want == "" {
		return false
	}
	gh := sha256.Sum256([]byte(got))
	wh := sha256.Sum256([]byte(want))
	return subtle.ConstantTimeCompare(gh[:], wh[:]) == 1
}
