package health

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// parseRetryAfterHeader decodes the two RFC 7231 forms of the
// Retry-After header (delay-seconds or HTTP-date). Shared by the GET
// poller and the POST emitter so both honor the header with identical
// semantics.
//
// Returns (duration, ok). `ok=false` means the value was empty or
// malformed; callers should fall back to a configured default rather
// than treating it as zero.
func parseRetryAfterHeader(s string) (time.Duration, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	if n, err := strconv.Atoi(s); err == nil && n >= 0 {
		return time.Duration(n) * time.Second, true
	}
	if t, err := http.ParseTime(s); err == nil {
		d := time.Until(t)
		if d < 0 {
			d = 0
		}
		return d, true
	}
	return 0, false
}

// RetryAfterError wraps a non-2xx push response whose Retry-After
// header the caller is expected to honor. The loop uses this to delay
// the next tick instead of firing on the fixed push-interval cadence.
// Any other non-2xx (including rate-limit responses without the
// header) produce a plain status error.
type RetryAfterError struct {
	// StatusCode is the HTTP status that carried the Retry-After header
	// (typically 429 or 503).
	StatusCode int
	// Delay is the backoff the server requested.
	Delay time.Duration
}

func (e *RetryAfterError) Error() string {
	return fmt.Sprintf("push rejected: %d with Retry-After=%s", e.StatusCode, e.Delay)
}

// AsRetryAfter extracts a RetryAfterError from err, or returns nil.
// Convenience for callers that want to respect backoff on success of
// errors.As without boilerplate.
func AsRetryAfter(err error) *RetryAfterError {
	var r *RetryAfterError
	if errors.As(err, &r) {
		return r
	}
	return nil
}
