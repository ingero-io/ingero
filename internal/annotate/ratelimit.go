package annotate

import "time"

// rateLimiter is a simple per-connection fixed-window rate limiter. One
// instance is created per connection in handleConn; it is not shared
// across connections, so it needs no mutex - a connection is handled by
// a single goroutine.
//
// The window is fixed (not sliding): the count resets when the window
// rolls. Fixed-window is sufficient here because the goal is to bound a
// runaway writer, not to enforce a precise smoothed rate.
type rateLimiter struct {
	limit       int
	window      time.Duration
	windowStart time.Time
	count       int
	now         func() time.Time // injectable for tests
}

// newRateLimiter builds a limiter allowing at most limit events per
// window.
func newRateLimiter(limit int, window time.Duration) *rateLimiter {
	return &rateLimiter{
		limit:  limit,
		window: window,
		now:    time.Now,
	}
}

// allow reports whether one more event is permitted in the current
// window. It rolls the window when the current one has elapsed.
func (r *rateLimiter) allow() bool {
	t := r.now()
	if r.windowStart.IsZero() || t.Sub(r.windowStart) >= r.window {
		r.windowStart = t
		r.count = 0
	}
	if r.count >= r.limit {
		return false
	}
	r.count++
	return true
}
