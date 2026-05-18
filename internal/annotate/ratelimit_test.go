package annotate

import (
	"testing"
	"time"
)

func TestRateLimiter_BoundsWithinWindow(t *testing.T) {
	now := time.Unix(0, 0)
	rl := newRateLimiter(5, time.Second)
	rl.now = func() time.Time { return now }

	allowed := 0
	for i := 0; i < 20; i++ {
		if rl.allow() {
			allowed++
		}
	}
	if allowed != 5 {
		t.Errorf("allowed %d in one window, want 5", allowed)
	}
}

func TestRateLimiter_WindowRolls(t *testing.T) {
	now := time.Unix(0, 0)
	rl := newRateLimiter(3, time.Second)
	rl.now = func() time.Time { return now }

	for i := 0; i < 3; i++ {
		if !rl.allow() {
			t.Fatalf("event %d should be allowed", i)
		}
	}
	if rl.allow() {
		t.Fatal("fourth event in the window should be rejected")
	}

	// Advance past the window.
	now = now.Add(2 * time.Second)
	if !rl.allow() {
		t.Error("first event in the new window should be allowed")
	}
}
