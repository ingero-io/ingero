package health

import (
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/contract"
)

func newHTTPHeaders(kv map[string]string) http.Header {
	h := http.Header{}
	for k, v := range kv {
		h.Set(k, v)
	}
	return h
}

var cacheNow = time.Date(2026, 4, 16, 12, 0, 0, 0, time.UTC)

func TestNewCache_InitiallyEmpty(t *testing.T) {
	c := NewThresholdCache()
	if _, ok := c.Get(); ok {
		t.Fatal("new cache should return ok=false")
	}
	if c.PiggybackAvailable() {
		t.Fatal("new cache should report piggyback as unavailable")
	}
}

func TestParseAndSet_ValidHeaders(t *testing.T) {
	c := NewThresholdCache()
	h := newHTTPHeaders(map[string]string{
		contract.HeaderThreshold: "0.82",
		contract.HeaderQuorumMet: "true",
	})
	if !c.ParseAndSetHTTPHeaders(h, cacheNow) {
		t.Fatal("valid headers should be accepted")
	}
	snap, ok := c.Get()
	if !ok {
		t.Fatal("expected ok=true after successful parse")
	}
	if snap.Value != 0.82 {
		t.Fatalf("value = %v, want 0.82", snap.Value)
	}
	if !snap.QuorumMet {
		t.Fatal("quorum_met should be true")
	}
	if !snap.ReceivedAt.Equal(cacheNow) {
		t.Fatalf("receivedAt = %v, want %v", snap.ReceivedAt, cacheNow)
	}
	if !c.PiggybackAvailable() {
		t.Fatal("piggyback should be available after valid parse")
	}
	hits, _, _ := c.Stats()
	if hits != 1 {
		t.Fatalf("hits = %d, want 1", hits)
	}
}

func TestParseAndSet_MissingHeaders_Miss(t *testing.T) {
	c := NewThresholdCache()
	h := newHTTPHeaders(nil)
	if c.ParseAndSetHTTPHeaders(h, cacheNow) {
		t.Fatal("empty headers should not update cache")
	}
	if c.PiggybackAvailable() {
		t.Fatal("piggyback should be unavailable")
	}
	_, misses, _ := c.Stats()
	if misses != 1 {
		t.Fatalf("misses = %d, want 1", misses)
	}
}

func TestParseAndSet_PartialHeaders_TreatedAsAbsent(t *testing.T) {
	c := NewThresholdCache()
	h := newHTTPHeaders(map[string]string{
		contract.HeaderThreshold: "0.82",
		// quorum_met missing
	})
	if c.ParseAndSetHTTPHeaders(h, cacheNow) {
		t.Fatal("partial header pair should be treated as absent")
	}
	if c.PiggybackAvailable() {
		t.Fatal("piggyback should be unavailable for partial pair")
	}
}

func TestParseAndSet_OutOfBoundsRejected(t *testing.T) {
	cases := []string{"0.05", "0.999", "0", "1.0", "-0.5", "1.5"}
	for _, v := range cases {
		t.Run("val_"+v, func(t *testing.T) {
			c := NewThresholdCache()
			h := newHTTPHeaders(map[string]string{
				contract.HeaderThreshold: v,
				contract.HeaderQuorumMet: "true",
			})
			if c.ParseAndSetHTTPHeaders(h, cacheNow) {
				t.Fatalf("value %q should be rejected", v)
			}
			// Piggyback IS available even on sanity rejection — the
			// server did send headers, they just don't pass our bounds.
			if !c.PiggybackAvailable() {
				t.Fatal("piggyback should remain available (server sent headers)")
			}
			// Cache still has no valid threshold.
			if _, ok := c.Get(); ok {
				t.Fatal("cache should not hold an invalid threshold")
			}
			_, _, rejected := c.Stats()
			if rejected != 1 {
				t.Fatalf("rejected count = %d, want 1", rejected)
			}
		})
	}
}

// After unification: malformed threshold is treated as "headers were
// present but unusable" — piggyback stays available (server did attempt
// to deliver), rejected counter increments, poller stays suspended.
// Empty/whitespace headers are the one exception: they're treated as
// ABSENT, since there's nothing to distinguish them from no-headers.
func TestParseAndSet_MalformedThreshold_RejectedKeepsPiggyback(t *testing.T) {
	cases := []string{"abc", "NaN", "Inf", "1e400"}
	for _, v := range cases {
		t.Run("val_"+v, func(t *testing.T) {
			c := NewThresholdCache()
			h := newHTTPHeaders(map[string]string{
				contract.HeaderThreshold: v,
				contract.HeaderQuorumMet: "true",
			})
			c.ParseAndSetHTTPHeaders(h, cacheNow)
			if !c.PiggybackAvailable() {
				t.Fatalf("malformed %q should keep piggyback available (server sent headers)", v)
			}
			_, _, rejected := c.Stats()
			if rejected != 1 {
				t.Fatalf("rejected count = %d, want 1", rejected)
			}
		})
	}
}

// Empty/whitespace threshold with a valid quorum header is a partial
// pair, treated as absent (piggyback unavailable, miss).
func TestParseAndSet_EmptyThreshold_TreatedAsAbsent(t *testing.T) {
	for _, v := range []string{"", "  "} {
		t.Run("val_"+v, func(t *testing.T) {
			c := NewThresholdCache()
			h := newHTTPHeaders(map[string]string{
				contract.HeaderThreshold: v,
				contract.HeaderQuorumMet: "true",
			})
			c.ParseAndSetHTTPHeaders(h, cacheNow)
			if c.PiggybackAvailable() {
				t.Fatalf("empty threshold should be treated as absent: piggyback must be unavailable")
			}
			_, misses, _ := c.Stats()
			if misses != 1 {
				t.Fatalf("misses = %d, want 1", misses)
			}
		})
	}
}

// Malformed quorum_met keeps piggyback available (server sent headers,
// just unusable).
func TestParseAndSet_MalformedQuorum_RejectedKeepsPiggyback(t *testing.T) {
	c := NewThresholdCache()
	h := newHTTPHeaders(map[string]string{
		contract.HeaderThreshold: "0.82",
		contract.HeaderQuorumMet: "yes",
	})
	c.ParseAndSetHTTPHeaders(h, cacheNow)
	if !c.PiggybackAvailable() {
		t.Fatal("malformed quorum should keep piggyback available")
	}
	_, _, rejected := c.Stats()
	if rejected != 1 {
		t.Fatalf("rejected count = %d, want 1", rejected)
	}
}

// ParseAndSetHTTPHeaders with nil header argument does not panic.
func TestParseAndSet_NilHeader_TreatedAsAbsent(t *testing.T) {
	c := NewThresholdCache()
	// http.Header is a map type; passing nil must not panic.
	ok := c.ParseAndSetHTTPHeaders(nil, cacheNow)
	if ok {
		t.Fatal("nil headers should not successfully parse")
	}
	if c.PiggybackAvailable() {
		t.Fatal("nil headers should mark piggyback unavailable")
	}
}

func TestParseAndSet_MalformedQuorum_Rejected(t *testing.T) {
	cases := []string{"yes", "1", "True", "FALSE"}
	// Note: "True" and "FALSE" ARE accepted (case-insensitive), so they
	// should succeed. "yes" and "1" should fail strict parse.
	for _, v := range cases {
		t.Run("val_"+v, func(t *testing.T) {
			c := NewThresholdCache()
			h := newHTTPHeaders(map[string]string{
				contract.HeaderThreshold: "0.82",
				contract.HeaderQuorumMet: v,
			})
			applied := c.ParseAndSetHTTPHeaders(h, cacheNow)
			expectAccept := strings.EqualFold(v, "true") || strings.EqualFold(v, "false")
			if expectAccept && !applied {
				t.Fatalf("quorum_met %q should be accepted (case-insensitive)", v)
			}
			if !expectAccept && applied {
				t.Fatalf("strict quorum_met: %q should be rejected", v)
			}
		})
	}
}

func TestParseAndSet_QuorumFalse_StillMarksPiggybackAvailable(t *testing.T) {
	c := NewThresholdCache()
	h := newHTTPHeaders(map[string]string{
		contract.HeaderThreshold: "0.82",
		contract.HeaderQuorumMet: "false",
	})
	if !c.ParseAndSetHTTPHeaders(h, cacheNow) {
		t.Fatal("quorum=false is a legitimate response; should be accepted")
	}
	snap, ok := c.Get()
	if !ok || snap.QuorumMet {
		t.Fatalf("want QuorumMet=false, got snap=%+v ok=%v", snap, ok)
	}
}

func TestMarkPiggybackUnavailable_IncrementsMisses(t *testing.T) {
	c := NewThresholdCache()
	c.MarkPiggybackUnavailable()
	c.MarkPiggybackUnavailable()
	_, misses, _ := c.Stats()
	if misses != 2 {
		t.Fatalf("misses = %d, want 2", misses)
	}
}

func TestCache_ConcurrentAccess(t *testing.T) {
	c := NewThresholdCache()
	var wg sync.WaitGroup
	// Writer via Set.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 1000; i++ {
			c.Set(0.5+float64(i%4)*0.1, i%2 == 0, cacheNow)
		}
	}()
	// Writer via ParseAndSetHTTPHeaders.
	wg.Add(1)
	go func() {
		defer wg.Done()
		h := newHTTPHeaders(map[string]string{
			contract.HeaderThreshold: "0.82",
			contract.HeaderQuorumMet: "true",
		})
		for i := 0; i < 1000; i++ {
			c.ParseAndSetHTTPHeaders(h, cacheNow)
		}
	}()
	// Readers.
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				_, _ = c.Get()
				_ = c.PiggybackAvailable()
				_, _, _ = c.Stats()
			}
		}()
	}
	wg.Wait()
	// Just assert we didn't crash; race detector would catch torn reads.
	if !c.PiggybackAvailable() {
		t.Fatal("expected piggyback available after writer finished")
	}
}

func TestParseBoolStrict(t *testing.T) {
	cases := map[string]struct {
		v  bool
		ok bool
	}{
		"true":  {true, true},
		"True":  {true, true},
		"TRUE":  {true, true},
		"false": {false, true},
		"False": {false, true},
		"yes":   {false, false},
		"1":     {false, false},
		"":      {false, false},
	}
	for in, want := range cases {
		v, ok := parseBoolStrict(in)
		if v != want.v || ok != want.ok {
			t.Errorf("parseBoolStrict(%q) = (%v, %v), want (%v, %v)", in, v, ok, want.v, want.ok)
		}
	}
}
