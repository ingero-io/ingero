//go:build linux

package store

import (
	"context"
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestAdversarial_ExecuteReadOnlyMoreExploits -- follow-up with CPU,
// sort stress, write-bypass attempts, and tricky string escapes.
func TestAdversarial_ExecuteReadOnlyMoreExploits(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	// Seed a tiny schema so we can try JOIN / ORDER BY stressors.
	type R struct {
		label string
		q     string
	}
	cases := []R{
		// CPU bomb: expensive computation per row.
		{"like-regex-nested", "SELECT * FROM pragma_function_list WHERE name LIKE \"%a%a%a%a%a%a%a%a%a%a%a%\""},
		// JSON_each on nested JSON — any O(n^2)?
		{"json-each-huge-array", "SELECT value FROM json_each(printf(\"[\" || (%.*c) || \"]\", 1000000, 49)) LIMIT 10"},

		// Write-bypass attempts: sneak keywords past the simple Contains check.
		{"upsert-sneak", "SELECT * FROM sqlite_master WHERE name = \"INSERTABLE\""}, // INSERTABLE contains "INSERT " (with space)? No, checks "INSERT " strictly.
		{"lowercase-write", "SELECT name FROM sqlite_master; --insert into events values (1)"},
		{"unicode-case-keyword", "sElEcT name FROM sqlite_master WHERE 1=1 and 1=1 union select name from sqlite_master where 1=1"},
		{"exfil-via-union", "SELECT name FROM sqlite_master UNION SELECT sqlite_version()"},

		// Sort order DoS — expensive ORDER BY on computed column.
		{"sort-complex", "SELECT abs(random() * random() * random()) as r FROM pragma_function_list ORDER BY r DESC"},

		// Deeply nested parens (parser stack depth)
		{"nested-subquery", "SELECT (SELECT (SELECT (SELECT (SELECT (SELECT (SELECT (SELECT 1))))))) as x"},

		// Very long identifier
		{"long-identifier", fmt.Sprintf("SELECT 1 as %s", strings.Repeat("a", 50000))},

		// Sneaky keywords: "insert" lowercase not caught? The validator uppercases.
		{"sneaky-lowercase", "select name from sqlite_master where 1=1"},

		// Backtick-quoted identifier with newline
		{"backtick-newline", "SELECT 1 as "},

		// Unicode quoting variants — the validator doesn not canonicalize quotes
		{"unicode-quote-literal", "SELECT \u201cINSERT\u201d as x"}, // smart quotes

		// A query ending with semicolon + whitespace (should pass)
		{"trailing-semicolon-ws", "SELECT 1 ;   "},

		// Empty
		{"empty", "   "},

		// Only whitespace semicolons
		{"semicolons-ws", ";;;  ;"},

		// Trying to inject via comment
		{"c-comment-inject", "SELECT 1 /* */ INSERT /* */ INTO events values(1)"},

		// Trying via line continuation
		{"line-cont-inject", "SELECT 1;\n\nDELETE FROM events"},

		// UNION with writeish string
		{"union-with-delete-literal", "SELECT \"DELETE\" as op"},
	}

	for _, tc := range cases {
		t.Run(tc.label, func(t *testing.T) {
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			start := time.Now()
			cols, rows, truncated, err := s.ExecuteReadOnly(ctx, tc.q, 10)
			elapsed := time.Since(start)
			runtime.ReadMemStats(&after)
			growth := float64(after.HeapAlloc-before.HeapAlloc) / 1024 / 1024
			_ = cols
			_ = truncated
			if err != nil {
				t.Logf("err (%s, ΔMB=%.1f): %v", elapsed, growth, err)
				if strings.Contains(err.Error(), "write operations") {
					t.Logf("  -> validator correctly rejected as write")
				}
				return
			}
			t.Logf("ok (%s, ΔMB=%.1f, rows=%d)", elapsed, growth, len(rows))
		})
	}
}
