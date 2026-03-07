package cli

import (
	"testing"
)

func TestToUint32Slice(t *testing.T) {
	tests := []struct {
		name string
		in   []int
		want []uint32
	}{
		{"single", []int{123}, []uint32{123}},
		{"multiple", []int{123, 456}, []uint32{123, 456}},
		{"zero filtered", []int{0}, nil},
		{"nil input", nil, nil},
		{"mixed with zero", []int{123, 0, 456}, []uint32{123, 456}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := toUint32Slice(tt.in)
			if tt.want == nil {
				if got != nil {
					t.Errorf("toUint32Slice(%v) = %v, want nil", tt.in, got)
				}
				return
			}
			if len(got) != len(tt.want) {
				t.Fatalf("toUint32Slice(%v) len = %d, want %d", tt.in, len(got), len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("toUint32Slice(%v)[%d] = %d, want %d", tt.in, i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestSinglePIDOrZero(t *testing.T) {
	tests := []struct {
		name string
		in   []int
		want uint32
	}{
		{"single", []int{123}, 123},
		{"multiple", []int{123, 456}, 0},
		{"nil", nil, 0},
		{"empty", []int{}, 0},
		{"single zero", []int{0}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := singlePIDOrZero(tt.in)
			if got != tt.want {
				t.Errorf("singlePIDOrZero(%v) = %d, want %d", tt.in, got, tt.want)
			}
		})
	}
}

func TestPidSetFromInts(t *testing.T) {
	// nil input → nil (no filter)
	if got := pidSetFromInts(nil); got != nil {
		t.Errorf("pidSetFromInts(nil) = %v, want nil", got)
	}

	// empty input → nil
	if got := pidSetFromInts([]int{}); got != nil {
		t.Errorf("pidSetFromInts([]) = %v, want nil", got)
	}

	// single PID
	m := pidSetFromInts([]int{123})
	if m == nil {
		t.Fatal("pidSetFromInts([123]) returned nil")
	}
	if !m[123] {
		t.Error("expected 123 in set")
	}
	if m[999] {
		t.Error("expected 999 not in set")
	}

	// multiple PIDs
	m = pidSetFromInts([]int{123, 456})
	if !m[123] || !m[456] {
		t.Error("expected both 123 and 456 in set")
	}
	if m[789] {
		t.Error("expected 789 not in set")
	}

	// zeros filtered out
	m = pidSetFromInts([]int{0})
	if m != nil {
		t.Errorf("pidSetFromInts([0]) = %v, want nil", m)
	}
}
