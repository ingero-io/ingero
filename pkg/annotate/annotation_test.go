package annotate

import (
	"strings"
	"testing"

	"github.com/ingero-io/ingero/pkg/contract"
)

func TestProcessIncarnation_Scoped(t *testing.T) {
	if (ProcessIncarnation{}).Scoped() {
		t.Error("zero incarnation should be unscoped")
	}
	if !(ProcessIncarnation{PID: 1}).Scoped() {
		t.Error("incarnation with PID should be scoped")
	}
	if got := (ProcessIncarnation{}).String(); got != "unscoped" {
		t.Errorf("String() = %q, want unscoped", got)
	}
	if got := (ProcessIncarnation{PID: 7, StartTime: 99}).String(); got != "pid=7/start=99" {
		t.Errorf("String() = %q", got)
	}
}

func TestAnnotation_IsSpan(t *testing.T) {
	if (Annotation{}).IsSpan() {
		t.Error("annotation with no span fields is not a span")
	}
	if !(Annotation{SpanStartNs: 1, SpanEndNs: 2}).IsSpan() {
		t.Error("annotation with span fields is a span")
	}
}

func TestAnnotation_Validate(t *testing.T) {
	good := Annotation{
		TimestampNs: 1,
		Labels:      map[string]string{"step": "10", "epoch": "1"},
	}
	if err := good.Validate(); err != nil {
		t.Fatalf("valid annotation rejected: %v", err)
	}

	span := good
	span.SpanStartNs, span.SpanEndNs = 100, 200
	if err := span.Validate(); err != nil {
		t.Fatalf("valid span rejected: %v", err)
	}

	cases := []struct {
		name string
		a    Annotation
	}{
		{"no labels", Annotation{Labels: nil}},
		{"empty labels", Annotation{Labels: map[string]string{}}},
		{"bad key", Annotation{Labels: map[string]string{"bad key": "v"}}},
		{"oversized value", Annotation{Labels: map[string]string{
			"k": strings.Repeat("x", contract.AnnotationMaxLabelValueLen+1),
		}}},
		{"half span start", Annotation{
			Labels: map[string]string{"k": "v"}, SpanStartNs: 5,
		}},
		{"half span end", Annotation{
			Labels: map[string]string{"k": "v"}, SpanEndNs: 5,
		}},
		{"reversed span", Annotation{
			Labels: map[string]string{"k": "v"}, SpanStartNs: 9, SpanEndNs: 2,
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.a.Validate(); err == nil {
				t.Errorf("expected %s to fail validation", tc.name)
			}
		})
	}
}

func TestAnnotation_Validate_TooManyLabels(t *testing.T) {
	labels := make(map[string]string, contract.AnnotationMaxLabelsPerAnnotation+1)
	for i := 0; i <= contract.AnnotationMaxLabelsPerAnnotation; i++ {
		labels["k"+string(rune('a'+i%26))+string(rune('a'+i/26))] = "v"
	}
	a := Annotation{Labels: labels}
	if err := a.Validate(); err == nil {
		t.Error("expected too-many-labels annotation to fail validation")
	}
}
