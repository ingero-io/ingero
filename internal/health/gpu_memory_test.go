package health

import (
	"context"
	"errors"
	"testing"
	"time"
)

// fakeRun returns a closure suitable as gpuMemReader.run, producing the
// given stdout and error.
func fakeRun(stdout string, runErr error) func(ctx context.Context) ([]byte, error) {
	return func(ctx context.Context) ([]byte, error) {
		return []byte(stdout), runErr
	}
}

func TestGPUMem_Unavailable(t *testing.T) {
	// Zero-value reader (no run func): Read must error without panicking.
	r := &gpuMemReader{timeout: time.Second, log: discardLogger()}
	if r.Available() {
		t.Fatal("Available() should be false when run is nil")
	}
	_, _, err := r.Read(context.Background())
	if err == nil {
		t.Fatal("expected error when nvidia-smi missing, got nil")
	}
}

func TestGPUMem_SingleGPU(t *testing.T) {
	r := &gpuMemReader{
		run:     fakeRun("4096, 16384\n", nil),
		timeout: time.Second,
		log:     discardLogger(),
	}
	used, total, err := r.Read(context.Background())
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if used != 4096 || total != 16384 {
		t.Errorf("used=%d total=%d, want 4096/16384", used, total)
	}
}

func TestGPUMem_MultiGPU(t *testing.T) {
	r := &gpuMemReader{
		run:     fakeRun("4096, 16384\n8192, 16384\n", nil),
		timeout: time.Second,
		log:     discardLogger(),
	}
	used, total, err := r.Read(context.Background())
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	// Sum across GPUs.
	if used != 12288 || total != 32768 {
		t.Errorf("used=%d total=%d, want 12288/32768", used, total)
	}
}

func TestGPUMem_EmptyOutput(t *testing.T) {
	r := &gpuMemReader{
		run:     fakeRun("", nil),
		timeout: time.Second,
		log:     discardLogger(),
	}
	if _, _, err := r.Read(context.Background()); err == nil {
		t.Error("expected error on empty output, got nil")
	}
}

func TestGPUMem_MalformedOutput(t *testing.T) {
	cases := []string{
		"not-a-number, 16384\n",
		"4096\n",                    // missing column
		"4096, 16384, extra\n",      // extra column
		"4096, not-a-number\n",
		"0, 0\n",                    // non-positive total
		"-1, 16384\n",               // negative used
	}
	for _, in := range cases {
		r := &gpuMemReader{
			run:     fakeRun(in, nil),
			timeout: time.Second,
			log:     discardLogger(),
		}
		if _, _, err := r.Read(context.Background()); err == nil {
			t.Errorf("expected error for input %q, got nil", in)
		}
	}
}

func TestGPUMem_SubprocessError(t *testing.T) {
	want := errors.New("exit 127")
	r := &gpuMemReader{
		run:     fakeRun("", want),
		timeout: time.Second,
		log:     discardLogger(),
	}
	_, _, err := r.Read(context.Background())
	if err == nil || !errors.Is(err, want) {
		t.Errorf("got %v, want wrapped %v", err, want)
	}
}

func TestGPUMem_Timeout(t *testing.T) {
	// A run that blocks until the injected ctx is cancelled; Read must
	// still return within a short window because of its own timeout.
	r := &gpuMemReader{
		run: func(ctx context.Context) ([]byte, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		},
		timeout: 50 * time.Millisecond,
		log:     discardLogger(),
	}
	start := time.Now()
	if _, _, err := r.Read(context.Background()); err == nil {
		t.Error("expected timeout error, got nil")
	}
	if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
		t.Errorf("Read did not honor timeout, took %v", elapsed)
	}
}

func TestGPUMem_RejectsOversizedOutput(t *testing.T) {
	// Adversarial review R1: a hostile nvidia-smi could flood stdout.
	// Parser must cap the input rather than materialize it into memory.
	huge := make([]byte, maxNvidiaSMIOutput+1)
	for i := range huge {
		huge[i] = 'A'
	}
	r := &gpuMemReader{
		run:     fakeRun(string(huge), nil),
		timeout: time.Second,
		log:     discardLogger(),
	}
	if _, _, err := r.Read(context.Background()); err == nil {
		t.Error("expected error on oversized output, got nil")
	}
}

func TestGPUMem_IgnoresBlankLines(t *testing.T) {
	// Some nvidia-smi builds add trailing blank lines.
	r := &gpuMemReader{
		run:     fakeRun("\n4096, 16384\n\n", nil),
		timeout: time.Second,
		log:     discardLogger(),
	}
	used, total, err := r.Read(context.Background())
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if used != 4096 || total != 16384 {
		t.Errorf("used=%d total=%d, want 4096/16384", used, total)
	}
}
