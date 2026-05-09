package cli

import (
	"context"
	"log/slog"

	"github.com/ingero-io/ingero/internal/ebpf/memfrag"
)

// startMemfragTracer attaches the v0.15 W1 memfrag IOCTL kprobe and
// spawns a goroutine that drains its ringbuf into the per-cmd
// counters. Failure to attach is logged at info and the function
// returns; the agent does not crash. Caller has already verified
// the experimental-kprobes flag + allowlist gate.
func startMemfragTracer(ctx context.Context, log *slog.Logger) {
	tr := memfrag.New()
	if err := tr.Attach(); err != nil {
		log.Info("memfrag tracer: attach failed; counter will stay empty", "err", err)
		return
	}
	go func() {
		defer tr.Close()
		// Run blocks until ctx is cancelled or the ringbuf reader
		// is closed. Errors are logged at debug.
		errCh := make(chan error, 1)
		go func() { errCh <- tr.Run(ctx) }()
		for {
			select {
			case <-ctx.Done():
				return
			case ev, ok := <-tr.Events():
				if !ok {
					return
				}
				recordMemfragEvent(ev)
			case err := <-errCh:
				if err != nil {
					log.Debug("memfrag tracer: Run exited", "err", err)
				}
				return
			}
		}
	}()
	log.Info("memfrag tracer: attached to nvidia_unlocked_ioctl")
}
