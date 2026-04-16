package cli

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
	"unicode/utf8"

	"github.com/ingero-io/ingero/internal/health"
	"github.com/spf13/cobra"
)

// validDetectionModes is the allow-list for the --fleet-detection-mode
// flag's value. Matches the five modes defined for Story 3.3.
var validDetectionModes = map[string]struct{}{
	"none":           {},
	"fleet":          {},
	"fleet-cached":   {},
	"local-cached":   {},
	"local-baseline": {},
}

var (
	fleetPushEndpoint      string
	fleetPushClusterID     string
	fleetPushNodeID        string
	fleetPushWorkloadType  string
	fleetPushInterval      time.Duration
	fleetPushTimeout       time.Duration
	fleetPushInsecure      bool
	fleetPushTLSCA         string
	fleetPushTLSCert       string
	fleetPushTLSKey        string
	fleetPushPersistPath   string
	fleetPushStaleAge      time.Duration
	fleetPushDetectionMode string
	fleetPushWorldSize     int
	fleetPushNodeRank      int
	fleetPushStubCollector bool
)

var fleetPushCmd = &cobra.Command{
	Use:   "fleet-push",
	Short: "Push health scores to Fleet (experimental, Epic 2 integration)",
	Long: `Push per-node health scores to an Ingero Fleet instance (or any OTLP/HTTP
collector) for peer-relative straggler detection.

This is the Epic 2 integration entry point. Without --fleet-endpoint the
agent behaves exactly as before — nothing about the default 'trace',
'explain', or 'query' workflows is changed.

Real signal collection from CUDA/CPU sources lands in a later story. For
now, pass --stub to drive the push loop with synthetic constant signals
(useful for smoke-testing a Fleet endpoint).`,
	RunE: runFleetPush,
}

func init() {
	fleetPushCmd.Flags().StringVar(&fleetPushEndpoint, "fleet-endpoint", "",
		"Fleet OTLP/HTTP endpoint, e.g. fleet.example:4318 or https://fleet.example/v1/metrics (required)")
	fleetPushCmd.Flags().StringVar(&fleetPushClusterID, "fleet-cluster-id", "",
		"Logical cluster identifier — Fleet keeps a separate score map per cluster (required)")
	fleetPushCmd.Flags().StringVar(&fleetPushNodeID, "fleet-node-id", "",
		"Stable node identifier. Defaults to $HOSTNAME if empty.")
	fleetPushCmd.Flags().StringVar(&fleetPushWorkloadType, "fleet-workload-type", "unknown",
		"Workload label attached to every push: training, inference, or unknown")
	fleetPushCmd.Flags().DurationVar(&fleetPushInterval, "fleet-push-interval", 10*time.Second,
		"Push cadence")
	fleetPushCmd.Flags().DurationVar(&fleetPushTimeout, "fleet-timeout", 5*time.Second,
		"Timeout per push request")
	fleetPushCmd.Flags().BoolVar(&fleetPushInsecure, "fleet-insecure", false,
		"Use HTTP instead of HTTPS when fleet-endpoint is bare host:port")
	fleetPushCmd.Flags().StringVar(&fleetPushTLSCA, "fleet-tls-ca", "",
		"Path to CA cert for mTLS")
	fleetPushCmd.Flags().StringVar(&fleetPushTLSCert, "fleet-tls-cert", "",
		"Path to client cert for mTLS")
	fleetPushCmd.Flags().StringVar(&fleetPushTLSKey, "fleet-tls-key", "",
		"Path to client key for mTLS")
	fleetPushCmd.Flags().StringVar(&fleetPushPersistPath, "fleet-persist-path",
		health.DefaultPersistencePath,
		"Path for baseline persistence across restarts")
	fleetPushCmd.Flags().DurationVar(&fleetPushStaleAge, "fleet-persist-stale-age", health.DefaultStaleAge,
		"Baseline file older than this is discarded on startup")
	fleetPushCmd.Flags().StringVar(&fleetPushDetectionMode, "fleet-detection-mode", "none",
		"Placeholder detection mode label (lands in Story 3.3)")
	fleetPushCmd.Flags().IntVar(&fleetPushWorldSize, "fleet-world-size", 0,
		"Distributed training world size (0 = not distributed)")
	fleetPushCmd.Flags().IntVar(&fleetPushNodeRank, "fleet-node-rank", 0,
		"Rank within the distributed group, when world_size > 0")
	fleetPushCmd.Flags().BoolVar(&fleetPushStubCollector, "stub", false,
		"Use a synthetic-signal collector (for endpoint smoke testing)")

	_ = fleetPushCmd.MarkFlagRequired("fleet-endpoint")
	_ = fleetPushCmd.MarkFlagRequired("fleet-cluster-id")

	rootCmd.AddCommand(fleetPushCmd)
}

func runFleetPush(cmd *cobra.Command, args []string) error {
	// Reject empty-after-trim values that Cobra's MarkFlagRequired does
	// not catch (the flag is "present" but value is ""):
	if strings.TrimSpace(fleetPushEndpoint) == "" {
		return errors.New("--fleet-endpoint must not be empty")
	}
	if strings.TrimSpace(fleetPushClusterID) == "" {
		return errors.New("--fleet-cluster-id must not be empty")
	}
	if _, ok := validDetectionModes[fleetPushDetectionMode]; !ok {
		return fmt.Errorf("--fleet-detection-mode %q: must be one of none, fleet, fleet-cached, local-cached, local-baseline",
			fleetPushDetectionMode)
	}
	// --fleet-insecure + TLS material is a silent contradiction: the URL
	// becomes http:// and the TLS config is silently ignored. Reject.
	tlsSet := fleetPushTLSCA != "" || fleetPushTLSCert != "" || fleetPushTLSKey != ""
	if fleetPushInsecure && tlsSet {
		return errors.New("--fleet-insecure is incompatible with --fleet-tls-* flags")
	}

	if fleetPushNodeID == "" {
		host, err := os.Hostname()
		if err != nil || host == "" {
			return fmt.Errorf("--fleet-node-id not set and hostname lookup failed: %w", err)
		}
		fleetPushNodeID = host
	}
	// Sanitize node ID so it carries through OTLP / labels predictably.
	fleetPushNodeID = strings.TrimSpace(fleetPushNodeID)
	if !utf8.ValidString(fleetPushNodeID) {
		return fmt.Errorf("--fleet-node-id is not valid UTF-8")
	}

	log := slog.Default()

	baseliner, err := health.NewBaseliner(health.DefaultBaselineConfig(), log)
	if err != nil {
		return fmt.Errorf("baseliner: %w", err)
	}

	// Attempt restore. Missing/stale/corrupt are all non-fatal — just log.
	// A successful restore lets the state machine skip CALIBRATING and
	// start in ACTIVE, avoiding a quorum-dip during rolling-update restart
	// (Story 2.4 AC3 / NFR16).
	ps, status, lerr := health.Load(fleetPushPersistPath, fleetPushStaleAge, time.Now(), log)
	if lerr != nil {
		log.Warn("baseline load failed, starting fresh", "err", lerr.Error(), "status", status.String())
	}
	restored := false
	if status == health.LoadFresh {
		if rerr := baseliner.Restore(ps); rerr != nil {
			log.Warn("baseline restore failed, starting fresh", "err", rerr.Error())
		} else {
			restored = true
			log.Info("baseline restored from disk", "sample_count", ps.SampleCount)
		}
	} else {
		log.Info("baseline not restored", "status", status.String())
	}

	var sm health.StateMachine
	if restored {
		sm, err = health.NewStateMachineFromRestore(health.DefaultStateConfig(), log)
	} else {
		sm, err = health.NewStateMachine(health.DefaultStateConfig(), log)
	}
	if err != nil {
		return fmt.Errorf("state machine: %w", err)
	}

	// Start from library defaults and override only what the CLI exposes.
	// Avoids drift where DefaultEmitterConfig gets a new field and the CLI
	// silently zeros it.
	emCfg := health.DefaultEmitterConfig()
	emCfg.Endpoint = fleetPushEndpoint
	emCfg.ClusterID = fleetPushClusterID
	emCfg.NodeID = fleetPushNodeID
	emCfg.WorkloadType = fleetPushWorkloadType
	emCfg.PushInterval = fleetPushInterval
	emCfg.Timeout = fleetPushTimeout
	emCfg.Insecure = fleetPushInsecure
	emCfg.WorldSize = fleetPushWorldSize
	emCfg.NodeRank = fleetPushNodeRank
	if fleetPushTLSCA != "" || fleetPushTLSCert != "" || fleetPushTLSKey != "" {
		emCfg.TLS = health.TLSConfig{
			CACertPath:     fleetPushTLSCA,
			ClientCertPath: fleetPushTLSCert,
			ClientKeyPath:  fleetPushTLSKey,
		}
	}
	em, err := health.NewEmitter(emCfg, log)
	if err != nil {
		return fmt.Errorf("emitter: %w", err)
	}

	var collector health.Collector
	switch {
	case fleetPushStubCollector:
		collector = &stubCollector{}
	default:
		return errors.New("real signal collector not yet wired — re-run with --stub for smoke testing")
	}

	loop, err := health.NewLoop(health.LoopConfig{
		Baseliner:     baseliner,
		StateMachine:  sm,
		Emitter:       em,
		Collector:     collector,
		ScoreConfig:   health.DefaultConfig(),
		PushInterval:  fleetPushInterval,
		DetectionMode: fleetPushDetectionMode,
		WorkloadType:  fleetPushWorkloadType,
		Log:           log,
	})
	if err != nil {
		return fmt.Errorf("loop: %w", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	log.Info("fleet-push starting",
		"endpoint", fleetPushEndpoint,
		"cluster_id", fleetPushClusterID,
		"node_id", fleetPushNodeID,
		"interval", fleetPushInterval.String(),
		"stub", fleetPushStubCollector,
	)

	runErr := loop.Run(ctx)

	// Persistence on graceful shutdown. Only write if the baseline is
	// warm (>= WarmupSamples) — a cold snapshot would be restored by the
	// next boot and fool the state machine into starting in ACTIVE with
	// garbage baselines.
	baselineWarmupMin := health.DefaultBaselineConfig().WarmupSamples
	if baseliner.SampleCount() < baselineWarmupMin {
		log.Info("baseline not saved: below warmup threshold",
			"sample_count", baseliner.SampleCount(), "warmup_min", baselineWarmupMin)
	} else if serr := health.Save(fleetPushPersistPath, baseliner.Snapshot(), time.Now()); serr != nil {
		log.Warn("baseline save failed", "err", serr.Error())
		// Bubble up the save failure so operators see a non-zero exit
		// rather than having to grep the logs.
		if errors.Is(runErr, context.Canceled) || errors.Is(runErr, context.DeadlineExceeded) {
			return fmt.Errorf("shutdown: save baseline: %w", serr)
		}
		// If Run also returned a non-graceful error, prefer that.
	} else {
		log.Info("baseline saved", "path", fleetPushPersistPath, "sample_count", baseliner.SampleCount())
	}

	// Graceful = ctx cancelled OR deliberate deadline expired.
	if errors.Is(runErr, context.Canceled) || errors.Is(runErr, context.DeadlineExceeded) {
		return nil
	}
	return runErr
}

// stubCollector returns constant synthetic signals. Only for smoke-testing
// the wire-format against a real Fleet endpoint — never ship a user on this.
type stubCollector struct{}

func (s *stubCollector) Collect(ctx context.Context, now time.Time) (health.RawObservation, int, error) {
	return health.RawObservation{
		Throughput: 100,
		Compute:    0.9,
		Memory:     0.9,
		CPU:        0.9,
	}, 10, nil
}
