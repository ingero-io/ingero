package cli

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"

	"github.com/ingero-io/ingero/internal/health"
	"github.com/ingero-io/ingero/internal/remediate"
	"github.com/ingero-io/ingero/internal/store"
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
	fleetPushThresholdURL  string
	fleetPushPollInterval  time.Duration
	fleetPushHysteresis    float64
	fleetPushRemediate     bool
	fleetPushRemediateSock string
	fleetPushRemediateGid  int
	fleetPushSignalDBPath  string
	fleetPushSignalWindow  time.Duration
	fleetPushSignalNumGPUs int
	fleetPushWarmupSamples int
)

var fleetPushCmd = &cobra.Command{
	Use:   "fleet-push",
	Short: "Push health scores to Fleet (experimental, Epic 2 integration)",
	Long: `Push per-node health scores to an Ingero Fleet instance (or any OTLP/HTTP
collector) for peer-relative straggler detection.

This is the Epic 2 integration entry point. Without --fleet-endpoint the
agent behaves exactly as before — nothing about the default 'trace',
'explain', or 'query' workflows is changed.

Signal sources:
  * --stub: synthetic constant signals for endpoint smoke-testing.
  * Default (no --stub): reads rolling aggregates from the SQLite DB populated
    by a sibling 'ingero trace --record' process. Point --signal-db-path at
    the same DB both sides share (Helm chart defaults to a hostPath volume).

Warmup: the agent observes --warmup-samples × --fleet-push-interval before
it emits non-calibrating scores. With defaults that is 30 × 5s = 150 s.
Tune --warmup-samples lower for dev smoke tests or higher for noisy workloads.`,
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
	fleetPushCmd.Flags().StringVar(&fleetPushThresholdURL, "fleet-threshold-url", "",
		"Fleet threshold API URL for GET fallback polling (Story 3.2). Empty disables polling; piggyback headers are still consumed.")
	fleetPushCmd.Flags().DurationVar(&fleetPushPollInterval, "fleet-poll-interval", 10*time.Second,
		"GET fallback polling cadence (jittered +/-20%%)")
	fleetPushCmd.Flags().Float64Var(&fleetPushHysteresis, "fleet-classifier-hysteresis", 0.02,
		"Straggler classification hysteresis band (Story 3.4)")
	fleetPushCmd.Flags().BoolVar(&fleetPushRemediate, "remediate", false,
		"Publish straggler state to the remediation UDS socket (Story 3.4)")
	fleetPushCmd.Flags().StringVar(&fleetPushRemediateSock, "remediate-socket", "/tmp/ingero-remediate.sock",
		"UDS path for remediation messages when --remediate is set")
	fleetPushCmd.Flags().IntVar(&fleetPushRemediateGid, "remediate-gid", 65532,
		"Numeric GID granted group access to the remediation socket (chown -1:gid + chmod 0770). Default 65532 matches distroless 'nonroot'. Set < 0 to keep the socket owner-only (0700).")
	fleetPushCmd.Flags().StringVar(&fleetPushSignalDBPath, "signal-db-path", store.DefaultDBPath(),
		"Path to the SQLite DB written by a sibling 'ingero trace --record' process (ignored when --stub)")
	fleetPushCmd.Flags().DurationVar(&fleetPushSignalWindow, "signal-window", 60*time.Second,
		"Rolling window for throughput/compute derivation. event_aggregates has minute buckets, so shorter windows still read full-minute data.")
	fleetPushCmd.Flags().IntVar(&fleetPushSignalNumGPUs, "signal-num-gpus", 0,
		"GPU count for normalizing compute signal. 0 = autodetect (defaults to 1).")
	fleetPushCmd.Flags().IntVar(&fleetPushWarmupSamples, "warmup-samples", health.DefaultBaselineConfig().WarmupSamples,
		"Samples observed before first non-calibrating push (warmup = samples × push-interval). Default 30 × 5s = 150s.")

	_ = fleetPushCmd.MarkFlagRequired("fleet-endpoint")
	_ = fleetPushCmd.MarkFlagRequired("fleet-cluster-id")

	rootCmd.AddCommand(fleetPushCmd)
}

// remediateSink adapts a *remediate.Server to the health.StragglerSink
// interface. Lives in the CLI package so that internal/health stays free
// of a dependency on internal/remediate.
type remediateSink struct {
	server *remediate.Server
}

func (r *remediateSink) SendStragglerState(ev health.StragglerEvent) error {
	return r.server.SendFleetStragglerState(
		ev.Timestamp,
		ev.NodeID,
		ev.ClusterID,
		string(ev.DetectionMode),
		ev.DominantSignal,
		ev.Score,
		ev.Threshold,
	)
}

func (r *remediateSink) SendStragglerResolved(nodeID, clusterID string, ts time.Time) error {
	return r.server.SendFleetStragglerResolved(ts, nodeID, clusterID)
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

	baseCfg := health.DefaultBaselineConfig()
	if fleetPushWarmupSamples > 0 {
		baseCfg.WarmupSamples = fleetPushWarmupSamples
	}
	baseliner, err := health.NewBaseliner(baseCfg, log)
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

	stCfg := health.DefaultStateConfig()
	if fleetPushWarmupSamples > 0 {
		stCfg.WarmupSamples = fleetPushWarmupSamples
	}
	var sm health.StateMachine
	if restored {
		sm, err = health.NewStateMachineFromRestore(stCfg, log)
	} else {
		sm, err = health.NewStateMachine(stCfg, log)
	}
	if err != nil {
		return fmt.Errorf("state machine: %w", err)
	}

	// ThresholdCache is populated by the Emitter on every push response
	// (piggyback headers) and optionally by the GET-endpoint Poller.
	thresholdCache := health.NewThresholdCache()

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
	emCfg.ThresholdCache = thresholdCache
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
		collector, err = health.NewSQLiteCollector(health.SQLiteCollectorConfig{
			DBPath:  fleetPushSignalDBPath,
			Window:  fleetPushSignalWindow,
			NumGPUs: fleetPushSignalNumGPUs,
			Log:     log,
		})
		if err != nil {
			return fmt.Errorf("signal collector: %w\n(hint: run 'ingero trace --record' in a sibling pod/process to populate this DB, or pass --stub for smoke testing)", err)
		}
		if closer, ok := collector.(interface{ Close() error }); ok {
			defer closer.Close()
		}
	}

	// Story 3.3: mode evaluator replaces the static --fleet-detection-mode
	// label. Consumes the ThresholdCache + emitter-reachability + baseliner
	// warmup count. Threshold flows through to Story 3.4's classifier.
	modeEvaluator, err := health.NewModeEvaluator(
		health.DefaultModeConfig(),
		thresholdCache,
		em,
		baseliner,
		baseCfg.WarmupSamples,
		log,
	)
	if err != nil {
		return fmt.Errorf("mode evaluator: %w", err)
	}

	// Story 3.4: classifier + optional UDS sink.
	classifier, err := health.NewClassifier(health.ClassifierConfig{Hysteresis: fleetPushHysteresis})
	if err != nil {
		return fmt.Errorf("classifier: %w", err)
	}

	// Optional UDS: only started when --remediate is set. Shutdown cleanup
	// happens via the deferred Close below.
	var (
		udsServer *remediate.Server
		sink      health.StragglerSink
	)
	if fleetPushRemediate {
		udsServer = remediate.NewServer(fleetPushRemediateSock)
		udsServer.SetSocketGid(fleetPushRemediateGid)
		if err := udsServer.Start(); err != nil {
			return fmt.Errorf("remediate server: %w", err)
		}
		defer udsServer.Close()
		sink = &remediateSink{server: udsServer}
		log.Info("remediate UDS server started", "socket", fleetPushRemediateSock)
	}

	// Optional GET-endpoint poller (Story 3.2). Only started when the
	// operator configures a threshold URL distinct from the OTLP endpoint.
	var pollerWg sync.WaitGroup
	if strings.TrimSpace(fleetPushThresholdURL) != "" {
		pollerCfg := health.PollerConfig{
			BaseURL:   fleetPushThresholdURL,
			ClusterID: fleetPushClusterID,
			Interval:  fleetPushPollInterval,
			Timeout:   fleetPushTimeout,
			Insecure:  fleetPushInsecure,
		}
		poller, perr := health.NewPoller(pollerCfg, thresholdCache, log)
		if perr != nil {
			return fmt.Errorf("poller: %w", perr)
		}
		// Start the poller after the loop ctx is available (see below).
		defer func() { pollerWg.Wait() }()
		log.Info("threshold poller configured", "url", fleetPushThresholdURL, "interval", fleetPushPollInterval.String())
		// Deferred start (ctx is created below); capture via closure.
		defer func(p *health.Poller) {}(poller)
		_ = poller // actual start happens after ctx
	}

	loop, err := health.NewLoop(health.LoopConfig{
		Baseliner:     baseliner,
		StateMachine:  sm,
		Emitter:       em,
		Collector:     collector,
		ModeEvaluator: modeEvaluator,
		Classifier:    classifier,
		StragglerSink: sink,
		NodeID:        fleetPushNodeID,
		ClusterID:     fleetPushClusterID,
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

	// Now that ctx exists, start the poller (if configured) on a goroutine.
	if strings.TrimSpace(fleetPushThresholdURL) != "" {
		pollerCfg := health.PollerConfig{
			BaseURL:   fleetPushThresholdURL,
			ClusterID: fleetPushClusterID,
			Interval:  fleetPushPollInterval,
			Timeout:   fleetPushTimeout,
			Insecure:  fleetPushInsecure,
		}
		// Reuse the same mTLS material as the emitter so the GET poller
		// survives the same handshake that the push path does. Without
		// this, HTTPS threshold URLs fail with x509 errors despite
		// --fleet-tls-* flags being set.
		if fleetPushTLSCA != "" || fleetPushTLSCert != "" || fleetPushTLSKey != "" {
			tlsCfg, terr := health.LoadTLSConfig(health.TLSConfig{
				CACertPath:     fleetPushTLSCA,
				ClientCertPath: fleetPushTLSCert,
				ClientKeyPath:  fleetPushTLSKey,
			})
			if terr != nil {
				return fmt.Errorf("poller tls: %w", terr)
			}
			pollerCfg.TLSConfig = tlsCfg
		}
		poller, _ := health.NewPoller(pollerCfg, thresholdCache, log)
		pollerWg.Add(1)
		go func() {
			defer pollerWg.Done()
			if perr := poller.Run(ctx); perr != nil && !errors.Is(perr, context.Canceled) {
				log.Debug("threshold poller exited", "err", perr.Error())
			}
		}()
	}

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
	baselineWarmupMin := baseCfg.WarmupSamples
	if baseliner.SampleCount() < baselineWarmupMin {
		log.Info("baseline not saved: below warmup threshold",
			"sample_count", baseliner.SampleCount(), "warmup_min", baselineWarmupMin)
	} else if serr := health.Save(fleetPushPersistPath, baseliner.Snapshot(), time.Now()); serr != nil {
		// Persistence is best-effort. Do not mask a clean shutdown with a
		// non-zero exit when the only failure is that /var/lib/ingero is
		// not writable for this user. WARN is sufficient for operators; a
		// non-graceful runErr (below) still propagates.
		log.Warn("baseline save failed (best-effort; not propagated as exit error)", "err", serr.Error(), "path", fleetPushPersistPath)
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
