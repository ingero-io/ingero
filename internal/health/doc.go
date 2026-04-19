// Package health implements the agent-side health score used by Ingero Fleet
// to detect stragglers across a GPU cluster.
//
// A single score in [0,1] is computed each push interval from four signals:
// CUDA throughput ratio, compute efficiency, memory headroom, and CPU
// availability. Fleet accumulates scores from peers and computes a
// peer-relative straggler threshold via MAD; each agent self-classifies by
// comparing its own score against that threshold.
//
// The package is split into:
//
//   - score.go     — 4-signal formula with smooth floor penalty.
//   - baseline.go  — bias-corrected EMA + hard-floor baselines per signal.
//   - state.go     — CALIBRATING/ACTIVE/IDLE/STALE state machine.
//   - persist.go   — atomic baseline persistence across restarts.
//   - emitter.go   — OTLP Gauge emission to Fleet (or any OTEL Collector).
//
// All metric names, attribute keys and header names emitted from this package
// live in pkg/contract — that is the single source of truth shared with
// Fleet. No string literals in this package should duplicate a contract
// constant.
package health
