package store

import (
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// RolloverConfig drives file-level DB rollover. Set via
// (*Store).SetRolloverConfig before Run() to enable. Zero MaxSize
// disables rollover; the Store keeps the existing pruneBySize behavior
// when --max-db is set instead.
//
// Rollover swaps the live DB file for a fresh one when the live file
// crosses MaxSize bytes. The displaced file is renamed to a sibling
// `<basename>.<UTC-iso8601>.db` and retained until KeepFiles older
// rolled files exist; the oldest is then deleted to keep the disk
// footprint bounded.
//
// Mutually exclusive with --max-db (in-place row pruning). The CLI
// layer enforces that contract; the Store does not check.
type RolloverConfig struct {
	// MaxSize is the size (DB+WAL+SHM, in bytes) above which the next
	// pruneTicker tick or post-flush hook triggers a rollover. 0
	// disables rollover entirely.
	MaxSize int64

	// KeepFiles is the number of rolled-over files retained on disk.
	// Older files beyond this count are deleted oldest-first by
	// sweepOldRollovers. Default applied (= 6) when zero.
	KeepFiles int
}

// SetRolloverConfig installs (or replaces) the rollover policy. Calls
// from before Run() are picked up automatically by the maybeRollover
// hook; calls during Run() take effect on the next tick. Pass a
// zero-value cfg to disable.
func (s *Store) SetRolloverConfig(cfg RolloverConfig) {
	if cfg.KeepFiles <= 0 {
		cfg.KeepFiles = 6
	}
	s.rolloverCfg.Store(&cfg)
}

// MaybeRollover checks the active rollover config and, if the live DB
// has crossed MaxSize, performs a rollover. Called from the Run() loop
// after flushBatch and on each pruneTicker.C tick. No-op when no
// rollover config is set.
//
// Returns the rollover error (if any) and never panics. The caller
// (the Run goroutine) does not surface the error externally — it
// logs and continues so the daemon survives transient failure modes
// (cross-mount rename, full disk during checkpoint).
func (s *Store) MaybeRollover() error {
	cfg := s.rolloverCfg.Load()
	if cfg == nil || cfg.MaxSize <= 0 {
		return nil
	}
	if s.diskUsage() < cfg.MaxSize {
		return nil
	}
	return s.RolloverNow("size", cfg)
}

// RolloverNow forces a rollover regardless of the current size.
// Sequence:
//
//  1. Checkpoint WAL into the main file (TRUNCATE — must produce a
//     fully self-contained file, not just a quiesced one).
//  2. Close the live *sql.DB.
//  3. Rename the on-disk file to `<base>.<UTC-iso8601>.db`.
//  4. Open a fresh *sql.DB on the original path.
//  5. Re-apply schema migrations (idempotent).
//  6. Sweep old rollover files beyond KeepFiles.
//
// The rolloverMu RWMutex is held write-locked across (2)-(4) so that
// concurrent in-process readers (Stats, Query, etc.) either wait or
// see a coherent *sql.DB pointer; they never observe a closed handle
// being used.
func (s *Store) RolloverNow(reason string, cfg *RolloverConfig) error {
	if cfg == nil {
		c := s.rolloverCfg.Load()
		if c == nil {
			return fmt.Errorf("rollover: no config set")
		}
		cfg = c
	}
	// In-memory DBs cannot be rolled over (no on-disk file to rename).
	// The trace-time path never sets rollover on :memory:; defense-in-depth.
	if s.dbPath == ":memory:" || strings.HasPrefix(s.dbPath, "file::memory:") {
		return fmt.Errorf("rollover: in-memory DBs not supported")
	}
	if !s.rolloverInFlight.CompareAndSwap(false, true) {
		// Already rolling over from another caller; skip this one.
		return nil
	}
	defer s.rolloverInFlight.Store(false)

	rolledPath := buildRolledPath(s.dbPath, time.Now().UTC())
	s.rolloverLog().Info("rollover starting",
		"reason", reason,
		"current_path", s.dbPath,
		"rolled_path", rolledPath,
		"size_bytes", s.diskUsage(),
		"max_size_bytes", cfg.MaxSize,
	)

	// (1) Checkpoint. TRUNCATE drains the WAL into the main file AND
	// truncates the WAL to zero, so the rename target is a single
	// self-contained file with no companion -wal/-shm needed.
	if _, err := s.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)"); err != nil {
		s.rolloverLog().Error("rollover: checkpoint failed; aborting",
			"err", err.Error())
		s.rolloverFailures.Add(1)
		return fmt.Errorf("rollover: checkpoint: %w", err)
	}

	// Acquire the swap lock. Concurrent readers that take rolloverMu.RLock
	// (currently none in the Store body, but any future reader must) will
	// block here.
	s.rolloverMu.Lock()
	defer s.rolloverMu.Unlock()

	// (2) Close the live DB.
	oldDB := s.db
	if err := oldDB.Close(); err != nil {
		s.rolloverFailures.Add(1)
		return fmt.Errorf("rollover: close: %w", err)
	}

	// Best-effort: remove leftover -wal and -shm. The TRUNCATE checkpoint
	// should have collapsed them; if any remain (rare), we don't want
	// the renamed file to leave dangling siblings.
	for _, suffix := range []string{"-wal", "-shm"} {
		_ = os.Remove(s.dbPath + suffix)
	}

	// (3) Rename. POSIX atomic; failure leaves the source file at the
	// original path so a subsequent reopen can recover.
	if err := os.Rename(s.dbPath, rolledPath); err != nil {
		// Reopen the original path so the daemon survives. The size
		// limit will be tripped again on the next tick and we'll retry.
		newDB, openErr := sql.Open("sqlite", s.dbPath)
		if openErr == nil {
			s.db = newDB
			s.rolloverFailures.Add(1)
			s.rolloverLog().Error("rollover: rename failed; reopened original path",
				"src", s.dbPath, "dst", rolledPath, "err", err.Error())
			return fmt.Errorf("rollover: rename %s -> %s: %w", s.dbPath, rolledPath, err)
		}
		// Catastrophic: rename failed AND reopen failed. The live handle
		// is closed (line above) and we cannot resurrect a working DB.
		// Leave s.db pointing at the closed handle so subsequent writes
		// fail loudly with sql.ErrConnDone (caller logs via the same
		// flushFailures path). Operators must restart the agent; emit
		// a high-severity log so the operator log surfaces this.
		s.db = oldDB
		s.rolloverFailures.Add(1)
		s.rolloverLog().Error("rollover: rename failed AND reopen failed; DB handle is closed; RESTART REQUIRED",
			"src", s.dbPath, "dst", rolledPath,
			"rename_err", err.Error(),
			"reopen_err", openErr.Error())
		return fmt.Errorf("rollover: rename %s -> %s failed: %w; reopen also failed: %v; RESTART REQUIRED",
			s.dbPath, rolledPath, err, openErr)
	}

	// (4) Open fresh DB at the original path. Use buildWriteDSN so the
	// rolled-over pool gets busy_timeout on every connection, exactly as
	// New() does — a post-open Exec would only set it on one connection.
	newDB, err := sql.Open("sqlite", buildWriteDSN(s.dbPath))
	if err != nil {
		// The live DB is closed (oldDB.Close at line 131) and the on-disk
		// file has already been renamed away. We cannot recover here; the
		// agent must be restarted. Surface the broken state so subsequent
		// flushes fail loudly via the same flushFailures path rather than
		// using a closed handle.
		s.db = oldDB
		s.rolloverFailures.Add(1)
		s.rolloverLog().Error("rollover: reopen failed; DB handle is closed; RESTART REQUIRED",
			"path", s.dbPath, "err", err.Error())
		return fmt.Errorf("rollover: reopen failed: %w; RESTART REQUIRED", err)
	}
	// Replicate the PRAGMAs from New() so the rolled-over DB has the
	// same operational characteristics. Errors here are non-fatal —
	// the DB is functional without them, just slower or less safe.
	// busy_timeout is carried by buildWriteDSN on every connection.
	newDB.Exec("PRAGMA auto_vacuum = INCREMENTAL")
	if _, err := newDB.Exec("PRAGMA journal_mode=WAL"); err != nil {
		s.rolloverLog().Warn("rollover: WAL mode not set on fresh DB", "err", err.Error())
	}

	// Publish the new handle BEFORE schema-apply so a schema failure
	// leaves a functional (if partially migrated) handle attached to
	// the Store. New writes will land on it, and the next agent
	// restart re-runs migrations idempotently. Holding the closed
	// oldDB during applyAgentSchema would silently break every
	// subsequent write with sql.ErrConnDone.
	s.db = newDB

	// (5) Re-apply the agent's schema sequence on the fresh DB.
	// Delegates to applyAgentSchema which is the same helper New()
	// uses, so the rolled-over DB cannot drift from a freshly created
	// one.
	if err := rolloverApplySchema(newDB); err != nil {
		// Schema-fail is recoverable: the new DB is open and writable.
		// applyAgentSchema is idempotent so the next agent restart
		// re-runs and reaches a coherent state. Surface as Error so
		// the operator log shows the partially-migrated state.
		s.rolloverFailures.Add(1)
		s.rolloverLog().Error("rollover: schema apply failed; live DB attached but partially migrated; RESTART RECOMMENDED",
			"path", s.dbPath, "err", err.Error())
		return fmt.Errorf("rollover: schema: %w; RESTART RECOMMENDED", err)
	}

	// The read-only sibling pool's connections still point at the
	// rolled-away inode (POSIX file handles survive rename). Close it
	// and reopen against the fresh file so ExecuteReadOnly sees the
	// post-rollover schema + (empty) data.
	if s.roDB != nil {
		s.roDB.Close()
		s.roDB = nil
	}
	s.openReadOnlyDB()

	s.rolloverCount.Add(1)
	s.rolloverLog().Info("rollover complete",
		"rolled_path", rolledPath,
		"keep_files", cfg.KeepFiles,
	)

	// (6) Sweep. Errors here are warnings, not fatal — the rollover
	// itself is already done.
	if err := sweepOldRollovers(s.dbPath, cfg.KeepFiles, s.rolloverLog()); err != nil {
		s.rolloverLog().Warn("rollover: sweep failed", "err", err.Error())
	}
	return nil
}

// rolloverApplySchema runs the agent's full schema-creation +
// migration sequence on a freshly-opened DB. Delegates to
// applyAgentSchema (in schema_setup.go) so the rollover path is
// guaranteed-identical to New()'s initialization sequence.
//
// onDiskVersion is 0 for a brand-new file; the user_version ratchet
// runs at the end of applyAgentSchema and stamps the file with the
// running binary's CurrentUserVersion.
func rolloverApplySchema(db *sql.DB) error {
	return applyAgentSchema(db, 0)
}

// buildRolledPath produces the rolled filename:
// `<dir>/<basename-stem>.<UTC-iso8601-ns>-<rand4>.db`. The original
// extension (.db) is preserved on the rolled file so operators
// recognize the extension; the timestamp + random suffix is inserted
// before it.
//
// Format: `20060102T150405.000000000Z-<8 hex chars>` — nanosecond
// resolution covers same-second rollovers; the random suffix covers
// NTP backwards drift (clock can step back below the previous nano)
// and any future single-millisecond re-roll. 4 random bytes (8 hex
// chars) gives 2^32 possible suffixes per timestamp, so two rollovers
// landing on the same nanosecond would still collide only at a rate
// of ~1 in 4 billion.
func buildRolledPath(currentPath string, ts time.Time) string {
	dir := filepath.Dir(currentPath)
	base := filepath.Base(currentPath)
	ext := filepath.Ext(base)
	stem := strings.TrimSuffix(base, ext)
	stamp := ts.Format("20060102T150405.000000000Z")
	var rnd [4]byte
	if _, err := rand.Read(rnd[:]); err != nil {
		// rand.Read on Linux essentially always succeeds. If it does
		// fail, fall back to a fixed sentinel; the nanosecond stamp is
		// already enough to differentiate rapid rollovers, so this
		// degraded path is still collision-resistant in practice.
		rnd = [4]byte{0xff, 0xff, 0xff, 0xff}
	}
	return filepath.Join(dir, fmt.Sprintf("%s.%s-%s%s", stem, stamp, hex.EncodeToString(rnd[:]), ext))
}

// sweepOldRollovers finds rolled siblings of livePath, sorts by
// embedded timestamp (filename order is sufficient because the format
// is lexicographically time-sortable), and deletes oldest-first
// until at most keep files remain.
func sweepOldRollovers(livePath string, keep int, log *slog.Logger) error {
	dir := filepath.Dir(livePath)
	base := filepath.Base(livePath)
	ext := filepath.Ext(base)
	stem := strings.TrimSuffix(base, ext)
	// Match: <stem>.<YYYYMMDDTHHMMSSZ><ext>
	pattern := filepath.Join(dir, stem+".*"+ext)
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("glob %q: %w", pattern, err)
	}
	// Filter out the live file itself (Glob can pick it up if base
	// happens to also satisfy the wildcard — defensive).
	var rolled []string
	for _, m := range matches {
		if m == livePath {
			continue
		}
		// Reject anything that doesn't look like a timestamped roll
		// file (e.g. "ingero.db" with stem="ingero" and a sibling
		// "ingero.notes.db" — wildcard match but not a roll file).
		mid := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(m), stem+"."), ext)
		if !looksLikeRolloverTimestamp(mid) {
			continue
		}
		rolled = append(rolled, m)
	}
	if len(rolled) <= keep {
		return nil
	}
	// Filenames are lexicographically sortable by embedded UTC ISO
	// timestamp; oldest-first is the natural order.
	sort.Strings(rolled)
	for _, m := range rolled[:len(rolled)-keep] {
		if err := os.Remove(m); err != nil {
			log.Warn("rollover sweep: remove failed", "path", m, "err", err.Error())
			continue
		}
		log.Info("rollover sweep: removed", "path", m)
	}
	return nil
}

// looksLikeRolloverTimestamp returns true when s matches a rollover
// timestamp suffix. Two forms are accepted:
//
//   - Pre-collision-fix:  YYYYMMDDTHHMMSSZ              (16 chars,  e.g. 20260510T191425Z)
//   - Current:            YYYYMMDDTHHMMSS.NNNNNNNNNZ-RR (35 chars,  e.g. 20260510T191425.123456789Z-deadbeef)
//
// The pre-collision-fix form stays accepted so sweep + ListRolledFiles
// continue to recognize legacy rolled files that operators may still
// have on disk from earlier agent versions. We don't pull in time.Parse
// because the volume of files in the dir at glob time is tiny and a
// regex would need its own import; a small character-class check is
// adequate.
func looksLikeRolloverTimestamp(s string) bool {
	switch len(s) {
	case 16:
		// Pre-fix: YYYYMMDDTHHMMSSZ.
		// 8 digits, 'T', 6 digits, 'Z'.
		for i := 0; i < 8; i++ {
			if s[i] < '0' || s[i] > '9' {
				return false
			}
		}
		if s[8] != 'T' {
			return false
		}
		for i := 9; i < 15; i++ {
			if s[i] < '0' || s[i] > '9' {
				return false
			}
		}
		return s[15] == 'Z'

	case 35:
		// Current: YYYYMMDDTHHMMSS.NNNNNNNNNZ-XXXXXXXX.
		// 8 digits, 'T', 6 digits, '.', 9 digits, 'Z', '-', 8 hex.
		for i := 0; i < 8; i++ {
			if s[i] < '0' || s[i] > '9' {
				return false
			}
		}
		if s[8] != 'T' {
			return false
		}
		for i := 9; i < 15; i++ {
			if s[i] < '0' || s[i] > '9' {
				return false
			}
		}
		if s[15] != '.' {
			return false
		}
		for i := 16; i < 25; i++ {
			if s[i] < '0' || s[i] > '9' {
				return false
			}
		}
		if s[25] != 'Z' || s[26] != '-' {
			return false
		}
		for i := 27; i < 35; i++ {
			c := s[i]
			isDigit := c >= '0' && c <= '9'
			isLowerHex := c >= 'a' && c <= 'f'
			if !isDigit && !isLowerHex {
				return false
			}
		}
		return true

	default:
		return false
	}
}

// rolloverLog returns the logger used for rollover operations. Today
// it's just slog.Default(); separated to a method so a future
// SetRolloverLogger can plumb in a per-Store logger without touching
// every call site.
func (s *Store) rolloverLog() *slog.Logger {
	return slog.Default()
}

// RolloverStats reports cumulative rollover counters. Values are
// monotonic across the lifetime of the Store and never decrease.
// Callers (the OTLP exporter, tests) read these via Stats().
type RolloverStats struct {
	Count    uint64
	Failures uint64
}

// RolloverStats returns a snapshot of cumulative rollover telemetry.
func (s *Store) RolloverStats() RolloverStats {
	return RolloverStats{
		Count:    s.rolloverCount.Load(),
		Failures: s.rolloverFailures.Load(),
	}
}

// ListRolledFiles returns the absolute paths of every rolled-over
// sibling DB next to the live file, oldest-first. The live file
// itself is excluded; only files matching the rollover timestamp
// pattern (`<stem>.<YYYYMMDDTHHMMSSZ><ext>`) are included so a
// hand-placed file like `<stem>.notes<ext>` doesn't accidentally
// show up in query --include-rolled.
//
// Returns an empty slice when no rolled siblings exist or the dir
// is unreadable. Standalone helper (not Glob'd from the rollover
// sweeper because that one prunes too); callers consume the list
// for read-only purposes like cross-rollover query.
func ListRolledFiles(livePath string) ([]string, error) {
	dir := filepath.Dir(livePath)
	base := filepath.Base(livePath)
	ext := filepath.Ext(base)
	stem := strings.TrimSuffix(base, ext)
	pattern := filepath.Join(dir, stem+".*"+ext)
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("glob %q: %w", pattern, err)
	}
	out := make([]string, 0, len(matches))
	for _, m := range matches {
		if m == livePath {
			continue
		}
		mid := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(m), stem+"."), ext)
		if !looksLikeRolloverTimestamp(mid) {
			continue
		}
		out = append(out, m)
	}
	// Filename is lexicographically time-sortable thanks to the
	// fixed-width compact-ISO timestamp, so a string sort is also a
	// chronological sort. Oldest first.
	sort.Strings(out)
	return out, nil
}
