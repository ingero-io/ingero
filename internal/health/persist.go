package health

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// DefaultPersistencePath is the on-disk location used when the operator
// does not override via config. Linux-flavored by convention — Windows /
// BSD runners must override via config.
const DefaultPersistencePath = "/var/lib/ingero/health-baseline.json"

// DefaultStaleAge matches the story: a baseline file older than 10 minutes
// is assumed to describe a workload that has since changed.
const DefaultStaleAge = 10 * time.Minute

// maxBaselineFileSize bounds os.ReadFile input so that a pathological or
// malicious 10 GB baseline file cannot OOM the agent at startup. The real
// baseline file is a few hundred bytes; 1 MB is generous.
const maxBaselineFileSize = 1 << 20

// maxQuarantineSuffix bounds the counter used to avoid filename
// collisions between quarantined files when two corruptions occur within
// the same nanosecond (extremely unlikely but defensively handled).
const maxQuarantineSuffix = 1000

// LoadStatus enumerates the outcomes of Load. Callers switch on this to
// decide whether to Restore(), log, move aside, or fall through to
// CALIBRATING.
type LoadStatus int

const (
	// LoadMissing: no file exists at the path. Treated as first-boot.
	LoadMissing LoadStatus = iota
	// LoadFresh: file parsed, schema OK, content finite, within maxAge.
	// Caller should Restore.
	LoadFresh
	// LoadStale: file parsed, schema OK, but older than maxAge. Discarded.
	LoadStale
	// LoadCorrupt: file unreadable, schema invalid, or content rejected
	// (NaN, Inf, negative baselines, out-of-range values). The file is
	// moved aside as `{path}.corrupt-{stamp}` when possible.
	LoadCorrupt
	// LoadUnreadable: a transient filesystem error (EACCES, EIO, permission
	// denied). The file was NOT quarantined — caller should treat as
	// "try again later" rather than "corruption."
	LoadUnreadable
	// LoadNewerVersion: the file declares a schema_version higher than this
	// binary understands. The file is preserved unchanged (NOT quarantined)
	// so a newer agent can still use it after a rollback. Caller should
	// start CALIBRATING and leave the file alone. Supports rolling upgrades
	// where N-1 and N agents share a persist path.
	LoadNewerVersion
)

// CurrentSchemaVersion is the on-disk schema the current binary reads
// and writes. Bumped whenever savedFile gains or renames a field.
const CurrentSchemaVersion = 1

func (s LoadStatus) String() string {
	switch s {
	case LoadMissing:
		return "missing"
	case LoadFresh:
		return "fresh"
	case LoadStale:
		return "stale"
	case LoadCorrupt:
		return "corrupt"
	case LoadUnreadable:
		return "unreadable"
	case LoadNewerVersion:
		return "newer_version"
	default:
		return "unknown"
	}
}

// savedFile is the on-disk shape. PersistedState lives in-memory and does
// not carry a timestamp; savedFile wraps it with SavedAt so age can be
// evaluated without trusting mtime. FastAlpha/FloorAlpha travel with the
// raw EMA so a post-restore alpha mismatch can be rejected (see
// Baseliner.Restore).
type savedFile struct {
	SchemaVersion int       `json:"schema_version"`
	SavedAt       time.Time `json:"saved_at"`
	SampleCount   int       `json:"sample_count"`
	FastAlpha     float64   `json:"fast_alpha"`
	FloorAlpha    float64   `json:"floor_alpha"`
	FastEMA       Baselines `json:"fast_ema"`
	HardFloor     Baselines `json:"hard_floor"`
}

// Save atomically writes s to path using write-tmp + fsync + rename +
// parent-dir-fsync. Parent directories are created with mode 0755 if
// missing; the final file is written with mode 0644.
//
// The tmp filename is unique per Save call (os.CreateTemp) so concurrent
// writers, or a user-created `{path}.tmp`, cannot collide with or be
// clobbered by this code.
//
// If any step fails, the tmp file is removed and path is left untouched.
func Save(path string, s PersistedState, now time.Time) error {
	if path == "" {
		return errors.New("health.Save: empty path")
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("health.Save: mkdir parent: %w", err)
	}

	// s.SchemaVersion drives the on-disk schema. Current Baseliner.Snapshot
	// always emits 1; future Baseliners may emit higher. Respect it.
	schemaVersion := s.SchemaVersion
	if schemaVersion == 0 {
		schemaVersion = 1
	}
	payload := savedFile{
		SchemaVersion: schemaVersion,
		SavedAt:       now.UTC(),
		SampleCount:   s.SampleCount,
		FastAlpha:     s.FastAlpha,
		FloorAlpha:    s.FloorAlpha,
		FastEMA:       s.FastEMA,
		HardFloor:     s.HardFloor,
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("health.Save: marshal: %w", err)
	}

	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("health.Save: create tmp: %w", err)
	}
	tmpPath := tmp.Name()
	cleanup := func() { _ = os.Remove(tmpPath) }

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		cleanup()
		return fmt.Errorf("health.Save: write: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		cleanup()
		return fmt.Errorf("health.Save: fsync: %w", err)
	}
	if err := tmp.Close(); err != nil {
		cleanup()
		return fmt.Errorf("health.Save: close: %w", err)
	}
	// os.CreateTemp defaults to 0600. Relax to 0644 to match the documented
	// contract (and to match a world-readable daemon state file).
	if err := os.Chmod(tmpPath, 0o644); err != nil {
		cleanup()
		return fmt.Errorf("health.Save: chmod: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		cleanup()
		return fmt.Errorf("health.Save: rename: %w", err)
	}
	// fsync the parent directory so the rename is durable across crashes
	// (POSIX only — Windows provides these guarantees implicitly).
	if runtime.GOOS != "windows" {
		if err := syncDir(dir); err != nil {
			// Not fatal — the rename itself succeeded; the directory
			// fsync is belt-and-suspenders for power-loss durability.
			return fmt.Errorf("health.Save: sync dir: %w", err)
		}
	}
	return nil
}

// syncDir opens dir and calls Sync on the descriptor. Used to durably
// record a rename operation's effect.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}

// Load attempts to read path and return a PersistedState ready for
// Baseliner.Restore. The returned LoadStatus describes the outcome:
//
//   - LoadFresh      — caller should Restore() the PersistedState.
//   - LoadMissing    — no file; start CALIBRATING with a fresh Baseliner.
//   - LoadStale      — discard, start CALIBRATING. The stale file is removed.
//   - LoadCorrupt    — moved aside as {path}.corrupt-{stamp}. Start
//     CALIBRATING.
//   - LoadUnreadable — transient filesystem error; file was NOT moved.
//     Caller should retry or fall through to CALIBRATING.
//
// A non-nil error describes the underlying cause when LoadStatus is
// LoadCorrupt or LoadUnreadable.
func Load(path string, maxAge time.Duration, now time.Time, log *slog.Logger) (PersistedState, LoadStatus, error) {
	if log == nil {
		log = slog.Default()
	}

	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return PersistedState{}, LoadMissing, nil
		}
		// Permission denied or other transient FS error — caller decides.
		return PersistedState{}, LoadUnreadable, fmt.Errorf("health.Load: stat: %w", err)
	}
	if info.Size() > maxBaselineFileSize {
		log.Warn("health baseline file exceeds max size, quarantining",
			"path", path, "size", info.Size(), "max", maxBaselineFileSize)
		if qerr := quarantine(path, now); qerr != nil {
			return PersistedState{}, LoadCorrupt, fmt.Errorf("health.Load: quarantine oversized: %w", qerr)
		}
		return PersistedState{}, LoadCorrupt, fmt.Errorf("health.Load: file size %d exceeds %d", info.Size(), maxBaselineFileSize)
	}

	data, err := readCapped(path)
	if err != nil {
		return PersistedState{}, LoadUnreadable, fmt.Errorf("health.Load: read: %w", err)
	}

	var sf savedFile
	if err := json.Unmarshal(data, &sf); err != nil {
		return quarantineAndReturn(path, now, log, "corrupt json", "err", err.Error())
	}
	if sf.SchemaVersion != CurrentSchemaVersion {
		// Distinguish "newer than us" (rolling-upgrade case) from
		// "older/invalid than us" (corruption or downgrade-and-quarantine).
		// A newer-version file means a post-upgrade agent wrote it; if a
		// rollback or autoscaler race lands this older binary on the same
		// persist path, we must NOT destroy that file — the newer agent
		// will come back. Just log + calibrate.
		if sf.SchemaVersion > CurrentSchemaVersion {
			log.Warn("health baseline was written by a newer schema_version; preserving file and calibrating",
				"path", path,
				"file_schema_version", sf.SchemaVersion,
				"this_schema_version", CurrentSchemaVersion)
			return PersistedState{}, LoadNewerVersion, nil
		}
		return quarantineAndReturn(path, now, log, "unsupported schema_version",
			"schema_version", sf.SchemaVersion)
	}
	if sf.SampleCount < 0 {
		return quarantineAndReturn(path, now, log, "negative sample_count",
			"sample_count", sf.SampleCount)
	}
	if !baselinesFinite(sf.FastEMA) || !baselinesFinite(sf.HardFloor) {
		return quarantineAndReturn(path, now, log, "non-finite baseline values")
	}
	if !baselinesNonNegative(sf.FastEMA) || !baselinesNonNegative(sf.HardFloor) {
		return quarantineAndReturn(path, now, log, "negative baseline values")
	}

	// Clamp a clock-backwards SavedAt so a file "from the future" is
	// treated as stale rather than silently considered fresh forever.
	age := now.Sub(sf.SavedAt)
	if age < 0 {
		log.Warn("health baseline saved_at is in the future, treating as stale",
			"path", path, "saved_at", sf.SavedAt, "now", now)
		age = math.MaxInt64
	}
	if age > maxAge {
		log.Info("health baseline file is stale, discarding",
			"path", path, "age", age.String(), "max_age", maxAge.String())
		if rerr := os.Remove(path); rerr != nil && !errors.Is(rerr, fs.ErrNotExist) {
			log.Warn("health baseline stale-remove failed",
				"path", path, "err", rerr.Error())
		}
		return PersistedState{}, LoadStale, nil
	}

	return PersistedState{
		SchemaVersion: sf.SchemaVersion,
		SampleCount:   sf.SampleCount,
		FastAlpha:     sf.FastAlpha,
		FloorAlpha:    sf.FloorAlpha,
		FastEMA:       sf.FastEMA,
		HardFloor:     sf.HardFloor,
	}, LoadFresh, nil
}

// readCapped reads at most maxBaselineFileSize+1 bytes from path and
// releases the file handle before returning. Closing eagerly matters on
// Windows where an open file cannot be renamed or removed.
func readCapped(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	data, rerr := io.ReadAll(io.LimitReader(f, maxBaselineFileSize+1))
	cerr := f.Close()
	if rerr != nil {
		return nil, rerr
	}
	if cerr != nil {
		return nil, cerr
	}
	return data, nil
}

// quarantineAndReturn wraps the common "log + move aside + return
// LoadCorrupt" path. Keeps Load() linear and single-branch per check.
func quarantineAndReturn(path string, now time.Time, log *slog.Logger, reason string, kv ...any) (PersistedState, LoadStatus, error) {
	args := append([]any{"path", path, "reason", reason}, kv...)
	log.Warn("health baseline file is corrupt, quarantining", args...)
	if qerr := quarantine(path, now); qerr != nil {
		return PersistedState{}, LoadCorrupt, fmt.Errorf("health.Load: quarantine: %w", qerr)
	}
	return PersistedState{}, LoadCorrupt, nil
}

// quarantine renames path to path.corrupt-{nanos} so it's preserved for
// forensic inspection but does not block startup. Because os.Rename
// overwrites the destination on both POSIX and Windows, the quarantine
// target is pre-checked with os.Stat; a short counter suffix is used
// when a prior quarantine shares the same nanosecond timestamp.
//
// The stat-then-rename sequence is TOCTOU-sensitive: two agents racing
// to quarantine the same file could both win the stat and then one
// rename overwrites the other's forensic copy. Acceptable trade-off for
// a best-effort archival operation — data integrity of the live
// baseline is unaffected.
func quarantine(path string, now time.Time) error {
	base := now.UTC().Format("20060102T150405.000000000Z")
	baseDst := fmt.Sprintf("%s.corrupt-%s", path, base)
	for i := 0; i < maxQuarantineSuffix; i++ {
		candidate := baseDst
		if i > 0 {
			candidate = fmt.Sprintf("%s.%d", baseDst, i)
		}
		if _, err := os.Stat(candidate); err != nil {
			if !errors.Is(err, fs.ErrNotExist) {
				return fmt.Errorf("stat quarantine candidate %s: %w", candidate, err)
			}
			// Candidate does not exist — claim it.
			if rerr := os.Rename(path, candidate); rerr != nil {
				return fmt.Errorf("rename %s -> %s: %w", path, candidate, rerr)
			}
			return nil
		}
		// Candidate exists — try the next suffix.
	}
	return fmt.Errorf("quarantine: exhausted %d collision suffixes", maxQuarantineSuffix)
}
