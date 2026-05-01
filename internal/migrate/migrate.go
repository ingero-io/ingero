// Package migrate ships the v0.11 framework for SQLite schema
// migrations on the local trace DB at ~/.ingero/ingero.db. v0.11
// itself defines no migrations; the package's job is to declare the
// contract so future versions can land migrations without adding CLI
// surface or breaking older agents reading newer DBs.
//
// The contract:
//
//   - schema_version is a single-row table with one int column.
//     Created lazily on first Plan() call; defaults to 0.
//   - migrations slice in this file is ordered, append-only. Each
//     entry has a unique Version > 0, a short Name, and an Up()
//     function that returns nil on success.
//   - Plan(path) inspects the DB's current schema_version and returns
//     the list of pending migrations (those whose Version is greater
//     than the DB's current value).
//   - Apply(path, plan) runs the pending migrations in order inside a
//     single transaction per migration. Bumps schema_version on each
//     success; rolls back on the first failure and returns the
//     number applied so far.
//   - If the DB's schema_version is NEWER than the binary's max
//     known version, Plan returns ErrSchemaNewer to prevent an old
//     binary from corrupting a DB written by a newer one.
package migrate

import (
	"database/sql"
	"errors"
	"fmt"

	// SQLite driver — same modernc.org/sqlite as the rest of the agent
	// uses (see internal/store). No CGO required.
	_ "modernc.org/sqlite"
)

// ErrSchemaNewer is returned by Plan when the DB's schema_version is
// strictly greater than the binary's max known migration version.
// Operators see this when they downgrade the agent below the version
// that wrote the DB. The fix is to upgrade the binary or to delete
// the DB if the recording is disposable.
var ErrSchemaNewer = errors.New("migrate: DB schema_version is newer than this binary supports")

// Migration declares one forward step.
type Migration struct {
	Version int                   // strictly increasing, > 0
	Name    string                // short, kebab-case, used in log lines
	Up      func(tx *sql.Tx) error // executed inside a single transaction
}

// migrations is the ordered, append-only list of all migrations the
// binary knows about. v0.11 defines none; future versions append.
//
// Append a new migration by adding a new entry with the next Version
// number. Never reorder, renumber, or rewrite a published entry —
// older binaries will already have applied it under the original ID.
var migrations = []Migration{
	// v0.11.0: framework only, no migrations.
}

// MaxVersion returns the highest known migration version, or 0 when
// the binary defines no migrations (v0.11.0).
func MaxVersion() int {
	if len(migrations) == 0 {
		return 0
	}
	return migrations[len(migrations)-1].Version
}

// PlanResult enumerates what Apply would do.
type PlanResult struct {
	CurrentVersion int         // schema_version row in the DB before any apply
	Pending        []Migration // migrations whose Version > CurrentVersion, in order
}

// Plan opens the DB, reads (or initializes) schema_version, and
// returns the list of migrations whose Version is greater than the
// stored value. The DB is closed before return.
func Plan(path string) (PlanResult, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return PlanResult{}, fmt.Errorf("open %s: %w", path, err)
	}
	defer db.Close()

	if _, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS schema_version (
			version INTEGER NOT NULL
		)
	`); err != nil {
		return PlanResult{}, fmt.Errorf("create schema_version: %w", err)
	}

	var current int
	row := db.QueryRow(`SELECT version FROM schema_version LIMIT 1`)
	if err := row.Scan(&current); err != nil {
		if !errors.Is(err, sql.ErrNoRows) {
			return PlanResult{}, fmt.Errorf("read schema_version: %w", err)
		}
		// First run on this DB; seed at 0.
		if _, err := db.Exec(`INSERT INTO schema_version(version) VALUES (0)`); err != nil {
			return PlanResult{}, fmt.Errorf("seed schema_version: %w", err)
		}
		current = 0
	}

	if current > MaxVersion() {
		return PlanResult{CurrentVersion: current}, fmt.Errorf("%w (db=%d, max=%d)", ErrSchemaNewer, current, MaxVersion())
	}

	plan := PlanResult{CurrentVersion: current}
	for _, m := range migrations {
		if m.Version > current {
			plan.Pending = append(plan.Pending, m)
		}
	}
	return plan, nil
}

// Apply runs every pending migration in order. Each migration runs in
// its own transaction; a failure rolls back the failing migration
// only and returns the count applied so far. Returns the number of
// migrations that were applied successfully.
func Apply(path string, plan PlanResult) (int, error) {
	if len(plan.Pending) == 0 {
		return 0, nil
	}

	db, err := sql.Open("sqlite", path)
	if err != nil {
		return 0, fmt.Errorf("open %s: %w", path, err)
	}
	defer db.Close()

	applied := 0
	for _, m := range plan.Pending {
		if err := applyOne(db, m); err != nil {
			return applied, fmt.Errorf("v%d %s: %w", m.Version, m.Name, err)
		}
		applied++
	}
	return applied, nil
}

func applyOne(db *sql.DB, m Migration) error {
	tx, err := db.Begin()
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if err := m.Up(tx); err != nil {
		return fmt.Errorf("up: %w", err)
	}
	if _, err := tx.Exec(`UPDATE schema_version SET version = ?`, m.Version); err != nil {
		return fmt.Errorf("bump: %w", err)
	}
	return tx.Commit()
}
