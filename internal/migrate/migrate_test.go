package migrate

import (
	"database/sql"
	"errors"
	"fmt"
	"path/filepath"
	"testing"
)

// Plan on a fresh DB seeds schema_version=0 and returns no pending
// migrations (v0.11.0 defines none).
func TestPlan_FreshDB_v011_NoMigrations(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fresh.db")
	plan, err := Plan(path)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if plan.CurrentVersion != 0 {
		t.Errorf("CurrentVersion=%d, want 0", plan.CurrentVersion)
	}
	if len(plan.Pending) != 0 {
		t.Errorf("Pending=%v, want empty", plan.Pending)
	}
}

// Apply on an empty plan is a clean no-op.
func TestApply_EmptyPlan_NoOp(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fresh.db")
	plan, err := Plan(path)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	n, err := Apply(path, plan)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if n != 0 {
		t.Errorf("applied=%d, want 0", n)
	}
}

// A DB written by a future binary (schema_version=99) trips
// ErrSchemaNewer instead of being mutated. Older binaries must refuse
// to operate on newer DBs.
func TestPlan_RefusesNewerSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "future.db")

	// Seed schema_version=99 manually so this test does not depend
	// on Plan's seed code.
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if _, err := db.Exec(`CREATE TABLE schema_version (version INTEGER NOT NULL)`); err != nil {
		t.Fatalf("create: %v", err)
	}
	if _, err := db.Exec(`INSERT INTO schema_version(version) VALUES (99)`); err != nil {
		t.Fatalf("insert: %v", err)
	}
	db.Close()

	_, err = Plan(path)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrSchemaNewer) {
		t.Errorf("err=%v, want ErrSchemaNewer", err)
	}
}

// TestPlan_RefusesNewerSchema_RealSeed covers v0.12.1 (QA #6):
// goes through Plan's own seed code instead of the hand-crafted
// CREATE+INSERT in TestPlan_RefusesNewerSchema. After the first
// Plan succeeds, manually bump schema_version to 99 and re-Plan,
// asserting ErrSchemaNewer fires. Locks both code paths together
// so a future change to Plan's seed (table name, column shape) can't
// silently break the refuses-newer-schema check.
func TestPlan_RefusesNewerSchema_RealSeed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "real-seed.db")

	// First Plan succeeds and seeds schema_version=0 via real code.
	plan, err := Plan(path)
	if err != nil {
		t.Fatalf("first Plan: %v", err)
	}
	if plan.CurrentVersion != 0 {
		t.Fatalf("first Plan currentVersion=%d, want 0", plan.CurrentVersion)
	}

	// Bump schema_version using whatever shape Plan's seed code produced.
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if _, err := db.Exec(`UPDATE schema_version SET version = 99`); err != nil {
		t.Fatalf("UPDATE: %v", err)
	}
	db.Close()

	// Second Plan must error.
	_, err = Plan(path)
	if err == nil {
		t.Fatal("re-Plan: expected error, got nil")
	}
	if !errors.Is(err, ErrSchemaNewer) {
		t.Errorf("re-Plan err=%v, want ErrSchemaNewer", err)
	}
}

// MaxVersion returns 0 in v0.11.0 (no migrations registered).
func TestMaxVersion_v011(t *testing.T) {
	if got := MaxVersion(); got != 0 {
		t.Errorf("MaxVersion=%d, want 0 in v0.11.0", got)
	}
}

// TestApply_FailingMigrationRollsBackSchemaVersion covers v0.12.1
// (QA #4): when a migration's Up() returns an error, the transaction
// is rolled back and schema_version stays at the pre-Apply value.
// Pre-fix this code path was uncovered (v0.11 ships zero migrations).
// First real schema migration (v0.12.x or v1.0) hits this path on
// every operator's box; a botched rollback corrupts the agent DB.
func TestApply_FailingMigrationRollsBackSchemaVersion(t *testing.T) {
	dbPath := tempDBPath(t)

	// Seed the DB so schema_version exists at 0.
	plan, err := Plan(dbPath)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}

	// Install a stub failing migration via plan.Pending.
	plan.Pending = []Migration{{
		Version: 1,
		Name:    "stub_failing",
		Up: func(tx *sql.Tx) error {
			// Make a real change inside the tx (so rollback is observable),
			// then fail. Without rollback, half-applied state would leak.
			if _, err := tx.Exec(`CREATE TABLE never_committed(id INTEGER)`); err != nil {
				return err
			}
			return fmt.Errorf("intentional failure")
		},
	}}

	applied, err := Apply(dbPath, plan)
	if err == nil {
		t.Fatal("Apply must return error for failing migration")
	}
	if applied != 0 {
		t.Errorf("applied=%d, want 0 (failing migration rolled back)", applied)
	}

	// Verify schema_version unchanged.
	plan2, err := Plan(dbPath)
	if err != nil {
		t.Fatalf("re-plan: %v", err)
	}
	if plan2.CurrentVersion != 0 {
		t.Errorf("schema_version=%d after failed Apply, want 0", plan2.CurrentVersion)
	}

	// Verify the table that was created inside the rolled-back tx
	// is NOT present.
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	var count int
	row := db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='never_committed'`)
	if err := row.Scan(&count); err != nil {
		t.Fatalf("query sqlite_master: %v", err)
	}
	if count != 0 {
		t.Errorf("never_committed table leaked through rollback")
	}
}

// tempDBPath returns a unique path under t.TempDir for sqlite-open.
func tempDBPath(t *testing.T) string {
	t.Helper()
	return t.TempDir() + "/migrate-test.db"
}
