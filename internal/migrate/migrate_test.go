package migrate

import (
	"database/sql"
	"errors"
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

// MaxVersion returns 0 in v0.11.0 (no migrations registered).
func TestMaxVersion_v011(t *testing.T) {
	if got := MaxVersion(); got != 0 {
		t.Errorf("MaxVersion=%d, want 0 in v0.11.0", got)
	}
}
