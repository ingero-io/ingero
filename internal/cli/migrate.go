package cli

import (
	"errors"
	"fmt"

	"github.com/ingero-io/ingero/internal/migrate"
	"github.com/ingero-io/ingero/internal/store"
	"github.com/spf13/cobra"
)

var (
	migrateDryRun bool
	migrateDBPath string
)

// migrateCmd ships the v0.11 framework for SQLite schema migrations.
// v0.11 itself defines no migrations; the command's job is to declare
// the contract so future versions can land migrations without adding
// CLI surface or breaking older agents reading newer DBs.
var migrateCmd = &cobra.Command{
	Use:   "migrate",
	Short: "Apply pending SQLite schema migrations",
	Long: `Apply any pending schema migrations to the local trace DB.

The DB at ~/.ingero/ingero.db (or --db-path) carries a schema_version
row. ingero migrate compares it against the binary's current
migration list and applies missing forward migrations in order.

v0.11 defines zero migrations; the command is a no-op on every fresh
install. Future versions add migrations here without changing how
operators invoke the command.

The command refuses to operate on a DB whose schema_version is NEWER
than the binary expects, to prevent an old binary from corrupting a
DB written by a newer one.`,
	RunE: runMigrate,
}

func init() {
	migrateCmd.Flags().BoolVar(&migrateDryRun, "dry-run", false,
		"Report what would be applied without modifying the DB")
	migrateCmd.Flags().StringVar(&migrateDBPath, "db-path", store.DefaultDBPath(),
		"Path to the SQLite trace DB")
	rootCmd.AddCommand(migrateCmd)
}

func runMigrate(cmd *cobra.Command, args []string) error {
	plan, err := migrate.Plan(migrateDBPath)
	if err != nil {
		if errors.Is(err, migrate.ErrSchemaNewer) {
			return fmt.Errorf("DB schema is newer than this binary expects (refusing to downgrade): %w", err)
		}
		return fmt.Errorf("plan: %w", err)
	}

	if len(plan.Pending) == 0 {
		fmt.Printf("ingero migrate: %s is at schema_version=%d (current); no migrations pending\n",
			migrateDBPath, plan.CurrentVersion)
		return nil
	}

	fmt.Printf("ingero migrate: %s is at schema_version=%d; %d migration(s) pending:\n",
		migrateDBPath, plan.CurrentVersion, len(plan.Pending))
	for _, m := range plan.Pending {
		fmt.Printf("  - v%d: %s\n", m.Version, m.Name)
	}

	if migrateDryRun {
		fmt.Println("ingero migrate: --dry-run set; not applying.")
		return nil
	}

	applied, err := migrate.Apply(migrateDBPath, plan)
	if err != nil {
		return fmt.Errorf("apply: %w", err)
	}
	fmt.Printf("ingero migrate: applied %d migration(s); DB is now at schema_version=%d\n",
		applied, plan.CurrentVersion+len(plan.Pending))
	return nil
}
