package cli

import (
	"database/sql"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	_ "modernc.org/sqlite"

	"github.com/ingero-io/ingero/internal/store"
)

var (
	mergeOutput         string
	mergeForceNode      string
	mergeClockSkew      string
)

var mergeCmd = &cobra.Command{
	Use:   "merge [source.db ...]",
	Short: "Merge SQLite databases from multiple Ingero nodes into one",
	Long: `Merge SQLite databases from multiple Ingero nodes into a single queryable
database for offline cross-node analysis. Use this in air-gapped environments
or when you prefer offline analysis over fan-out queries.

The merged database works with standard ingero query and ingero explain.

Examples:
  ingero merge node-a.db node-b.db node-c.db -o cluster.db
  ingero merge old.db --force-node legacy -o merged.db
  ingero query -d cluster.db --since 1h
  ingero explain -d cluster.db`,

	Args: cobra.MinimumNArgs(1),
	RunE: mergeRunE,
}

func init() {
	mergeCmd.Flags().StringVarP(&mergeOutput, "output", "o", "", "output database path (required)")
	mergeCmd.MarkFlagRequired("output")
	mergeCmd.Flags().StringVar(&mergeForceNode, "force-node", "", "assign this node name to databases missing the node column")
	mergeCmd.Flags().StringVar(&mergeClockSkew, "clock-skew-threshold", "100ms", "warn if session timestamps suggest clock skew exceeding this threshold")
	rootCmd.AddCommand(mergeCmd)
}

func mergeRunE(cmd *cobra.Command, args []string) error {
	// Validate output doesn't collide with sources (resolve paths to catch aliases).
	absOutput, _ := filepath.Abs(mergeOutput)
	for _, src := range args {
		absSrc, _ := filepath.Abs(src)
		if absSrc == absOutput {
			return fmt.Errorf("output path %q collides with source path %q", mergeOutput, src)
		}
	}

	// Validate source files exist.
	for _, src := range args {
		if _, err := os.Stat(src); err != nil {
			return fmt.Errorf("source database %q: %w", src, err)
		}
	}

	// Remove output if it exists (fresh merge).
	if err := os.Remove(mergeOutput); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("cannot remove existing output %q: %w", mergeOutput, err)
	}

	// Create output database with current schema.
	outStore, err := store.New(mergeOutput)
	if err != nil {
		return fmt.Errorf("creating output database: %w", err)
	}
	defer outStore.Close()

	outDB := outStore.DB()
	totalEvents := 0
	totalChains := 0
	totalStacks := 0
	stackCache := make(map[int64]bool) // track seen stack hashes
	forceNodeSeq := 0                  // shared counter for --force-node ID generation across all source DBs

	for _, srcPath := range args {
		fmt.Fprintf(os.Stderr, "  Merging %s...\n", srcPath)

		srcDB, err := sql.Open("sqlite", srcPath+"?mode=ro")
		if err != nil {
			return fmt.Errorf("opening %s: %w", srcPath, err)
		}

		// Check if source has node column.
		hasNode, err := hasColumn(srcDB, "events", "node")
		if err != nil {
			srcDB.Close()
			return fmt.Errorf("checking schema of %s: %w", srcPath, err)
		}

		if !hasNode && mergeForceNode == "" {
			srcDB.Close()
			return fmt.Errorf("database %s missing node column. Run ingero trace once to migrate schema, or use --force-node <name> to assign a node identity during merge", srcPath)
		}

		evts, err := copyEvents(srcDB, outDB, hasNode, mergeForceNode, &forceNodeSeq)
		if err != nil {
			srcDB.Close()
			return fmt.Errorf("copying events from %s: %w", srcPath, err)
		}
		totalEvents += evts

		chains, err := copyChains(srcDB, outDB, hasNode, mergeForceNode)
		if err != nil {
			srcDB.Close()
			return fmt.Errorf("copying chains from %s: %w", srcPath, err)
		}
		totalChains += chains

		stacks, err := copyStacks(srcDB, outDB, stackCache)
		if err != nil {
			srcDB.Close()
			return fmt.Errorf("copying stacks from %s: %w", srcPath, err)
		}
		totalStacks += stacks

		copySessions(srcDB, outDB, hasNode, mergeForceNode)
		copySnapshots(srcDB, outDB)
		copyProcessNames(srcDB, outDB)
		copyCGroupMetadata(srcDB, outDB)

		fmt.Fprintf(os.Stderr, "    %d events, %d chains, %d stacks\n", evts, chains, stacks)
		srcDB.Close()
	}

	// Validation.
	var dupeCount int
	outDB.QueryRow("SELECT COUNT(*) FROM (SELECT id FROM events GROUP BY id HAVING COUNT(*) > 1)").Scan(&dupeCount)
	if dupeCount > 0 {
		fmt.Fprintf(os.Stderr, "  WARNING: %d duplicate event IDs found in merged database\n", dupeCount)
	}

	fmt.Fprintf(os.Stderr, "\n  Merged %d database(s) → %s: %d events, %d chains, %d unique stacks\n",
		len(args), mergeOutput, totalEvents, totalChains, totalStacks)

	// Clock skew heuristic: compare earliest event timestamps per node.
	checkMergeClockSkew(outDB, mergeClockSkew)

	return nil
}

// checkMergeClockSkew compares earliest event timestamps between nodes and
// warns if the gap suggests clock skew. This is a heuristic — different workloads
// start at different times — but catches gross skew (> 100ms).
func checkMergeClockSkew(db *sql.DB, thresholdFlag string) {
	threshold := 100 * time.Millisecond
	if d, err := time.ParseDuration(thresholdFlag); err == nil {
		threshold = d
	}

	rows, err := db.Query("SELECT node, MIN(timestamp) FROM events WHERE node != '' GROUP BY node")
	if err != nil {
		return
	}
	defer rows.Close()

	type nodeTime struct {
		node string
		ts   int64
	}
	var nodes []nodeTime
	for rows.Next() {
		var nt nodeTime
		if err := rows.Scan(&nt.node, &nt.ts); err == nil {
			nodes = append(nodes, nt)
		}
	}

	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			diffNs := nodes[j].ts - nodes[i].ts
			diffMs := float64(diffNs) / 1e6
			if math.Abs(diffMs) > float64(threshold)/float64(time.Millisecond) {
				dir := "after"
				if diffMs < 0 {
					dir = "before"
					diffMs = -diffMs
				}
				fmt.Fprintf(os.Stderr, "  WARNING: Possible clock skew — %s earliest event is %.0fms %s %s\n",
					nodes[j].node, diffMs, dir, nodes[i].node)
			}
		}
	}
}

func hasColumn(db *sql.DB, table, column string) (bool, error) {
	rows, err := db.Query(fmt.Sprintf("PRAGMA table_info(%s)", table))
	if err != nil {
		return false, err
	}
	defer rows.Close()

	for rows.Next() {
		var cid int
		var name, typ string
		var notNull, pk int
		var dflt sql.NullString
		if err := rows.Scan(&cid, &name, &typ, &notNull, &dflt, &pk); err != nil {
			return false, err
		}
		if name == column {
			return true, nil
		}
	}
	return false, nil
}

func copyEvents(src, dst *sql.DB, hasNode bool, forceNode string, seqCounter *int) (int, error) {
	// Determine source columns based on schema version.
	hasRank, _ := hasColumn(src, "events", "rank")
	hasCgroup, _ := hasColumn(src, "events", "cgroup_id")
	hasComm, _ := hasColumn(src, "events", "comm")

	// Build SELECT for source.
	srcCols := "id, timestamp, pid, tid, source, op, duration, gpu_id, arg0, arg1, ret_code, stack_hash"
	if hasCgroup {
		srcCols += ", cgroup_id"
	}
	if hasNode {
		srcCols += ", node"
	}
	if hasRank {
		srcCols += ", rank, local_rank, world_size"
	}
	if hasComm {
		srcCols += ", comm"
	}

	rows, err := src.Query("SELECT " + srcCols + " FROM events")
	if err != nil {
		return 0, err
	}
	defer rows.Close()

	tx, err := dst.Begin()
	if err != nil {
		return 0, err
	}

	// Destination always has comm column (initialized with current schema), so
	// we always pass it — empty string for legacy source rows.
	stmt, err := tx.Prepare(`INSERT OR IGNORE INTO events
		(id, timestamp, pid, tid, source, op, duration, gpu_id, arg0, arg1, ret_code, stack_hash, cgroup_id, node, rank, local_rank, world_size, comm)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		tx.Rollback()
		return 0, err
	}
	defer stmt.Close()

	count := 0
	// seqCounter is shared across all source DBs to prevent duplicate IDs with --force-node.
	for rows.Next() {
		var (
			id                              interface{}
			timestamp, pid, tid             int64
			source, op                      int
			duration, gpuID                 int64
			arg0, arg1, retCode, stackHash  int64
			cgroupID                        int64
			node                            string
			rank, localRank, worldSize      sql.NullInt64
			comm                            string
		)

		// Scan based on available columns.
		scanArgs := []interface{}{&id, &timestamp, &pid, &tid, &source, &op, &duration, &gpuID, &arg0, &arg1, &retCode, &stackHash}
		if hasCgroup {
			scanArgs = append(scanArgs, &cgroupID)
		}
		if hasNode {
			scanArgs = append(scanArgs, &node)
		}
		if hasRank {
			scanArgs = append(scanArgs, &rank, &localRank, &worldSize)
		}
		if hasComm {
			scanArgs = append(scanArgs, &comm)
		}

		if err := rows.Scan(scanArgs...); err != nil {
			tx.Rollback()
			return 0, fmt.Errorf("scanning event: %w", err)
		}

		// Handle legacy DBs.
		if !hasNode {
			node = forceNode
		}

		// If the ID is an integer (legacy), generate a node-namespaced ID.
		idStr := fmt.Sprintf("%v", id)
		if !strings.Contains(idStr, ":") {
			nodeName := node
			if nodeName == "" {
				nodeName = "unknown"
			}
			*seqCounter++
			idStr = fmt.Sprintf("%s:%d", nodeName, *seqCounter)
		}

		var rankVal, localRankVal, worldSizeVal interface{}
		if hasRank && rank.Valid {
			rankVal = rank.Int64
		}
		if hasRank && localRank.Valid {
			localRankVal = localRank.Int64
		}
		if hasRank && worldSize.Valid {
			worldSizeVal = worldSize.Int64
		}

		stmt.Exec(idStr, timestamp, pid, tid, source, op, duration, gpuID, arg0, arg1, retCode, stackHash, cgroupID, node, rankVal, localRankVal, worldSizeVal, comm)
		count++

		if count%500 == 0 {
			tx.Commit()
			var txErr error
			tx, txErr = dst.Begin()
			if txErr != nil {
				return count, fmt.Errorf("begin transaction at row %d: %w", count, txErr)
			}
			stmt.Close()
			stmt, txErr = tx.Prepare(`INSERT OR IGNORE INTO events
				(id, timestamp, pid, tid, source, op, duration, gpu_id, arg0, arg1, ret_code, stack_hash, cgroup_id, node, rank, local_rank, world_size, comm)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
			if txErr != nil {
				tx.Rollback()
				return count, fmt.Errorf("prepare statement at row %d: %w", count, txErr)
			}
		}
	}

	tx.Commit()
	return count, rows.Err()
}

func copyChains(src, dst *sql.DB, hasNode bool, forceNode string) (int, error) {
	hasChainsNode, _ := hasColumn(src, "causal_chains", "node")

	srcCols := "id, detected_at, severity, summary, root_cause, explanation, recommendations, cuda_op, cuda_p99_us, cuda_p50_us, tail_ratio, timeline"
	if hasChainsNode {
		srcCols += ", node"
	}

	rows, err := src.Query("SELECT " + srcCols + " FROM causal_chains")
	if err != nil {
		// Table might not exist in old DBs.
		return 0, nil
	}
	defer rows.Close()

	tx, err := dst.Begin()
	if err != nil {
		return 0, err
	}

	stmt, err := tx.Prepare(`INSERT OR IGNORE INTO causal_chains
		(id, detected_at, severity, summary, root_cause, explanation, recommendations, cuda_op, cuda_p99_us, cuda_p50_us, tail_ratio, timeline, node)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		tx.Rollback()
		return 0, err
	}
	defer stmt.Close()

	count := 0
	for rows.Next() {
		var (
			id, severity, summary, rootCause, explanation string
			recommendations, cudaOp, timeline            string
			detectedAt, cudaP99, cudaP50                 int64
			tailRatio                                    float64
			node                                         string
		)

		scanArgs := []interface{}{&id, &detectedAt, &severity, &summary, &rootCause, &explanation, &recommendations, &cudaOp, &cudaP99, &cudaP50, &tailRatio, &timeline}
		if hasChainsNode {
			scanArgs = append(scanArgs, &node)
		}

		if err := rows.Scan(scanArgs...); err != nil {
			tx.Rollback()
			return 0, err
		}

		if !hasChainsNode {
			node = forceNode
		}

		stmt.Exec(id, detectedAt, severity, summary, rootCause, explanation, recommendations, cudaOp, cudaP99, cudaP50, tailRatio, timeline, node)
		count++
	}

	tx.Commit()
	return count, rows.Err()
}

func copyStacks(src, dst *sql.DB, cache map[int64]bool) (int, error) {
	rows, err := src.Query("SELECT hash, ips, frames FROM stack_traces")
	if err != nil {
		return 0, nil // table might not exist
	}
	defer rows.Close()

	tx, err := dst.Begin()
	if err != nil {
		return 0, err
	}

	stmt, err := tx.Prepare("INSERT OR IGNORE INTO stack_traces (hash, ips, frames) VALUES (?, ?, ?)")
	if err != nil {
		tx.Rollback()
		return 0, err
	}
	defer stmt.Close()

	count := 0
	for rows.Next() {
		var hash int64
		var ips, frames string
		if err := rows.Scan(&hash, &ips, &frames); err != nil {
			tx.Rollback()
			return 0, err
		}

		if cache[hash] {
			continue // deduplicate
		}
		cache[hash] = true

		stmt.Exec(hash, ips, frames)
		count++
	}

	tx.Commit()
	return count, rows.Err()
}

func copySessions(src, dst *sql.DB, hasNode bool, forceNode string) {
	hasSessionNode, _ := hasColumn(src, "sessions", "node")
	hasRank, _ := hasColumn(src, "sessions", "rank")

	srcCols := "started_at, stopped_at, gpu_model, gpu_driver, cpu_model, cpu_cores, mem_total, kernel, os_release, cuda_ver, python_ver, ingero_ver, pid_filter, flags"
	if hasSessionNode {
		srcCols += ", node"
	}
	if hasRank {
		srcCols += ", rank, local_rank, world_size"
	}

	rows, err := src.Query("SELECT " + srcCols + " FROM sessions")
	if err != nil {
		return
	}
	defer rows.Close()

	for rows.Next() {
		var (
			startedAt, stoppedAt                                    int64
			gpuModel, gpuDriver, cpuModel                          string
			cpuCores, memTotal                                     int64
			kernel, osRelease, cudaVer, pythonVer, ingeroVer       string
			pidFilter, flags                                       string
			node                                                   string
			rank, localRank, worldSize                             sql.NullInt64
		)

		scanArgs := []interface{}{&startedAt, &stoppedAt, &gpuModel, &gpuDriver, &cpuModel, &cpuCores, &memTotal, &kernel, &osRelease, &cudaVer, &pythonVer, &ingeroVer, &pidFilter, &flags}
		if hasSessionNode {
			scanArgs = append(scanArgs, &node)
		}
		if hasRank {
			scanArgs = append(scanArgs, &rank, &localRank, &worldSize)
		}

		if err := rows.Scan(scanArgs...); err != nil {
			continue
		}

		if !hasSessionNode {
			node = forceNode
		}

		var rankVal, localRankVal, worldSizeVal interface{}
		if rank.Valid {
			rankVal = rank.Int64
		}
		if localRank.Valid {
			localRankVal = localRank.Int64
		}
		if worldSize.Valid {
			worldSizeVal = worldSize.Int64
		}

		dst.Exec(`INSERT INTO sessions
			(started_at, stopped_at, gpu_model, gpu_driver, cpu_model, cpu_cores, mem_total,
			 kernel, os_release, cuda_ver, python_ver, ingero_ver, pid_filter, flags,
			 node, rank, local_rank, world_size)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			startedAt, stoppedAt, gpuModel, gpuDriver, cpuModel, cpuCores, memTotal,
			kernel, osRelease, cudaVer, pythonVer, ingeroVer, pidFilter, flags,
			node, rankVal, localRankVal, worldSizeVal)
	}
}

func copySnapshots(src, dst *sql.DB) {
	rows, err := src.Query("SELECT timestamp, cpu_pct, mem_pct, mem_avail, swap_mb, load_avg FROM system_snapshots")
	if err != nil {
		return
	}
	defer rows.Close()

	for rows.Next() {
		var ts int64
		var cpuPct, memPct, loadAvg float64
		var memAvail, swapMB int64
		if err := rows.Scan(&ts, &cpuPct, &memPct, &memAvail, &swapMB, &loadAvg); err != nil {
			continue
		}
		dst.Exec("INSERT OR IGNORE INTO system_snapshots (timestamp, cpu_pct, mem_pct, mem_avail, swap_mb, load_avg) VALUES (?, ?, ?, ?, ?, ?)",
			ts, cpuPct, memPct, memAvail, swapMB, loadAvg)
	}
}

func copyProcessNames(src, dst *sql.DB) {
	rows, err := src.Query("SELECT pid, name, seen_at FROM process_names")
	if err != nil {
		return
	}
	defer rows.Close()

	for rows.Next() {
		var pid int
		var name string
		var seenAt int64
		if err := rows.Scan(&pid, &name, &seenAt); err != nil {
			continue
		}
		dst.Exec("INSERT OR IGNORE INTO process_names (pid, name, seen_at) VALUES (?, ?, ?)", pid, name, seenAt)
	}
}

func copyCGroupMetadata(src, dst *sql.DB) {
	rows, err := src.Query("SELECT cgroup_id, container_id, cgroup_path, pod_name, namespace FROM cgroup_metadata")
	if err != nil {
		return
	}
	defer rows.Close()

	for rows.Next() {
		var cgroupID int64
		var containerID, cgroupPath, podName, namespace string
		if err := rows.Scan(&cgroupID, &containerID, &cgroupPath, &podName, &namespace); err != nil {
			continue
		}
		dst.Exec("INSERT OR IGNORE INTO cgroup_metadata (cgroup_id, container_id, cgroup_path, pod_name, namespace) VALUES (?, ?, ?, ?, ?)",
			cgroupID, containerID, cgroupPath, podName, namespace)
	}
}
