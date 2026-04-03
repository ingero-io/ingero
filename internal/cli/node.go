package cli

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// ResolveNodeIdentity returns the node name using precedence:
//   1. CLI --node flag (if non-empty)
//   2. Config file agent.node value (if non-empty)
//   3. os.Hostname()
//
// Returns an error if the resolved name contains a colon (reserved as ID separator).
func ResolveNodeIdentity(cliFlag string) (string, error) {
	node := cliFlag

	// Precedence 2: config file.
	if node == "" {
		node = readNodeFromConfig()
	}

	// Precedence 3: hostname.
	if node == "" {
		h, err := os.Hostname()
		if err != nil {
			return "", fmt.Errorf("resolving node identity: %w", err)
		}
		node = h
	}

	if strings.Contains(node, ":") {
		return "", fmt.Errorf("node name %q contains colon — colon is reserved as ID separator", node)
	}

	return node, nil
}

// readNodeFromConfig reads the agent.node value from ingero.yaml.
// Searches standard config locations. Returns "" if not found or empty.
func readNodeFromConfig() string {
	paths := configSearchPaths()
	for _, p := range paths {
		if v := parseNodeFromYAML(p); v != "" {
			return v
		}
	}
	return ""
}

// configSearchPaths returns candidate paths for ingero.yaml.
func configSearchPaths() []string {
	var paths []string

	// 1. INGERO_CONFIG env var.
	if p := os.Getenv("INGERO_CONFIG"); p != "" {
		paths = append(paths, p)
	}

	// 2. Current directory.
	paths = append(paths, "ingero.yaml")

	// 3. /etc/ingero/ingero.yaml.
	paths = append(paths, "/etc/ingero/ingero.yaml")

	// 4. ~/.config/ingero/ingero.yaml.
	if home, err := os.UserHomeDir(); err == nil {
		paths = append(paths, home+"/.config/ingero/ingero.yaml")
	}

	return paths
}

// ReadFleetNodes returns the fleet node list from config, or nil if not configured.
// CLI --nodes flag takes precedence (handled by caller).
func ReadFleetNodes() []string {
	paths := configSearchPaths()
	for _, p := range paths {
		if nodes := parseFleetNodesFromYAML(p); len(nodes) > 0 {
			return nodes
		}
	}
	return nil
}

// parseFleetNodesFromYAML reads the fleet.nodes list from ingero.yaml.
// Supports both inline [host:port, ...] and multi-line YAML list formats.
func parseFleetNodesFromYAML(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	inFleet := false
	inNodes := false
	var nodes []string

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}

		// Top-level section detection.
		if len(line) > 0 && line[0] != ' ' && line[0] != '\t' {
			if inFleet {
				break // left fleet section
			}
			inFleet = strings.HasPrefix(trimmed, "fleet:")
			inNodes = false
			continue
		}

		if !inFleet {
			continue
		}

		// Detect "nodes:" key within fleet section.
		if strings.HasPrefix(trimmed, "nodes:") {
			inNodes = true
			// Check for inline list: nodes: [host1:8443, host2:8443]
			val := strings.TrimPrefix(trimmed, "nodes:")
			val = strings.TrimSpace(val)
			if strings.HasPrefix(val, "[") {
				val = strings.Trim(val, "[]")
				for _, n := range strings.Split(val, ",") {
					n = strings.TrimSpace(n)
					n = strings.Trim(n, `"'`)
					if n != "" {
						nodes = append(nodes, n)
					}
				}
				return nodes
			}
			continue
		}

		// Multi-line YAML list items: "  - host:port"
		if inNodes && strings.HasPrefix(trimmed, "- ") {
			val := strings.TrimPrefix(trimmed, "- ")
			val = strings.TrimSpace(val)
			val = strings.Trim(val, `"'`)
			if val != "" {
				nodes = append(nodes, val)
			}
			continue
		}

		// Any other key under fleet: stops the nodes list.
		if inNodes && !strings.HasPrefix(trimmed, "- ") {
			inNodes = false
		}
	}
	return nodes
}

// parseNodeFromYAML does a minimal parse of ingero.yaml to extract the
// agent.node value. No YAML library dependency — just scans for
// "  node:" under the "agent:" section.
func parseNodeFromYAML(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	inAgent := false
	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		// Skip comments and empty lines.
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}

		// Detect top-level section.
		if len(line) > 0 && line[0] != ' ' && line[0] != '\t' {
			inAgent = strings.HasPrefix(trimmed, "agent:")
			continue
		}

		if inAgent && strings.HasPrefix(trimmed, "node:") {
			val := strings.TrimPrefix(trimmed, "node:")
			val = strings.TrimSpace(val)
			// Remove surrounding quotes.
			val = strings.Trim(val, `"'`)
			return val
		}
	}
	return ""
}
