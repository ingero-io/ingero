package cli

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveNodeIdentity(t *testing.T) {
	tests := []struct {
		name     string
		cliFlag  string
		wantErr  bool
		errMsg   string
	}{
		{
			name:    "CLI flag takes precedence",
			cliFlag: "my-node",
		},
		{
			name:    "empty flag falls back to hostname",
			cliFlag: "",
		},
		{
			name:    "colon in name is rejected",
			cliFlag: "node:bad",
			wantErr: true,
			errMsg:  "colon",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResolveNodeIdentity(tt.cliFlag)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error, got nil")
				} else if tt.errMsg != "" && !contains(err.Error(), tt.errMsg) {
					t.Errorf("error = %q, want to contain %q", err.Error(), tt.errMsg)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.cliFlag != "" && got != tt.cliFlag {
				t.Errorf("got %q, want %q", got, tt.cliFlag)
			}
			if tt.cliFlag == "" && got == "" {
				t.Errorf("hostname fallback returned empty string")
			}
		})
	}
}

func TestNodeNameValidation(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{"simple hostname", "gpu-node-07", false},
		{"dotted hostname", "node.cluster.local", false},
		{"numeric", "12345", false},
		{"colon rejected", "node:1", true},
		{"multiple colons rejected", "a:b:c", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ResolveNodeIdentity(tt.input)
			if tt.wantErr && err == nil {
				t.Errorf("expected error for %q", tt.input)
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error for %q: %v", tt.input, err)
			}
		})
	}
}

func TestResolveNodeFromConfig(t *testing.T) {
	// Create a temp config file with agent.node set.
	dir := t.TempDir()
	configPath := filepath.Join(dir, "ingero.yaml")
	err := os.WriteFile(configPath, []byte(`# Test config
agent:
  node: "config-node"
  log_level: info
`), 0o644)
	if err != nil {
		t.Fatalf("writing config: %v", err)
	}

	// Set INGERO_CONFIG to point to our temp file.
	t.Setenv("INGERO_CONFIG", configPath)

	got, err := ResolveNodeIdentity("") // empty CLI flag → should read config
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "config-node" {
		t.Errorf("got %q, want %q", got, "config-node")
	}
}

func TestCLIOverridesConfig(t *testing.T) {
	// Create config with one node, but CLI flag should win.
	dir := t.TempDir()
	configPath := filepath.Join(dir, "ingero.yaml")
	os.WriteFile(configPath, []byte(`agent:
  node: "config-value"
`), 0o644)
	t.Setenv("INGERO_CONFIG", configPath)

	got, err := ResolveNodeIdentity("cli-value")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "cli-value" {
		t.Errorf("got %q, want %q (CLI should override config)", got, "cli-value")
	}
}

func TestParseNodeFromYAML(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    string
	}{
		{
			name: "basic",
			content: `agent:
  node: "worker-3"
`,
			want: "worker-3",
		},
		{
			name: "unquoted",
			content: `agent:
  node: worker-3
`,
			want: "worker-3",
		},
		{
			name: "empty value",
			content: `agent:
  node: ""
`,
			want: "",
		},
		{
			name: "missing node",
			content: `agent:
  log_level: info
`,
			want: "",
		},
		{
			name: "node in wrong section",
			content: `store:
  node: wrong-section
agent:
  log_level: info
`,
			want: "",
		},
		{
			name: "comment ignored",
			content: `agent:
  # node: commented-out
  node: actual-value
`,
			want: "actual-value",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "ingero.yaml")
			os.WriteFile(path, []byte(tt.content), 0o644)

			got := parseNodeFromYAML(path)
			if got != tt.want {
				t.Errorf("parseNodeFromYAML() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestParseFleetNodesFromYAML(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    []string
	}{
		{
			name: "inline list",
			content: `fleet:
  nodes: [host1:8443, host2:8443]
`,
			want: []string{"host1:8443", "host2:8443"},
		},
		{
			name: "multi-line list",
			content: `fleet:
  nodes:
    - host1:8443
    - host2:8443
    - host3:8443
`,
			want: []string{"host1:8443", "host2:8443", "host3:8443"},
		},
		{
			name: "empty list",
			content: `fleet:
  nodes: []
`,
			want: nil,
		},
		{
			name: "no fleet section",
			content: `agent:
  node: test
`,
			want: nil,
		},
		{
			name: "quoted values",
			content: `fleet:
  nodes: ["host1:8443", "host2:8443"]
`,
			want: []string{"host1:8443", "host2:8443"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "ingero.yaml")
			os.WriteFile(path, []byte(tt.content), 0o644)

			got := parseFleetNodesFromYAML(path)
			if len(got) != len(tt.want) {
				t.Errorf("parseFleetNodesFromYAML() = %v, want %v", got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("node[%d] = %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestReadFleetNodes_ConfigOverride(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "ingero.yaml")
	os.WriteFile(configPath, []byte(`fleet:
  nodes: [node1:8443, node2:8443]
`), 0o644)
	t.Setenv("INGERO_CONFIG", configPath)

	nodes := ReadFleetNodes()
	if len(nodes) != 2 {
		t.Errorf("ReadFleetNodes() = %v, want 2 nodes", nodes)
	}
}

func TestResolveFleetNodes_CLIOverridesConfig(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "ingero.yaml")
	os.WriteFile(configPath, []byte(`fleet:
  nodes: [config-node:8443]
`), 0o644)
	t.Setenv("INGERO_CONFIG", configPath)

	// CLI flag should override config.
	nodes := resolveFleetNodes("cli-node:8443,cli-node2:8443")
	if len(nodes) != 2 {
		t.Fatalf("resolveFleetNodes() = %v, want 2 nodes from CLI", nodes)
	}
	if nodes[0] != "cli-node:8443" {
		t.Errorf("nodes[0] = %q, want %q", nodes[0], "cli-node:8443")
	}
}

func TestResolveFleetNodes_CLIBrackets(t *testing.T) {
	// Brackets around node list should be stripped (inline YAML format).
	nodes := resolveFleetNodes("[172.31.43.134:8080,172.31.40.229:8080]")
	if len(nodes) != 2 {
		t.Fatalf("resolveFleetNodes() = %v, want 2 nodes", nodes)
	}
	if nodes[0] != "172.31.43.134:8080" {
		t.Errorf("nodes[0] = %q, want %q", nodes[0], "172.31.43.134:8080")
	}
	if nodes[1] != "172.31.40.229:8080" {
		t.Errorf("nodes[1] = %q, want %q", nodes[1], "172.31.40.229:8080")
	}
}

func TestResolveFleetNodes_NoFlag_FallsToConfig(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "ingero.yaml")
	os.WriteFile(configPath, []byte(`fleet:
  nodes: [config-node:8443]
`), 0o644)
	t.Setenv("INGERO_CONFIG", configPath)

	nodes := resolveFleetNodes("")
	if len(nodes) != 1 || nodes[0] != "config-node:8443" {
		t.Errorf("resolveFleetNodes() = %v, want [config-node:8443]", nodes)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsSubstring(s, substr))
}

func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
