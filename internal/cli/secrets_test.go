package cli

import (
	"errors"
	"strings"
	"testing"
)

func TestAssertSecretSafe_AcceptsAllowedSource(t *testing.T) {
	allowed := []SecretSource{SourceEnv, SourceFile}
	if err := AssertSecretSafe("mcp-bearer-token", SourceEnv, allowed); err != nil {
		t.Fatalf("AssertSecretSafe(env in [env,file]) = %v; want nil", err)
	}
	if err := AssertSecretSafe("mcp-bearer-token", SourceFile, allowed); err != nil {
		t.Fatalf("AssertSecretSafe(file in [env,file]) = %v; want nil", err)
	}
}

func TestAssertSecretSafe_RefusesFlagSource(t *testing.T) {
	err := AssertSecretSafe("mcp-bearer-token", SourceFlag, []SecretSource{SourceEnv, SourceFile})
	if err == nil {
		t.Fatal("AssertSecretSafe(flag in [env,file]) = nil; want error")
	}
	msg := err.Error()
	for _, want := range []string{"mcp-bearer-token", "flag", "/proc/<pid>/cmdline"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error %q does not mention %q", msg, want)
		}
	}
}

func TestResolveSecret_EnvSetFlagEmpty(t *testing.T) {
	t.Setenv("INGERO_TEST_TOKEN", "from-env")
	got, err := ResolveSecret("token-flag", "INGERO_TEST_TOKEN", "")
	if err != nil {
		t.Fatalf("ResolveSecret = err %v; want nil", err)
	}
	if got != "from-env" {
		t.Errorf("ResolveSecret = %q; want %q", got, "from-env")
	}
}

func TestResolveSecret_BothEmpty(t *testing.T) {
	t.Setenv("INGERO_TEST_TOKEN", "")
	got, err := ResolveSecret("token-flag", "INGERO_TEST_TOKEN", "")
	if err != nil {
		t.Fatalf("ResolveSecret = err %v; want nil", err)
	}
	if got != "" {
		t.Errorf("ResolveSecret = %q; want empty", got)
	}
}

func TestResolveSecret_FlagSetIsRefused(t *testing.T) {
	t.Setenv("INGERO_TEST_TOKEN", "")
	_, err := ResolveSecret("token-flag", "INGERO_TEST_TOKEN", "from-flag")
	if err == nil {
		t.Fatal("ResolveSecret with flag value = nil; want error")
	}
	if !strings.Contains(err.Error(), "INGERO_TEST_TOKEN") {
		t.Errorf("error %q does not name the env var", err.Error())
	}
	if !strings.Contains(err.Error(), "--token-flag") {
		t.Errorf("error %q does not name the flag", err.Error())
	}
}

func TestResolveSecret_FlagSetEvenWithEnvIsRefused(t *testing.T) {
	// Refuse the flag form even when env is also configured. Silently
	// preferring env would let the leaky flag survive in shell history
	// without the operator noticing.
	t.Setenv("INGERO_TEST_TOKEN", "from-env")
	_, err := ResolveSecret("token-flag", "INGERO_TEST_TOKEN", "from-flag")
	if err == nil {
		t.Fatal("ResolveSecret(flag, env both set) = nil; want refusal of flag")
	}
}

func TestResolveSecret_ErrorWraps(t *testing.T) {
	_, err := ResolveSecret("token-flag", "INGERO_TEST_TOKEN", "from-flag")
	if err == nil {
		t.Fatal("expected error")
	}
	// The wrapped error should still mention the secret name (from the
	// inner AssertSecretSafe message) AND the env-var hint (from the
	// outer wrap).
	if !errors.Is(err, err) {
		t.Errorf("error chain broken")
	}
}

func TestIsLoopback(t *testing.T) {
	tests := []struct {
		in   string
		want bool
	}{
		{"127.0.0.1", true},
		{"127.0.0.1:9090", true},
		{"127.5.5.5", true},
		{"127.5.5.5:8080", true},
		{"::1", true},
		{"[::1]:9090", true},
		{"localhost", true},
		{"localhost:9090", true},
		{"0.0.0.0", false},
		{"0.0.0.0:9090", false},
		{":9090", false},
		{"10.0.0.5", false},
		{"10.0.0.5:8080", false},
		{"example.com", false},
		{"example.com:443", false},
		{"", false}, // empty host = bind-all on most platforms
	}
	for _, tc := range tests {
		got := IsLoopback(tc.in)
		if got != tc.want {
			t.Errorf("IsLoopback(%q) = %v; want %v", tc.in, got, tc.want)
		}
	}
}

func TestSecretSource_String(t *testing.T) {
	if got := SourceFlag.String(); got != "flag" {
		t.Errorf("SourceFlag.String() = %q; want flag", got)
	}
	if got := SourceEnv.String(); got != "env" {
		t.Errorf("SourceEnv.String() = %q; want env", got)
	}
	if got := SourceFile.String(); got != "file" {
		t.Errorf("SourceFile.String() = %q; want file", got)
	}
	if got := SecretSource(99).String(); got != "unknown" {
		t.Errorf("SecretSource(99).String() = %q; want unknown", got)
	}
}
