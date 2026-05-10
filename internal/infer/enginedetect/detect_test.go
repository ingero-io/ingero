package enginedetect

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeFakeCmdline creates /<dir>/<pid>/cmdline with NUL-separated
// args, mirroring what the kernel exposes via /proc.
func writeFakeCmdline(t *testing.T, procDir string, pid uint32, args ...string) {
	t.Helper()
	dir := filepath.Join(procDir, "12345_TMP")
	_ = dir
	pidDir := filepath.Join(procDir, itoa(pid))
	if err := os.MkdirAll(pidDir, 0o755); err != nil {
		t.Fatal(err)
	}
	body := strings.Join(args, "\x00") + "\x00"
	if err := os.WriteFile(filepath.Join(pidDir, "cmdline"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func itoa(n uint32) string {
	// Hand-rolled to avoid importing strconv just for this helper.
	if n == 0 {
		return "0"
	}
	var buf [12]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

func TestListEnginePIDs_FindsKnownEngines(t *testing.T) {
	// v0.16.4 #10: ListEnginePIDs walks /proc and returns PIDs whose
	// cmdline matches a known engine. Unknown PIDs and non-numeric
	// /proc entries are skipped.
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 100, "python", "-m", "vllm.entrypoints.openai.api_server", "--port", "8000")
	writeFakeCmdline(t, dir, 200, "text-generation-launcher", "--port", "8080")
	writeFakeCmdline(t, dir, 300, "/usr/bin/bash") // not an engine
	writeFakeCmdline(t, dir, 400, "tritonserver")

	// Put a non-numeric directory in /proc to confirm it's skipped.
	if err := os.MkdirAll(filepath.Join(dir, "self"), 0o755); err != nil {
		t.Fatal(err)
	}

	got := ListEnginePIDs(dir)
	want := map[uint32]bool{100: true, 200: true, 400: true}
	if len(got) != len(want) {
		t.Errorf("got %d pids, want %d (got=%v)", len(got), len(want), got)
	}
	for _, pid := range got {
		if !want[pid] {
			t.Errorf("unexpected PID %d in result", pid)
		}
		delete(want, pid)
	}
	if len(want) > 0 {
		t.Errorf("missing PIDs: %v", want)
	}
}

func TestListEnginePIDs_EmptyOrMissingProc(t *testing.T) {
	if got := ListEnginePIDs("/this/does/not/exist"); got != nil {
		t.Errorf("missing /proc should return nil, got %v", got)
	}
	dir := t.TempDir()
	if got := ListEnginePIDs(dir); got != nil {
		t.Errorf("empty /proc should return nil, got %v", got)
	}
}

func TestDetect_VLLM_Modulestyle(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 100,
		"python", "-m", "vllm.entrypoints.openai.api_server",
		"--model", "meta-llama/Llama-3-70b",
		"--port", "8000")
	d, ok := detectAt(dir, 100)
	if !ok {
		t.Fatal("expected vLLM detection")
	}
	if d.Engine != VLLM {
		t.Errorf("Engine = %v, want %v", d.Engine, VLLM)
	}
	if d.Port != 8000 {
		t.Errorf("Port = %d, want 8000", d.Port)
	}
	if d.Model != "meta-llama/Llama-3-70b" {
		t.Errorf("Model = %q, want meta-llama/Llama-3-70b", d.Model)
	}
}

func TestDetect_VLLM_ServeSubcommand(t *testing.T) {
	dir := t.TempDir()
	// `vllm serve <model>` form (post v0.5+).
	writeFakeCmdline(t, dir, 200,
		"vllm", "serve", "meta-llama/Llama-3-7b",
		"--port=8001")
	d, ok := detectAt(dir, 200)
	if !ok {
		t.Fatal("expected vLLM detection")
	}
	if d.Engine != VLLM {
		t.Errorf("Engine = %v, want %v", d.Engine, VLLM)
	}
	if d.Port != 8001 {
		t.Errorf("Port = %d, want 8001 (--port= form)", d.Port)
	}
}

func TestDetect_VLLM_NoPortFallsToDefault(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 201, "vllm", "serve", "fake-model")
	d, _ := detectAt(dir, 201)
	if d.Port != 8000 {
		t.Errorf("Port = %d, want default 8000", d.Port)
	}
}

func TestDetect_TGI(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 300,
		"text-generation-launcher",
		"--model-id", "bigscience/bloom-7b",
		"--port", "8080")
	d, ok := detectAt(dir, 300)
	if !ok {
		t.Fatal("expected TGI detection")
	}
	if d.Engine != TGI {
		t.Errorf("Engine = %v, want %v", d.Engine, TGI)
	}
	if d.Model != "bigscience/bloom-7b" {
		t.Errorf("Model = %q, want bigscience/bloom-7b", d.Model)
	}
	if d.Port != 8080 {
		t.Errorf("Port = %d, want 8080", d.Port)
	}
}

func TestDetect_TGI_DefaultPort(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 301, "text-generation-launcher", "--model-id", "x")
	d, _ := detectAt(dir, 301)
	if d.Port != 8080 {
		t.Errorf("TGI default port should be 8080, got %d", d.Port)
	}
}

func TestDetect_SGLang(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 400,
		"python", "-m", "sglang.launch_server",
		"--model-path", "Qwen/Qwen2-72B",
		"--port", "30000")
	d, ok := detectAt(dir, 400)
	if !ok {
		t.Fatal("expected SGLang detection")
	}
	if d.Engine != SGLang {
		t.Errorf("Engine = %v, want %v", d.Engine, SGLang)
	}
	if d.Port != 30000 {
		t.Errorf("Port = %d, want 30000", d.Port)
	}
	// Model identifier comes from `--model-path`; the canonical SGLang
	// flag. Without this assertion the original B-003 bug (extractModel
	// reading `--model` only) was invisible to the test suite.
	if d.Model != "Qwen/Qwen2-72B" {
		t.Errorf("Model = %q, want Qwen/Qwen2-72B", d.Model)
	}
}

func TestDetect_SGLang_ModelPathEqualsForm(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 401,
		"python", "-m", "sglang.launch_server",
		"--model-path=Qwen/Qwen2-7B-Instruct",
		"--port=30001")
	d, ok := detectAt(dir, 401)
	if !ok {
		t.Fatal("expected SGLang detection")
	}
	if d.Engine != SGLang {
		t.Errorf("Engine = %v, want %v", d.Engine, SGLang)
	}
	if d.Model != "Qwen/Qwen2-7B-Instruct" {
		t.Errorf("Model = %q, want Qwen/Qwen2-7B-Instruct", d.Model)
	}
}

func TestDetect_SGLang_LegacyModelFlag(t *testing.T) {
	dir := t.TempDir()
	// Older SGLang builds accept `--model` (no `-path` suffix); the
	// extractor falls back to it after the canonical key returns no match.
	writeFakeCmdline(t, dir, 402,
		"python", "-m", "sglang.launch_server",
		"--model", "meta-llama/Llama-3.1-8B")
	d, ok := detectAt(dir, 402)
	if !ok {
		t.Fatal("expected SGLang detection")
	}
	if d.Engine != SGLang {
		t.Errorf("Engine = %v, want %v", d.Engine, SGLang)
	}
	if d.Model != "meta-llama/Llama-3.1-8B" {
		t.Errorf("Model = %q, want meta-llama/Llama-3.1-8B", d.Model)
	}
}

func TestDetect_Triton(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 500,
		"tritonserver", "--model-repository=/models", "--http-port=8002")
	d, ok := detectAt(dir, 500)
	if !ok {
		t.Fatal("expected Triton detection")
	}
	if d.Engine != Triton {
		t.Errorf("Engine = %v, want %v", d.Engine, Triton)
	}
	if d.Port != 8002 {
		t.Errorf("Port = %d, want 8002", d.Port)
	}
	// Triton's --model-repository is a path, not a model id; we
	// should NOT extract it as Model.
	if d.Model != "" {
		t.Errorf("Triton Model = %q, want empty", d.Model)
	}
}

func TestDetect_RandomPython_NotEngine(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 600, "python3", "-m", "training.epoch_loop")
	if _, ok := detectAt(dir, 600); ok {
		t.Error("non-engine python process should not be detected")
	}
}

func TestDetect_VllmHelperNoFalseMatch(t *testing.T) {
	dir := t.TempDir()
	// User-built CLI named "vllm-helper" without 'serve' arg should
	// NOT match (the rule requires standalone 'serve' arg).
	writeFakeCmdline(t, dir, 700, "vllm-helper", "--config", "vllm.yaml")
	if _, ok := detectAt(dir, 700); ok {
		t.Error("vllm-helper should not false-match vLLM")
	}
}

func TestDetect_EmptyCmdline(t *testing.T) {
	dir := t.TempDir()
	pidDir := filepath.Join(dir, itoa(800))
	_ = os.MkdirAll(pidDir, 0o755)
	_ = os.WriteFile(filepath.Join(pidDir, "cmdline"), nil, 0o644)
	if _, ok := detectAt(dir, 800); ok {
		t.Error("empty cmdline should not detect anything")
	}
}

func TestDetect_MissingPid(t *testing.T) {
	dir := t.TempDir()
	if _, ok := detectAt(dir, 999); ok {
		t.Error("missing /proc/PID should not detect")
	}
}

func TestEngine_DefaultPort(t *testing.T) {
	cases := map[Engine]uint16{
		VLLM:          8000,
		TGI:           8080,
		SGLang:        30000,
		Triton:        8002,
		UnknownEngine: 0,
	}
	for e, want := range cases {
		if got := e.DefaultPort(); got != want {
			t.Errorf("%v.DefaultPort() = %d, want %d", e, got, want)
		}
	}
}

func TestEngine_IsKnown(t *testing.T) {
	cases := map[Engine]bool{
		VLLM:          true,
		TGI:           true,
		SGLang:        true,
		Triton:        true,
		UnknownEngine: false,
		Engine("nope"): false,
	}
	for e, want := range cases {
		if got := e.IsKnown(); got != want {
			t.Errorf("%v.IsKnown() = %v, want %v", e, got, want)
		}
	}
}
