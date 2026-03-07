package discover

import (
	"testing"
)

func TestParseKernelMajorMinor(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantMajor int
		wantMinor int
		wantErr   bool
	}{
		{
			name:      "Ubuntu 22.04 kernel",
			input:     "5.15.0-100-generic",
			wantMajor: 5,
			wantMinor: 15,
		},
		{
			name:      "Ubuntu 24.04 kernel",
			input:     "6.8.0-45-generic",
			wantMajor: 6,
			wantMinor: 8,
		},
		{
			name:      "simple version",
			input:     "5.15.0",
			wantMajor: 5,
			wantMinor: 15,
		},
		{
			name:      "major.minor only",
			input:     "6.1",
			wantMajor: 6,
			wantMinor: 1,
		},
		{
			name:    "garbage input",
			input:   "notaversion",
			wantErr: true,
		},
		{
			name:    "empty string",
			input:   "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			major, minor, err := ParseKernelMajorMinor(tt.input)

			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error for input %q, got major=%d minor=%d", tt.input, major, minor)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error for input %q: %v", tt.input, err)
			}

			if major != tt.wantMajor || minor != tt.wantMinor {
				t.Errorf("ParseKernelMajorMinor(%q) = (%d, %d), want (%d, %d)",
					tt.input, major, minor, tt.wantMajor, tt.wantMinor)
			}
		})
	}
}

func TestExtractPathFromMapsLine(t *testing.T) {
	tests := []struct {
		name string
		line string
		want string
	}{
		{
			name: "standard CUDA mapping",
			line: "7f1234560000-7f1234580000 r-xp 00000000 08:01 12345 /usr/local/cuda/lib64/libcudart.so.12",
			want: "/usr/local/cuda/lib64/libcudart.so.12",
		},
		{
			name: "conda CUDA",
			line: "7f0000000000-7f0000020000 r--p 00000000 fd:01 99999 /opt/conda/lib/libcudart.so.12.2.140",
			want: "/opt/conda/lib/libcudart.so.12.2.140",
		},
		{
			name: "no CUDA in line",
			line: "7f1234560000-7f1234580000 r-xp 00000000 08:01 12345 /usr/lib/libc.so.6",
			want: "",
		},
		{
			name: "unversioned libcudart",
			line: "7f0000000000-7f0000020000 r--p 00000000 fd:01 99999 /usr/lib/libcudart.so",
			want: "/usr/lib/libcudart.so",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractPathFromMapsLine(tt.line)
			if got != tt.want {
				t.Errorf("extractPathFromMapsLine(%q) = %q, want %q", tt.line, got, tt.want)
			}
		})
	}
}

func TestInt8sToString(t *testing.T) {
	tests := []struct {
		name  string
		input []int8
		want  string
	}{
		{
			name:  "null terminated",
			input: []int8{'L', 'i', 'n', 'u', 'x', 0, 0, 0},
			want:  "Linux",
		},
		{
			name:  "no null terminator",
			input: []int8{'L', 'i', 'n', 'u', 'x'},
			want:  "Linux",
		},
		{
			name:  "empty",
			input: []int8{0},
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := int8sToString(tt.input)
			if got != tt.want {
				t.Errorf("int8sToString(%v) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestCPUModel(t *testing.T) {
	model := CPUModel()
	if model == "" {
		t.Skip("/proc/cpuinfo not available or empty")
	}
	// Sanity: should contain some alphanumeric characters.
	if len(model) < 3 {
		t.Errorf("CPUModel() = %q — suspiciously short", model)
	}
}

func TestCPUCores(t *testing.T) {
	cores := CPUCores()
	if cores <= 0 {
		t.Errorf("CPUCores() = %d, expected > 0", cores)
	}
}

func TestOSRelease(t *testing.T) {
	release := OSRelease()
	if release == "" {
		t.Skip("/etc/os-release not available")
	}
	if len(release) < 3 {
		t.Errorf("OSRelease() = %q — suspiciously short", release)
	}
}

func TestCUDAProcessString(t *testing.T) {
	p := CUDAProcess{
		PID:         12345,
		Name:        "python3",
		LibCUDAPath: "/usr/local/cuda/lib64/libcudart.so.12",
	}

	got := p.String()
	want := "PID 12345 (python3) → /usr/local/cuda/lib64/libcudart.so.12"
	if got != want {
		t.Errorf("CUDAProcess.String() = %q, want %q", got, want)
	}
}
