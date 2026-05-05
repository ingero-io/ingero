package nvml

import "testing"

// TestDecodeReasons_PerBit walks every documented NVML bit and asserts the
// bucket mapping matches the v0.12.10 plan table. Closes QA audit ★5 H1.
func TestDecodeReasons_PerBit(t *testing.T) {
	cases := []struct {
		name string
		bit  uint64
		want ThrottleBuckets
	}{
		{"GpuIdle", ReasonGpuIdle, ThrottleBuckets{}},
		{"AppClocksSetting", ReasonApplicationsClocksSetting, ThrottleBuckets{SW: true}},
		{"SwPowerCap", ReasonSwPowerCap, ThrottleBuckets{Power: true, SW: true}},
		{"HwSlowdown", ReasonHwSlowdown, ThrottleBuckets{HW: true}},
		{"SyncBoost", ReasonSyncBoost, ThrottleBuckets{SW: true}},
		{"SwThermal", ReasonSwThermalSlowdown, ThrottleBuckets{Thermal: true, SW: true}},
		{"HwThermal", ReasonHwThermalSlowdown, ThrottleBuckets{Thermal: true, HW: true}},
		{"HwPowerBrake", ReasonHwPowerBrakeSlowdown, ThrottleBuckets{Power: true, HW: true}},
		{"DisplayClock", ReasonDisplayClockSetting, ThrottleBuckets{SW: true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := DecodeReasons(tc.bit)
			if got != tc.want {
				t.Fatalf("bit %#x: got %+v, want %+v", tc.bit, got, tc.want)
			}
		})
	}
}

// TestDecodeReasons_NoBits asserts that a zero bitmask yields a zero
// bucket struct. Idle is the steady state; we do not want a falsely-
// active metric emitted when the GPU is fine.
func TestDecodeReasons_NoBits(t *testing.T) {
	got := DecodeReasons(0)
	if got != (ThrottleBuckets{}) {
		t.Fatalf("zero bitmask must yield empty buckets, got %+v", got)
	}
}

// TestDecodeReasons_ConcurrentBits asserts the realistic case where a GPU
// is simultaneously hardware-thermal-throttled, software-power-capped,
// and hardware-power-braked. All four buckets must resolve true. Closes
// QA audit ★5 H1.
func TestDecodeReasons_ConcurrentBits(t *testing.T) {
	mask := ReasonHwThermalSlowdown | ReasonSwPowerCap | ReasonHwPowerBrakeSlowdown
	got := DecodeReasons(mask)
	want := ThrottleBuckets{Power: true, Thermal: true, SW: true, HW: true}
	if got != want {
		t.Fatalf("concurrent bits %#x: got %+v, want %+v", mask, got, want)
	}
}

// TestDecodeReasons_UnknownBit covers ★4 H2: an unknown high bit (future
// NVML reason) must surface in the hw umbrella so dashboards do not
// silently lose throttle visibility on a newer driver.
func TestDecodeReasons_UnknownBit(t *testing.T) {
	const future uint64 = 1 << 40
	got := DecodeReasons(future)
	if !got.HW {
		t.Fatalf("unknown bit %#x must set HW=true (umbrella), got %+v", future, got)
	}
	if got.Power || got.Thermal || got.SW {
		t.Fatalf("unknown bit %#x must NOT set power/thermal/sw, got %+v", future, got)
	}
}

// TestDecodeReasons_GpuIdleSuppressed asserts that the GpuIdle bit alone,
// or combined with other bits, does not contribute to any bucket. Idle
// is steady-state, not a throttle.
func TestDecodeReasons_GpuIdleSuppressed(t *testing.T) {
	got := DecodeReasons(ReasonGpuIdle)
	if got != (ThrottleBuckets{}) {
		t.Fatalf("GpuIdle alone must yield empty buckets, got %+v", got)
	}
	// Idle + SwPowerCap should still report power+sw (not affected by idle).
	got = DecodeReasons(ReasonGpuIdle | ReasonSwPowerCap)
	want := ThrottleBuckets{Power: true, SW: true}
	if got != want {
		t.Fatalf("idle+SwPowerCap: got %+v, want %+v", got, want)
	}
}
