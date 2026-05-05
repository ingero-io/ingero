package nvml

// NVML clock-throttle reason bit constants. Stable across driver releases
// (defined in <nvml.h>). Reproduced here verbatim so the agent does not
// have to link the NVIDIA C headers.
const (
	ReasonGpuIdle                   uint64 = 0x0000000000000001
	ReasonApplicationsClocksSetting uint64 = 0x0000000000000002
	ReasonSwPowerCap                uint64 = 0x0000000000000004
	ReasonHwSlowdown                uint64 = 0x0000000000000008
	ReasonSyncBoost                 uint64 = 0x0000000000000010
	ReasonSwThermalSlowdown         uint64 = 0x0000000000000020
	ReasonHwThermalSlowdown         uint64 = 0x0000000000000040
	ReasonHwPowerBrakeSlowdown      uint64 = 0x0000000000000080
	ReasonDisplayClockSetting       uint64 = 0x0000000000000100
)

// knownReasons is the OR of every bit DecodeReasons recognises. Used to
// detect future NVML reasons that should funnel into the hw umbrella so
// dashboards do not silently lose throttle visibility on a newer driver.
const knownReasons = ReasonGpuIdle |
	ReasonApplicationsClocksSetting |
	ReasonSwPowerCap |
	ReasonHwSlowdown |
	ReasonSyncBoost |
	ReasonSwThermalSlowdown |
	ReasonHwThermalSlowdown |
	ReasonHwPowerBrakeSlowdown |
	ReasonDisplayClockSetting

// ThrottleBuckets is the four-way decode of an NVML throttle bitmask. Each
// field maps directly to one OTel metric:
//
//	gpu.throttle.power_active
//	gpu.throttle.thermal_active
//	gpu.throttle.sw_active
//	gpu.throttle.hw_active
//
// Multiple buckets can be true at once; a thermally throttled GPU is
// often also power-capped.
type ThrottleBuckets struct {
	Power   bool
	Thermal bool
	SW      bool
	HW      bool
}

// DecodeReasons maps an NVML throttle bitmask to the four user-facing
// buckets. Mapping table (matches the v0.12.10 plan and the CHANGELOG
// caveat block):
//
//	NVML bit                                | bucket(s)
//	----------------------------------------+-----------------
//	nvmlClocksThrottleReasonGpuIdle         | (none, suppressed)
//	nvmlClocksThrottleReasonAppClocksSet    | sw
//	nvmlClocksThrottleReasonSwPowerCap      | power, sw
//	nvmlClocksThrottleReasonHwSlowdown      | hw
//	nvmlClocksThrottleReasonSyncBoost       | sw
//	nvmlClocksThrottleReasonSwThermal       | thermal, sw
//	nvmlClocksThrottleReasonHwThermal       | thermal, hw
//	nvmlClocksThrottleReasonHwPowerBrake    | power, hw
//	nvmlClocksThrottleReasonDisplayClockSet | sw
//
// The hw bucket is an umbrella catch-all so any future NVML hw_* reason
// surfaces in `gpu.throttle.hw_active` until a more specific bucket
// exists. Unknown bits set the hw bucket to true.
func DecodeReasons(bitmask uint64) ThrottleBuckets {
	var b ThrottleBuckets
	if bitmask == 0 {
		return b
	}
	if bitmask&ReasonApplicationsClocksSetting != 0 {
		b.SW = true
	}
	if bitmask&ReasonSwPowerCap != 0 {
		b.Power = true
		b.SW = true
	}
	if bitmask&ReasonHwSlowdown != 0 {
		b.HW = true
	}
	if bitmask&ReasonSyncBoost != 0 {
		b.SW = true
	}
	if bitmask&ReasonSwThermalSlowdown != 0 {
		b.Thermal = true
		b.SW = true
	}
	if bitmask&ReasonHwThermalSlowdown != 0 {
		b.Thermal = true
		b.HW = true
	}
	if bitmask&ReasonHwPowerBrakeSlowdown != 0 {
		b.Power = true
		b.HW = true
	}
	if bitmask&ReasonDisplayClockSetting != 0 {
		b.SW = true
	}
	if bitmask&^knownReasons != 0 {
		b.HW = true
	}
	return b
}
