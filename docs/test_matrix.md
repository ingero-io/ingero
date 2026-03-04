# Ingero Unit Test Matrix

> **Maintenance rule**: Update this file every time tests are added or removed.

Last updated: 2026-03-04 (v0.8.0, 217 total tests)

## Summary

| Package | Tests | Coverage Focus |
|---------|-------|----------------|
| correlate | 48 | Causal chain engine, cross-source correlation |
| ebpf/blockio | 3 | Block I/O event parsing |
| ebpf/tcp | 4 | TCP retransmit event parsing, edge cases |
| ebpf/net | 5 | Network socket event parsing, edge cases |
| ebpf/cuda | 5 | CUDA runtime event + stack parsing |
| ebpf/driver | 5 | CUDA driver event + stack parsing, managed alloc |
| ebpf/host | 8 | Host kernel event + pod lifecycle parsing |
| store | 5 | SQLite storage, chain round-trip, batch process names |
| stats | 22 | Percentiles, anomaly detection, spike patterns |
| mcp | 9 | TSC compression, aggregate/chain formatting |
| cli | 10 | Duration format, storage hierarchy, time parsing, PID name cache |
| filter | 11 | Deadband suppression, heartbeat, concurrency |
| cgroup | 6 | Container ID extraction, cgroup v1/v2 parsing |
| k8s | 7 | Pod list parsing, GPU pod filtering, cache |
| export | 18 | Prometheus metrics, OTLP push, error handling |
| symtab | 11 | ELF parsing, DWARF offsets, Python detection |
| discover | 7 | Kernel version, maps parsing, CPU/OS info |
| sysinfo | 3 | /proc/stat CPU, /proc/meminfo parsing |
| synth | 3 | Demo scenario registry, event creation |
| update | 2 | Semver comparison, version parsing |
| events | 11 | Stack IP parsing, source/op string names |

## Correlate Engine — Causal Chain Tests

### v0.2 Correlation Tests (Legacy)

| # | Test | Description | File |
|---|------|-------------|------|
| 1 | TestSchedSwitchCorrelation | sched_switch events → cudaStreamSync correlation with tail ratio | correlate_test.go |
| 2 | TestNoCorrelationBelowThreshold | <5 sched_switch events → no correlation | correlate_test.go |
| 3 | TestNoCorrelationHealthyTail | sched_switch + healthy tail (ratio <3) → no correlation | correlate_test.go |
| 4 | TestOOMAlwaysCorrelates | OOM kill always produces correlation regardless of CUDA tail | correlate_test.go |
| 5 | TestPageAllocCorrelation | >1GB page allocations → cudaMalloc correlation | correlate_test.go |
| 6 | TestWindowPruning | Events older than maxAge pruned from sliding window | correlate_test.go |
| 7 | TestCorrelationString | Correlation.String() formatting | correlate_test.go |
| 8 | TestEmptyCorrelations | No events/ops → empty correlations | correlate_test.go |

### v0.3 Causal Chain Tests (Host + System)

| # | Test | Description | File |
|---|------|-------------|------|
| 9 | TestCausalChainWithSystemContext | System(CPU+Mem+Swap+Load) + Host(sched_switch) + CUDA → HIGH chain with all layers | correlate_test.go |
| 10 | TestCausalChainNoSystemContext | Host(sched_switch) + CUDA → MEDIUM chain without system triggers | correlate_test.go |
| 11 | TestCausalChainOOM | Host(oom_kill) + healthy CUDA → HIGH chain (OOM always fires) | correlate_test.go |
| 12 | TestCausalChainNoChainsWhenHealthy | Low system metrics + healthy CUDA → 0 chains | correlate_test.go |
| 13 | TestSetSystemSnapshot | SystemContext stored and retrievable | correlate_test.go |
| 14 | TestWithMaxAgeOption | Custom maxAge applied to engine | correlate_test.go |
| 15 | TestWithMaxAgeZeroDisablesPruning | maxAge=0 retains all events (historical replay mode) | correlate_test.go |
| 16 | TestHostOpsIgnoredInCUDAAnalysis | Host-only ops not correlated with themselves | correlate_test.go |

### v0.8 Single-Source Chain Tests (IO, TCP, Net)

| # | Test | Description | File |
|---|------|-------------|------|
| 17 | TestCausalChainBlockIO | >50 IO ops + CUDA anomaly → chain with IO layer | correlate_test.go |
| 18 | TestCausalChainBlockIOWithoutGPUAnomaly | Heavy IO + healthy CUDA → no chain (no false positive) | correlate_test.go |
| 19 | TestCausalChainBlockIOBelowThreshold | 10 small IO ops → no IO chain (below 50-count threshold) | correlate_test.go |
| 20 | TestCausalChainBlockIOByDuration | 5 slow IO ops (1s total >500ms threshold) → chain by duration | correlate_test.go |
| 21 | TestCausalChainBlockIOReadHeavy | Read-dominant IO → "DataLoader" recommendation | correlate_test.go |
| 22 | TestCausalChainBlockIOWriteHeavy | Write-dominant IO → "checkpoint" recommendation | correlate_test.go |
| 23 | TestCausalChainBlockIOSpinningDisk | Peak 30ms IO → "NVMe" rec + "spinning disk" explanation | correlate_test.go |
| 24 | TestCausalChainBlockIOSeverityHigh | Peak 80ms IO → HIGH severity escalation | correlate_test.go |
| 25 | TestCausalChainBlockIOThroughput | MB throughput appears in chain detail | correlate_test.go |
| 26 | TestCausalChainTCPRetransmit | >15 retransmits + CUDA anomaly → NET layer + TCP root cause | correlate_test.go |
| 27 | TestCausalChainTCPStandalone | >20 retransmits + healthy CUDA → standalone TCP chain | correlate_test.go |
| 28 | TestCausalChainTCPHighCountSeverity | >100 retransmits → HIGH severity | correlate_test.go |
| 29 | TestCausalChainTCPExplanation | Explanation mentions "retransmission" + "NCCL" | correlate_test.go |
| 30 | TestCausalChainNetSocket | >120 net ops + >2.4MB + CUDA anomaly → NET layer | correlate_test.go |
| 31 | TestCausalChainNetBelowThreshold | 50 net ops → no NET/socket_io layer | correlate_test.go |
| 32 | TestCausalChainIOPlusTCP | IO + TCP → both IO and NET layers in chain | correlate_test.go |
| 33 | TestCausalChainIOPlusTCPPlusNet | IO + TCP + Net → all three infra layers | correlate_test.go |
| 34 | TestCausalChainExplanationBlockIODetail | IO explanation: "Block I/O activity", "spinning disk", "Read-dominant" | correlate_test.go |
| 35 | TestCausalChainExplanationNetworkIO | Net explanation: "network socket I/O" | correlate_test.go |

### v0.8 Compound Chain Tests (Old + New Sensors)

| # | Test | Description | File |
|---|------|-------------|------|
| 36 | TestCompoundCPUPlusIO | System(CPU) + Host(sched_switch) + IO + CUDA → 4-layer chain | correlate_test.go |
| 37 | TestCompoundCPUPlusTCP | System(CPU) + Host(sched_switch) + TCP → all layers present | correlate_test.go |
| 38 | TestCompoundMemoryPressurePlusIO | System(swap+mem) + Host(page_alloc) + IO(slow) → HIGH + "NVMe" + "RAM" | correlate_test.go |
| 39 | TestCompoundOOMPlusIO | Host(oom_kill) + IO(writes) → OOM chain + IO-correlated chain | correlate_test.go |
| 40 | TestCompoundPodRestartPlusIO | Host(pod_restart) + IO → both chains produced | correlate_test.go |
| 41 | TestCompoundPodEvictionPlusTCP | Host(pod_eviction) + TCP → both chains produced | correlate_test.go |
| 42 | TestCompoundFullStack | ALL layers: System + Host + IO + TCP + Net + CUDA + HIGH + all explanations | correlate_test.go |
| 43 | TestCompoundFullStackNoGPUAnomaly | Full infra load + healthy GPU → standalone TCP chain only, no false CUDA chains | correlate_test.go |
| 44 | TestCompoundNoisyNeighborPlusIO | Noisy neighbor (cgroup p99 20x peer) + IO → both chains | correlate_test.go |
| 45 | TestPodRestartChainStandalone | pod_restart → HIGH standalone chain | correlate_test.go |
| 46 | TestPodEvictionChainStandalone | pod_eviction → HIGH standalone chain | correlate_test.go |
| 47 | TestPodOOMKillChain | pod_oom_kill → HIGH chain (treated same as kernel OOM) | correlate_test.go |
| 48a | TestNoisyNeighborWithPreFilterTracking | Pre-filter RecordCGroupSchedSwitch populates peer data → noisy neighbor fires | correlate_test.go |
| 48b | TestNoisyNeighborWithoutPreFilter | Without pre-filter, peer cgroup missing → noisy neighbor does NOT fire | correlate_test.go |

## eBPF Event Parsing

### Block I/O (blockio_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 48 | TestParseEventBlockRead | Parses block read event: duration, sectors, device | blockio_test.go |
| 49 | TestParseEventBlockWrite | Parses block write event with correct op | blockio_test.go |
| 50 | TestParseEventTooShort | Rejects buffer shorter than 64 bytes | blockio_test.go |

### TCP (tcp_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 51 | TestParseEventRetransmit | Parses TCP retransmit: IP addrs, ports, TCP state | tcp_test.go |
| 52 | TestParseEventRetransmitZeroPorts | Retransmit with zero ports (TIME_WAIT state) | tcp_test.go |
| 53 | TestParseEventRetransmitAddressPacking | (saddr<<32)\|daddr packing near uint32 max | tcp_test.go |
| 54 | TestParseEventTooShort | Rejects buffer shorter than 48 bytes | tcp_test.go |

### Network Socket (net_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 55 | TestParseEventNetSend | Parses net send: fd, bytes, direction | net_test.go |
| 56 | TestParseEventNetRecv | Parses net recv with correct fields | net_test.go |
| 57 | TestParseEventNetZeroBytes | Send with zero bytes (keepalive/probe) | net_test.go |
| 58 | TestParseEventNetLargeTransfer | 1 GiB recv (NCCL collective edge case) | net_test.go |
| 59 | TestParseEventTooShort | Rejects buffer shorter than 56 bytes | net_test.go |

### CUDA Runtime (cuda_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 60 | TestParseEventCUDAMalloc | Parses cudaMalloc: allocation size, return code | cuda_test.go |
| 61 | TestParseEventCUDAFree | Parses cudaFree: device pointer | cuda_test.go |
| 62 | TestParseEventTooShort | Rejects short buffers | cuda_test.go |
| 63 | TestParseEventWithStack | Parses stack trace section with IP addresses | cuda_test.go |
| 64 | TestParseEventTruncatedStack | Handles truncated stack sections gracefully | cuda_test.go |

### CUDA Driver (driver_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 65 | TestParseEventDriverLaunchKernel | Parses cuLaunchKernel: function handle | driver_test.go |
| 66 | TestParseEventDriverMemAllocManaged | Parses cuMemAllocManaged: alloc size, GPU ID | driver_test.go |
| 67 | TestParseEventDriverTooShort | Rejects short buffers | driver_test.go |
| 68 | TestParseEventDriverWithStack | Parses driver event with stack section | driver_test.go |
| 69 | TestParseEventDriverTruncatedStack | Handles truncated stack in driver events | driver_test.go |

### Host Kernel (host_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 70 | TestParseEventSchedSwitch | Parses sched_switch: off-CPU duration, CPU, target PID | host_test.go |
| 71 | TestParseEventPageAlloc | Parses mm_page_alloc: allocation bytes | host_test.go |
| 72 | TestParseEventOOMKill | Parses OOM kill: victim PID | host_test.go |
| 73 | TestParseEventSchedWakeup | Parses sched_wakeup: no duration | host_test.go |
| 74 | TestParseEventPodRestart | Parses pod_restart: target PID, cgroup ID | host_test.go |
| 75 | TestParseEventPodEviction | Parses pod_eviction: evicted PID | host_test.go |
| 76 | TestParseEventPodOOMKill | Parses pod_oom_kill: killed PID, cgroup ID | host_test.go |
| 77 | TestParseEventTooShort | Rejects short buffers | host_test.go |

## Statistics & Anomaly Detection (stats/collector_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 78 | TestPercentileEmpty | Empty buffer returns 0 | collector_test.go |
| 79 | TestPercentileSingleElement | Percentile with one sample | collector_test.go |
| 80 | TestPercentileKnownValues | p50/p95/p99 against hand-computed values | collector_test.go |
| 81 | TestPercentilePartialBuffer | Partially filled circular buffer | collector_test.go |
| 82 | TestPercentileDoesNotMutateSamples | Percentile preserves buffer order | collector_test.go |
| 83 | TestCollectorRecord | Records events correctly | collector_test.go |
| 84 | TestCollectorMultipleOps | Tracks multiple op types separately | collector_test.go |
| 85 | TestCollectorMinMax | Tracks min/max durations | collector_test.go |
| 86 | TestTimeFraction | total_duration / wall_clock computation | collector_test.go |
| 87 | TestAnomalyDetection | Events >3x median flagged as anomalous | collector_test.go |
| 88 | TestAnomalyNotFlaggedWithoutBaseline | No false anomalies without sufficient data | collector_test.go |
| 89 | TestSpikePatternDetection | Detects periodic spike patterns | collector_test.go |
| 90 | TestSpikePatternNoPattern | Irregular spikes → no pattern | collector_test.go |
| 91 | TestSpikePatternTooFewSpikes | <3 spikes → no pattern | collector_test.go |
| 92 | TestSpikePatternWithNoise | Detects patterns with ±25% noise | collector_test.go |
| 93 | TestCircularBufferWrap | Window wrapping preserves all-time min/max | collector_test.go |
| 94 | TestSnapshotEmpty | Fresh collector → empty snapshot | collector_test.go |
| 95 | TestZeroDurationEvents | Handles zero-duration events | collector_test.go |
| 96 | TestTimeFractionSumsReasonably | Time fractions bounded and finite | collector_test.go |
| 97 | TestWithWindowSize | Window size option controls sliding window | collector_test.go |
| 98 | TestWithAnomalyThreshold | Anomaly threshold adjusts sensitivity | collector_test.go |
| 99 | TestMixedSourcesNoCollision | CUDA and Host ops tracked separately | collector_test.go |

## Storage (store/store_test.go, lookup_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 100 | TestNewInMemory | Creates in-memory SQLite database | store_test.go |
| 101 | TestRecordAndQuery | Records events and queries with correct counts | store_test.go |
| 102 | TestCausalChainsRoundTrip | Stores and retrieves causal chains with timeline | lookup_test.go |
| 103 | TestLookupTablesCreated | Lookup tables created on DB init | lookup_test.go |
| 103a | TestRecordProcessNames | Batch PID→name persistence, skip empty, overwrite | store_test.go |

## CLI (cli/trace_test.go, pidutil_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 104 | TestFormatDuration | Human-readable duration (ns/us/ms/s/min) | trace_test.go |
| 105 | TestDebugf | Debug message respects debugMode flag | trace_test.go |
| 106 | TestShouldStoreHierarchy | 7-tier selective storage decision hierarchy | trace_test.go |
| 107 | TestSelectiveStoragePreservesSchedSwitchChain | Stored subset sufficient for causal chains | trace_test.go |
| 108 | TestPageAllocAggregateOnly | mm_page_alloc aggregated only, not stored individually | trace_test.go |
| 109 | TestIsStackResolved | Stack resolution detection | trace_test.go |
| 110 | TestShouldStoreStackSampling | Stack sampling limit with anomaly bypass | trace_test.go |
| 111 | TestShouldStoreStackSamplingAnomalyBypass | Anomalies bypass stack sample limits | trace_test.go |
| 111a | TestPIDNameCacheNames | Names() returns snapshot copy, nil-safe, mutation-safe | trace_test.go |
| 112 | TestToUint32Slice | int slice → uint32 slice, filtering zeros | pidutil_test.go |
| 113 | TestSinglePIDOrZero | Returns single PID or 0 if multiple/empty | pidutil_test.go |
| 114 | TestPidSetFromInts | Creates PID filter set from int list | pidutil_test.go |
| 115 | TestParseTime | Time string parsing: full datetime, ISO, time-only, errors | explain_test.go |

## MCP (mcp/server_test.go, tsc_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 116 | TestFormatAggregateStatsTSC | Aggregate stats with telegraphic compression | server_test.go |
| 117 | TestFormatAggregateStatsVerbose | Aggregate stats with verbose keys | server_test.go |
| 118 | TestFormatAggregateStatsEmpty | Empty ops list → valid JSON | server_test.go |
| 119 | TestFormatCausalChainsEmpty | Empty chains → healthy message | server_test.go |
| 120 | TestFormatCausalChainsVerbose | Verbose chain formatting: severity, layers, recommendations | server_test.go |
| 121 | TestFormatCausalChainsTSC | TSC chain formatting: valid JSON, abbreviated keys | server_test.go |
| 122 | TestFormatCausalChainsMultiple | Multiple chains: count header, all summaries present | server_test.go |
| 123 | TestTSCMapFromServerTest | TSCMap with tsc=false uses full key names | server_test.go |
| 124 | TestTSCKey | Field name → abbreviated key mapping | tsc_test.go |
| 125 | TestTSCMap | TSC abbreviation mode toggle | tsc_test.go |
| 126 | TestTSCReverseMap | Bidirectional reverse mapping | tsc_test.go |

## Filter (filter/deadband_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 127 | TestShouldEmit_DisabledByDefault | Disabled when deadband=0 | deadband_test.go |
| 128 | TestShouldEmit_FirstCallAlwaysEmits | First snapshot always emitted | deadband_test.go |
| 129 | TestShouldEmit_WithinDeadbandSuppressed | Metrics within deadband suppressed | deadband_test.go |
| 130 | TestShouldEmit_BeyondDeadbandEmits | Metrics beyond deadband trigger emit | deadband_test.go |
| 131 | TestShouldEmit_AnyMetricTriggersEmit | Any metric exceeding deadband triggers | deadband_test.go |
| 132 | TestShouldEmit_HeartbeatForcesEmit | Heartbeat forces emit even when unchanged | deadband_test.go |
| 133 | TestShouldEmit_ZeroBaseMetric | Zero-to-nonzero transitions detected | deadband_test.go |
| 134 | TestShouldEmit_NilFilter | Nil filter always returns true | deadband_test.go |
| 135 | TestShouldEmit_ConcurrentAccess | Safe concurrent access | deadband_test.go |
| 136 | TestExceedsDeadband_EdgeCases | Exact threshold, negative, zero old, large changes | deadband_test.go |
| 137 | TestConfig_Disabled | Deadband disable detection | deadband_test.go |

## Container & K8s (cgroup_test.go, client_test.go, pods_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 138 | TestParseContainerID | Extracts container IDs from cgroup v1/v2 paths | cgroup_test.go |
| 139 | TestParseCGroupFile_V2 | Parses cgroup v2 format (0::/path) | cgroup_test.go |
| 140 | TestParseCGroupFile_V1 | Parses cgroup v1 (multiple hierarchies, longest path) | cgroup_test.go |
| 141 | TestParseCGroupFile_V1Root | Root process on cgroup v1 | cgroup_test.go |
| 142 | TestParseCGroupFile_Empty | Empty file → empty result | cgroup_test.go |
| 143 | TestParseCGroupFile_MixedV1V2 | Prefers v2 (hierarchy 0) over v1 | cgroup_test.go |
| 144 | TestIsInCluster | Detects in-cluster from KUBERNETES_SERVICE_HOST | client_test.go |
| 145 | TestClientGet | K8s API authenticated GET with Bearer token | client_test.go |
| 146 | TestClientGetError | Handles HTTP 403 errors | client_test.go |
| 147 | TestTruncate | String truncation with ellipsis | client_test.go |
| 148 | TestNodeName | Returns node name from client | client_test.go |
| 149 | TestParsePodList | Parses pod list with GPU and non-GPU pods | pods_test.go |
| 150 | TestStripRuntimePrefix | Strips containerd/docker/cri-o prefixes | pods_test.go |
| 151 | TestPodCacheLookup | Cache lookups by container ID | pods_test.go |
| 152 | TestGPUPods | Filters GPU-requesting pods | pods_test.go |
| 153 | TestPodCacheGracefulDegradation | Cache degrades on API errors | pods_test.go |
| 154 | TestParsePodListEmpty | Empty pod list parsing | pods_test.go |
| 155 | TestParsePodListInvalidJSON | Rejects invalid JSON | pods_test.go |

## Export (export/export_test.go)

| # | Test | Description | File |
|---|------|-------------|------|
| 156 | TestNewPrometheus_NilOnEmpty | Nil when addr empty | export_test.go |
| 157 | TestNewPrometheus_NonNil | Non-nil when addr provided | export_test.go |
| 158 | TestPrometheusUpdateSnapshot_NilSafe | Nil-safe snapshot update | export_test.go |
| 159 | TestPrometheusMetrics_NoData | Metrics endpoint with no data | export_test.go |
| 160 | TestPrometheusMetrics_WithSnapshot | Metrics with CUDA/Host ops | export_test.go |
| 161 | TestPrometheusMetrics_HTTPHandler | Real HTTP handler works | export_test.go |
| 162 | TestNewOTLP_NilOnEmpty | Nil when endpoint empty | export_test.go |
| 163 | TestNewOTLP_NonNil | Non-nil when endpoint provided | export_test.go |
| 164 | TestNewOTLP_DefaultInterval | Default export interval 10s | export_test.go |
| 165 | TestNewOTLP_DefaultProtocol | Default protocol http | export_test.go |
| 166 | TestOTLP_Interval | Returns configured interval | export_test.go |
| 167 | TestOTLPPush_NilSafe | Nil-safe push | export_test.go |
| 168 | TestOTLPStats_NilSafe | Nil-safe stats | export_test.go |
| 169 | TestOTLP_MetricsURL | Correct OTLP metrics URL | export_test.go |
| 170 | TestOTLP_BuildPayload | Valid OTLP JSON payload | export_test.go |
| 171 | TestOTLP_PushToServer | Pushes to test OTLP server | export_test.go |
| 172 | TestOTLP_PushServerError | Handles server errors | export_test.go |
| 173 | TestOTLP_PushConnectionRefused | Handles connection errors | export_test.go |

## Symbol Resolution (symtab/)

| # | Test | Description | File |
|---|------|-------------|------|
| 174 | TestParseMapsLine | Parses /proc/[pid]/maps lines | procmaps_test.go |
| 175 | TestFindRegion | Binary search finds memory region | procmaps_test.go |
| 176 | TestMapRegionContains | Region containment check | procmaps_test.go |
| 177 | TestResolver_MergePythonFrames | Inserts Python frames before libpython | resolver_test.go |
| 178 | TestResolver_MergePythonFrames_NoPython | Prepends Python frames when no libpython | resolver_test.go |
| 179 | TestIsLibPythonFrame | Detects libpython/python binary paths | resolver_test.go |
| 180 | TestParseELFSymbols_SystemLib | Parses ELF symbols from libc | elfsyms_test.go |
| 181 | TestFindSymbolByName_FunctionSymbol | Finds function symbol by name | elfsyms_test.go |
| 182 | TestFindSymbolByName_NotFound | Returns not found for missing symbols | elfsyms_test.go |
| 183 | TestFindSymbolByName_InvalidPath | Handles invalid file paths | elfsyms_test.go |
| 184 | TestFindSymbolByName_PIEDetection | Detects PIE binaries | elfsyms_test.go |
| 185 | TestExtractStructOffsets_Libc | DWARF struct field offsets from libc | dwarf_test.go |
| 186 | TestBuildPyOffsetsFromDWARF | Python struct offsets from DWARF | dwarf_test.go |
| 187 | TestDetectPythonFromRegions | Detects Python 3.x from /proc/maps | python_test.go |
| 188 | TestDetectPythonFromRegions_PrefersLibpython | Prefers libpython.so over binary | python_test.go |
| 189 | TestDetectPythonFromRegions_FallbackToBinary | Falls back when libpython missing | python_test.go |

## Discovery, System Info, Synth, Update, Events

| # | Test | Description | File |
|---|------|-------------|------|
| 190 | TestParseKernelMajorMinor | Kernel version parsing (5.15.0 → 5, 15) | discover_test.go |
| 191 | TestExtractPathFromMapsLine | Library paths from /proc/[pid]/maps | discover_test.go |
| 192 | TestInt8sToString | Null-terminated int8 array → string | discover_test.go |
| 193 | TestCPUModel | CPU model from /proc/cpuinfo | discover_test.go |
| 194 | TestCPUCores | CPU core count | discover_test.go |
| 195 | TestOSRelease | OS release from /etc/os-release | discover_test.go |
| 196 | TestCUDAProcessString | CUDA process info string format | discover_test.go |
| 197 | TestParseCPULine | /proc/stat CPU line parsing | sysinfo_test.go |
| 198 | TestReadCPU | CPU utilization from /proc/stat | sysinfo_test.go |
| 199 | TestReadMeminfo | Memory info from /proc/meminfo | sysinfo_test.go |
| 200 | TestRegistryCompleteness | All 6 demo scenarios registered | synth_test.go |
| 201 | TestFindScenario | Finds scenarios by name and alias | synth_test.go |
| 202 | TestMakeEvent | Creates synthetic events with metadata | synth_test.go |
| 203 | TestIsNewer | Semver comparison (0.7.0 > 0.6) | check_test.go |
| 204 | TestParseSemver | Parses semver strings, strips pre-release | check_test.go |
| 205 | TestParseStackIPs_Valid | Parses valid stack with 3 IPs | stack_test.go |
| 206 | TestParseStackIPs_DepthZero | Zero depth → nil | stack_test.go |
| 207 | TestParseStackIPs_DepthTooLarge | Depth >64 → nil | stack_test.go |
| 208 | TestParseStackIPs_BufferTooShort | Short buffer → nil | stack_test.go |
| 209 | TestParseStackIPs_Truncated | Truncated buffer yields available IPs | stack_test.go |
| 210 | TestParseStackIPs_ZeroIPTerminates | Zero IP terminates stack early | stack_test.go |
| 211 | TestParseStackIPs_MaxDepth | Depth=64 accepted at boundary | stack_test.go |
| 212 | TestSourceString | Source enum → string (cuda, driver, host, io, tcp, net) | types_test.go |
| 213 | TestCUDAOpString | CUDA op names | types_test.go |
| 214 | TestHostOpString | Host op names (sched_switch, pod_restart, etc.) | types_test.go |
| 215 | TestDriverOpString | Driver op names (cuLaunchKernel, etc.) | types_test.go |
