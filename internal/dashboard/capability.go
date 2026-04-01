// Package dashboard provides an HTTPS dashboard server for Ingero.
//
// The dashboard serves an embedded single-page HTML application that
// visualizes GPU causal observability data from the SQLite event store.
// Metrics that Ingero does not collect are grayed out with tooltips
// naming the required external tool.
package dashboard

// Capability describes whether a metric is available from Ingero's
// eBPF instrumentation or requires an external tool.
type Capability struct {
	ID          string `json:"id"`
	Label       string `json:"label"`
	Available   bool   `json:"available"`
	Source      string `json:"source,omitempty"`       // how Ingero collects it
	RequiredTool string `json:"required_tool,omitempty"` // what external tool is needed
	Tooltip     string `json:"tooltip,omitempty"`       // shown when grayed out
}

// Capabilities returns the full metric availability manifest.
// The frontend uses this to gray out panels for uncollected metrics.
func Capabilities() []Capability {
	return []Capability{
		// Available — Ingero collects these.
		{ID: "cpu_util", Label: "CPU Utilization", Available: true, Source: "/proc/stat (sysinfo)"},
		{ID: "mem_util", Label: "Memory Utilization", Available: true, Source: "/proc/meminfo (sysinfo)"},
		{ID: "load_avg", Label: "Load Average", Available: true, Source: "/proc/loadavg (sysinfo)"},
		{ID: "swap", Label: "Swap Usage", Available: true, Source: "/proc/meminfo (sysinfo)"},
		{ID: "cuda_ops", Label: "CUDA Runtime Ops", Available: true, Source: "eBPF uprobes on libcudart.so"},
		{ID: "driver_ops", Label: "CUDA Driver Ops", Available: true, Source: "eBPF uprobes on libcuda.so"},
		{ID: "host_events", Label: "Host Kernel Events", Available: true, Source: "eBPF tracepoints (sched_switch, etc.)"},
		{ID: "block_io", Label: "Block I/O", Available: true, Source: "eBPF tracepoints (block_rq_issue/complete)"},
		{ID: "tcp_retransmit", Label: "TCP Retransmits", Available: true, Source: "eBPF tracepoint (tcp_retransmit_skb)"},
		{ID: "net_io", Label: "Network Socket I/O", Available: true, Source: "eBPF tracepoints (sendto/recvfrom)"},
		{ID: "causal_chains", Label: "Causal Chains", Available: true, Source: "4-layer correlate engine"},
		{ID: "stack_traces", Label: "Stack Traces", Available: true, Source: "eBPF + DWARF + Python frames"},
		{ID: "cuda_graph", Label: "CUDA Graph Lifecycle", Available: true, Source: "eBPF uprobes on libcudart.so (graph APIs)"},

		// Grayed out — not yet collected by Ingero.
		{ID: "gpu_util", Label: "GPU SM Utilization", Available: false, RequiredTool: "NVML", Tooltip: "Planned: NVML integration (GPU utilization polling)"},
		{ID: "hbm_occupancy", Label: "HBM Occupancy", Available: false, RequiredTool: "NVML", Tooltip: "Planned: NVML integration (memory occupancy polling)"},
		{ID: "sm_efficiency", Label: "SM Efficiency", Available: false, RequiredTool: "CUPTI", Tooltip: "Planned: CUPTI Profiler API integration"},
		{ID: "nvlink_util", Label: "NVLink Utilization", Available: false, RequiredTool: "DCGM", Tooltip: "Planned: DCGM integration (NVLink counters)"},
		{ID: "ib_util", Label: "InfiniBand Utilization", Available: false, RequiredTool: "RDMA", Tooltip: "Planned: RDMA counter integration"},
		{ID: "power_draw", Label: "Power Draw", Available: false, RequiredTool: "NVML", Tooltip: "Planned: NVML integration (power monitoring)"},
		{ID: "pcie_util", Label: "PCIe Utilization", Available: false, RequiredTool: "DCGM", Tooltip: "Planned: DCGM integration (PCIe counters)"},
		{ID: "stall_breakdown", Label: "SM Stall Classification", Available: false, RequiredTool: "CUPTI", Tooltip: "Planned: CUPTI Profiler API integration"},
		{ID: "tif_sif", Label: "TIF/SIF Metrics", Available: false, RequiredTool: "Enterprise", Tooltip: "Ingero Enterprise: multi-GPU coordination"},
		{ID: "nvtx_phases", Label: "NVTX Training Phases", Available: false, RequiredTool: "libnvToolsExt", Tooltip: "Planned: uprobes on libnvToolsExt.so"},
		{ID: "cluster_map", Label: "Multi-Node Cluster View", Available: false, RequiredTool: "Enterprise", Tooltip: "Ingero Enterprise: federated cross-node query"},
		{ID: "what_if", Label: "What-If Scenarios", Available: false, RequiredTool: "NVML/DCGM", Tooltip: "Planned: NVML/DCGM integration (GPU utilization + pipeline metrics)"},
		{ID: "pipeline_flow", Label: "Pipeline Flow", Available: false, RequiredTool: "NVML/DCGM", Tooltip: "Planned: NVML/DCGM integration (per-stage pipeline metrics)"},
		{ID: "cost_waste", Label: "Cost Waste", Available: false, RequiredTool: "NVML", Tooltip: "Planned: NVML integration (GPU idle-time cost calculation)"},
		{ID: "storage_io", Label: "Storage I/O Rate", Available: false, RequiredTool: "Block I/O", Tooltip: "Planned: per-device block I/O breakdown"},
	}
}
