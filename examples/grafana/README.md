# Ingero Grafana dashboards

The reference dashboards for Ingero have moved out of this
repository. There are two ways to install them:

## Option A: Grafana plugin (recommended)

The [Ingero Grafana app
plugin](https://github.com/ingero-io/ingero-grafana-app) bundles
all 10 reference dashboards (4 single-host + 5 cluster + 1
pipeline-health) and an Ingero datasource for querying Echo
directly. One install, all dashboards auto-imported.

The plugin is published to the Grafana Plugin Catalog at
https://grafana.com/grafana/plugins/ingero-gpu-app (link active
once the plugin ships).

## Option B: Paste from grafana.com

Each dashboard has a community-catalog entry that can be imported
by ID into any Grafana:

| Dashboard | grafana.com ID |
|---|---|
| NVIDIA GPU Trace Overview (single host) | `25277` |
| CUDA Op Profiler (single host) | `25278` |
| GPU Data Movement (single host) | `25280` |
| GPU Memory and Throttle (single host) | `25281` |
| NVIDIA GPU Cluster Overview | `25271` |
| NCCL Stragglers | `25273` |
| GPU Memcpy Bandwidth | `25274` |
| GPU Memory Fragmentation | `25275` |
| Per-Node GPU Drill-Down | `25276` |

In Grafana, **Dashboards** > **New** > **Import**, paste the ID,
pick your Prometheus datasource. Same as any other community
dashboard.

All catalog entries: https://grafana.com/orgs/ingero
