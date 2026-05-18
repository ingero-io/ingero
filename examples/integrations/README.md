# Integrations

Copy-paste manifests for running the Ingero agent alongside common
parts of the GPU and ML stack. Each subdirectory is self-contained.

| Directory | What it shows |
|-----------|---------------|
| `vllm/` | The agent tracing a vLLM inference server in the same host |

The agent stays the eBPF data producer in every case. These examples
add no agent dependencies; they show placement and configuration only.
