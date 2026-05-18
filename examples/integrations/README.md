# Integrations

Copy-paste manifests for running the Ingero agent alongside common
parts of the GPU and ML stack. Each subdirectory is self-contained.

| Directory | What it shows |
|-----------|---------------|
| `vllm/` | The agent tracing a vLLM inference server in the same host |
| `pytorch-lightning/` | A Lightning callback that annotates a live trace with the training step and epoch |
| `ray/` | Decorator and context-manager helpers that annotate a live trace per Ray task |

The agent stays the eBPF data producer in every case. These examples
add no agent dependencies; they show placement and configuration only.

The `pytorch-lightning/` and `ray/` examples use the agent's external
annotation socket (agent v0.17.0): an external workload writes
step / epoch / task labels into a recorded trace, which `ingero query`
and `ingero explain` join to the eBPF event stream. They require the
agent to run with `trace --record --annotate`. The annotation wire
protocol they conform to is owned by the agent in
`pkg/contract/annotate.go`; these examples never import agent Go code.
