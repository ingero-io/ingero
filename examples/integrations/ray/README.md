# Ray

Helpers that annotate a live Ingero eBPF trace per Ray task, so a
recorded trace can be sliced per task after the job.

## What it does

A Ray task runs in a worker process. On task start the helper writes one
NDJSON annotation, scoped to the worker PID, carrying:

- `task_id` - the Ray task id when available, otherwise the
  caller-supplied task name.
- `task_name` - the caller-supplied name.
- `run_id` - the Ray job id, when running inside Ray.
- any extra labels the caller passes.

The agent joins those labels to the eBPF event stream by process
incarnation and time window. `ingero query` and `ingero explain` then
slice the trace by task.

Two entry points:

- `annotate_task(...)` - a decorator, and the recommended path. Apply it
  under `@ray.remote`; the task is annotated automatically each time it
  runs, always inside the worker process, so the annotation is correctly
  scoped to the worker PID.
- `task_annotation(...)` - a context manager, for annotating from inside
  an existing task body. It MUST be opened inside a running Ray task,
  never on the driver: the annotation is scoped to `os.getpid()`, so
  opening it on the driver would mis-scope the annotation to the driver
  PID and the recorded trace would not slice correctly. When in doubt,
  use the `annotate_task` decorator, which cannot be misused this way.

## Requirements

The agent must be running with the annotation ingest socket enabled:

    sudo ingero trace --record --annotate

If it is not, the helper degrades to a no-op: it logs one line and every
annotation call returns without sending. It never crashes or slows a
task.

## Run the example

    pip install -r requirements.txt
    python run_example.py

Then slice the recorded trace:

    ingero query --label task_id=matmul-3
    ingero explain --label task_name=matmul

## Use in your own Ray code

    import ray
    from ingero_ray import annotate_task, task_annotation

    @ray.remote
    @annotate_task(task_name="preprocess")
    def preprocess(shard):
        ...

    @ray.remote
    def train(shard):
        with task_annotation(task_name="train", labels={"shard": str(shard)}):
            ...

## Files

| File | Purpose |
|------|---------|
| `ingero_annotate.py` | Framework-agnostic NDJSON socket writer. No Ray dependency. |
| `ingero_ray.py` | The `annotate_task` decorator and `task_annotation` context manager. |
| `run_example.py` | A runnable end-to-end example. |
| `test_*.py`, `conftest.py` | pytest suite (fake socket server, no agent needed). |

## Design notes

Each Ray worker process opens the annotation socket once and reuses it
for every task it runs; no `ingero annotate` subprocess is spawned per
task. The cached writer is keyed by PID so a forked worker never
inherits a parent's socket fd. The protocol and socket logic live in
`ingero_annotate.py`, which is pure stdlib so it stays unit-testable
without installing Ray.

The annotation wire protocol and its validation limits mirror
`pkg/contract/annotate.go` in the agent. This example depends on the
agent's socket contract; the agent never imports this code.

## Tests

    pip install pytest
    pytest

The suite uses a fake Unix-domain socket server in place of a running
agent, and covers the NDJSON encoding, the protocol-limit conformance,
the decorator and context-manager paths, and the no-op path when the
socket is absent.
