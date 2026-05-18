# PyTorch Lightning

A Lightning callback that annotates a live Ingero eBPF trace with the
training step and epoch, so a recorded trace can be sliced per step /
per epoch after the run.

## What it does

`IngeroCallback` connects once to the agent's annotation socket and, on
every training-step and epoch boundary, writes one NDJSON annotation
scoped to the training process PID:

- `on_train_batch_start` emits `step` and `epoch` labels.
- `on_train_epoch_start` / `on_train_epoch_end` emit `epoch` (the end
  marker also carries `phase=epoch_end`).
- An optional `run_id` is attached to every annotation.

The agent joins those labels to the eBPF event stream by process
incarnation and time window. `ingero query` and `ingero explain` then
slice the trace by step or epoch.

## Requirements

The agent must be running with the annotation ingest socket enabled:

    sudo ingero trace --record --annotate

If it is not, the callback degrades to a no-op: it logs one line and
every annotation call returns without sending. It never crashes or slows
a training run.

## Rate cap

The agent enforces a per-connection annotation rate cap
(`AnnotationConnRateLimit`, see `pkg/contract/annotate.go`): a fixed
number of annotations per connection per window. A training loop that
emits step boundaries faster than that cap will have some step labels
dropped server-side - the connection and the agent stay healthy, but
those steps will not appear in the sliced trace. For very fast loops,
annotate at a coarser boundary (for example, every Nth step or per
epoch) so every emitted label is recorded.

## Run the example

    pip install -r requirements.txt
    python train_example.py

Then slice the recorded trace:

    ingero query --label step=10
    ingero explain --label epoch=2

## Use in your own training code

    from ingero_lightning import IngeroCallback

    trainer = pl.Trainer(callbacks=[IngeroCallback(run_id="my-run")])
    trainer.fit(model, dataloader)

## Files

| File | Purpose |
|------|---------|
| `ingero_annotate.py` | Framework-agnostic NDJSON socket writer. No PyTorch dependency. |
| `ingero_lightning.py` | The `IngeroCallback` Lightning adapter. |
| `train_example.py` | A runnable end-to-end example. |
| `test_*.py`, `conftest.py` | pytest suite (fake socket server, no agent needed). |

## Design notes

The socket connection is opened once in `setup` and reused for every
step; no `ingero annotate` subprocess is spawned per step. The protocol
and socket logic live in `ingero_annotate.py`, which is pure stdlib so
it stays unit-testable without installing PyTorch or Lightning.

The annotation wire protocol and its validation limits mirror
`pkg/contract/annotate.go` in the agent. This example depends on the
agent's socket contract; the agent never imports this code.

## Tests

    pip install pytest
    pytest

The suite uses a fake Unix-domain socket server in place of a running
agent, and covers the NDJSON encoding, the protocol-limit conformance,
the connected write path, and the no-op path when the socket is absent.
