"""Runnable PyTorch Lightning example with the Ingero annotation callback.

Trains a tiny model on synthetic data and attaches `IngeroCallback`, so
every training step and epoch is annotated into a live Ingero trace.

Run the agent first, in another shell, so the annotation socket exists:

    sudo ingero trace --record --annotate

Then run this script:

    python train_example.py

Afterwards, slice the recorded trace by step or epoch:

    ingero query --label step=10
    ingero explain --label epoch=2

If the agent is not running with `--annotate`, the callback degrades to
a no-op and the training run is unaffected.

Requires: pytorch-lightning, torch (see requirements.txt).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl

from ingero_lightning import IngeroCallback


class TinyModel(pl.LightningModule):
    """A single linear layer - enough to drive real training steps."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(8, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def make_loader(n: int = 256, batch_size: int = 32) -> DataLoader:
    x = torch.randn(n, 8)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def main() -> None:
    model = TinyModel()
    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[IngeroCallback(run_id="lightning-example")],
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, make_loader())


if __name__ == "__main__":
    main()
