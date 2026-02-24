#!/usr/bin/env python3
"""Fine-tune BERT-base on MRPC (GLUE benchmark).

Exercises: cudaMalloc, cudaLaunchKernel (attention + FFN), cudaMemcpy, cudaStreamSync
Pattern: Fast training loop with high kernel launch rate (small sequences)
VRAM: ~4GB | Time: ~5 minutes for 3 epochs
Expected Ingero output: High cudaLaunchKernel rate, fast cadence, good for launch latency stats
"""

import argparse
import time
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="BERT fine-tuning on MRPC")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"bert_glue: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}")
    print()

    # Model and tokenizer
    print("Loading BERT-base...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M\n")

    # Dataset — MRPC (paraphrase detection, small and fast)
    print("Loading MRPC dataset...")
    dataset = load_dataset("glue", "mrpc")
    train_dataset = dataset["train"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence1"], examples["sentence2"],
            truncation=True, max_length=args.max_length
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=data_collator, num_workers=2, pin_memory=True
    )
    print(f"  {len(train_dataset)} samples, {len(dataloader)} batches\n")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    model.train()

    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        dt = time.time() - t0
        print(f"  Epoch {epoch+1}/{args.epochs}: {dt:.1f}s, "
              f"Loss: {running_loss/len(dataloader):.3f}, "
              f"Acc: {100.*correct/total:.1f}%")

    print(f"\nbert_glue complete.")


if __name__ == "__main__":
    main()
