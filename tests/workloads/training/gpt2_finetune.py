#!/usr/bin/env python3
"""Fine-tune GPT-2 (124M) on WikiText-2.

Exercises: cudaMalloc (large attention buffers), cudaLaunchKernel (transformer ops),
           cudaMemcpy (tokenized data), cudaStreamSync
Pattern: Transformer training — large allocations, attention kernels, gradient accumulation
VRAM: ~8GB | Time: ~10 minutes for 1 epoch
Expected Ingero output: Higher cudaMalloc sizes, more complex kernel launch patterns
"""

import argparse
import time
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset


def tokenize_dataset(dataset, tokenizer, max_length=256):
    """Tokenize the dataset and chunk into fixed-length sequences."""
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")
    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser(description="GPT-2 fine-tuning on WikiText-2")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Sequence length")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--max-steps", type=int, default=500, help="Max training steps (0=full epoch)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"gpt2_finetune: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, seq_len={args.max_length}")
    print()

    # Model and tokenizer
    print("Loading GPT-2 (124M)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M\n")

    # Dataset
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Filter empty lines
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    tokenized = tokenize_dataset(dataset, tokenizer, args.max_length)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized, batch_size=args.batch_size, shuffle=True,
        collate_fn=data_collator, num_workers=2, pin_memory=True
    )
    print(f"  {len(tokenized)} samples, {len(dataloader)} batches\n")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    model.train()

    for epoch in range(args.epochs):
        running_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(dataloader):
            if args.max_steps > 0 and step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            if (step + 1) % 50 == 0:
                avg_loss = running_loss / (step + 1)
                elapsed = time.time() - t0
                samples_sec = (step + 1) * args.batch_size / elapsed
                print(f"  Epoch {epoch+1} [{step+1}] Loss: {avg_loss:.3f} "
                      f"({samples_sec:.0f} samples/sec)")

        steps_done = min(step + 1, args.max_steps) if args.max_steps > 0 else step + 1
        dt = time.time() - t0
        print(f"  Epoch {epoch+1} complete: {dt:.1f}s, {steps_done} steps, "
              f"Loss: {running_loss/steps_done:.3f}\n")

    print("gpt2_finetune complete.")


if __name__ == "__main__":
    main()
