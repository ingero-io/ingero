"""
Reproduce unslothai/unsloth#3943
Training starts quick (~2s/step) then slows dramatically (~45s/step) after ~15 iterations.
GPU utilization oscillates 99% -> 10%.

Uses Qwen2.5-Coder-7B-Instruct 4-bit with LoRA on synthetic instruction data.
Matches reporter's config: batch=2, grad_accum=4, seq=2048, dropout=0.05.
"""

import time
import torch
import os

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

from unsloth import FastLanguageModel, is_bfloat16_supported

max_seq_length = 2048
dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
load_in_4bit = True

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print("Setting up LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,  # Reporter uses 0.05
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Generate synthetic instruction dataset (avoids needing reporter's private data)
print("Generating synthetic dataset...")
from datasets import Dataset
import random
random.seed(42)

instructions = [
    "Explain the following concept:",
    "Write a function that:",
    "Debug the following code:",
    "Optimize this algorithm:",
    "Describe how to implement:",
    "What is the difference between:",
    "Create a class that:",
    "Write unit tests for:",
    "Refactor this code:",
    "Design a data structure for:",
]

topics = [
    "binary search tree", "hash map", "graph traversal", "dynamic programming",
    "linked list reversal", "sorting algorithm", "recursion", "memoization",
    "thread safety", "memory management", "garbage collection", "API design",
    "database indexing", "caching strategy", "load balancing", "microservices",
    "authentication", "encryption", "compression", "serialization",
]

def generate_sample():
    inst = random.choice(instructions)
    topic = random.choice(topics)
    # Generate varying length outputs to simulate real training data
    output_len = random.randint(100, 500)
    output_words = " ".join(random.choices(
        ["the", "a", "function", "returns", "value", "class", "method",
         "algorithm", "data", "structure", "implements", "efficient",
         "complexity", "time", "space", "memory", "process", "thread",
         "async", "await", "error", "handle", "exception", "null",
         "pointer", "reference", "object", "instance", "interface"],
        k=output_len
    ))

    messages = [
        {"role": "system", "content": "You are an expert programming assistant."},
        {"role": "user", "content": f"{inst} {topic}"},
        {"role": "assistant", "content": output_words},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

# Generate 2000 samples (enough for ~250 steps with batch=2, grad_accum=4)
samples = [generate_sample() for _ in range(2000)]
train_dataset = Dataset.from_list(samples[:1800])
eval_dataset = Dataset.from_list(samples[1800:])

print(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

# Setup trainer matching reporter's config
from trl import SFTTrainer
from transformers import TrainingArguments

print("Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,  # Just enough to observe the slowdown pattern
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="no",  # No checkpoint saves during repro
        eval_strategy="no",  # No eval during repro
        report_to="none",
    ),
)

print("\n" + "=" * 60)
print("STARTING TRAINING — watching for step time degradation")
print("=" * 60)
print()

start = time.perf_counter()
trainer.train()
elapsed = time.perf_counter() - start
print(f"\nTotal training time: {elapsed:.1f}s for 100 steps")
print(f"Average: {elapsed/100:.1f}s/step")
