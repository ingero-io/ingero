#!/usr/bin/env python3
"""Reproduce pytorch#173382: torch.cuda.empty_cache() doesn't free all GPU memory.

Loads a model, runs inference, deletes tensors, calls empty_cache(),
and logs CUDA memory at each step. The gap between "reserved" and
"allocated" after empty_cache() shows the caching allocator holding
blocks that empty_cache() didn't release.

Usage:
    python3 tests/workloads/pathological/cuda_empty_cache_leak.py

Reference: https://github.com/pytorch/pytorch/issues/173382
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def log_mem(label):
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{label:30s}] allocated={alloc:8.1f} MB  reserved={reserved:8.1f} MB  gap={reserved - alloc:8.1f} MB")

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Phase 1: Load model
    log_mem("before model load")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    log_mem("after model load")

    # Phase 2: Run inference (3 rounds to build up allocator state)
    for i in range(3):
        messages = [{"role": "user", "content": f"Explain GPU memory management in {50 * (i+1)} words."}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        log_mem(f"round {i+1}: after tokenize")

        with torch.inference_mode():
            output_ids = model.generate(input_ids, max_new_tokens=200, do_sample=True, temperature=0.7)
        log_mem(f"round {i+1}: after generate")

        response = tokenizer.batch_decode(output_ids)[0]
        print(f"  Generated {output_ids.shape[1]} tokens")

        # Tyler's pattern: del + empty_cache
        del output_ids
        del input_ids
        torch.cuda.empty_cache()
        log_mem(f"round {i+1}: after del+empty_cache")
        print()

    # Phase 3: Final state
    log_mem("final (model still loaded)")

    # Phase 4: Delete model too
    del model
    torch.cuda.empty_cache()
    log_mem("after del model+empty_cache")

    # Phase 5: Force GC
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after gc.collect+empty_cache")

if __name__ == "__main__":
    main()
