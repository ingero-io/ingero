#!/usr/bin/env python3
"""Run Stable Diffusion v1.5 inference (image generation).

Exercises: cudaMalloc (large UNet/VAE buffers), cudaLaunchKernel (diffusion steps),
           cudaMemcpy (latent transfers), cudaStreamSync
Pattern: Mixed compute — UNet denoising + VAE decode + CLIP encoding, interesting allocation pattern
VRAM: ~12GB with fp16 | Time: ~2 minutes for 4 images
Expected Ingero output: Complex allocation pattern, varying kernel sizes, periodic sync
"""

import argparse
import time
import torch
from diffusers import StableDiffusionPipeline


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion inference")
    parser.add_argument("--num-images", type=int, default=4, help="Images to generate")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion steps per image")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Model ID")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"sd_inference: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print(f"  model={args.model}, images={args.num_images}, steps={args.steps}")
    print()

    # Load pipeline
    print("Loading Stable Diffusion pipeline (fp16)...")
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    print(f"  Loaded in {time.time() - t0:.1f}s\n")

    # Prompts
    prompts = [
        "A photo of an astronaut riding a horse on the moon",
        "A cyberpunk city street at night, neon lights, rain",
        "A golden retriever wearing a lab coat in a science laboratory",
        "An oil painting of a mountain landscape at sunset",
        "A macro photo of a dewdrop on a leaf reflecting a forest",
        "A steampunk mechanical owl perched on gears",
        "A cozy cabin in the woods during snowfall, warm light",
        "An underwater coral reef with tropical fish, sunlight rays",
    ]

    # Generate
    for i in range(args.num_images):
        prompt = prompts[i % len(prompts)]
        print(f"  Image {i+1}/{args.num_images}: \"{prompt[:60]}...\"")

        t0 = time.time()
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=args.steps,
                guidance_scale=7.5,
            ).images[0]
        dt = time.time() - t0
        print(f"    Generated in {dt:.1f}s ({args.steps/dt:.1f} steps/sec)")

        # Save to /tmp (optional, mainly exercises the full pipeline)
        image.save(f"/tmp/ingero_sd_{i}.png")

    print(f"\nsd_inference complete. {args.num_images} images generated.")


if __name__ == "__main__":
    main()
