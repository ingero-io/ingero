#!/usr/bin/env python3
"""Train ResNet-50 on CIFAR-10 for a few epochs.

Exercises: cudaMalloc, cudaLaunchKernel, cudaMemcpy (H->D for data loading), cudaStreamSync
Pattern: Steady training loop — consistent kernel launches, periodic data transfers
VRAM: ~4GB | Time: ~5 minutes for 3 epochs
Expected Ingero output: Stable stats, regular cadence, good baseline for anomaly thresholds
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50


def main():
    parser = argparse.ArgumentParser(description="ResNet-50 CIFAR-10 training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"resnet50_cifar10: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}")
    print()

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    print(f"  {len(trainset)} training samples, {len(trainloader)} batches\n")

    # Model — ResNet-50 adapted for CIFAR-10 (32x32 images, 10 classes)
    model = resnet50(num_classes=10)
    # Replace first conv for 32x32 input (CIFAR-10 is small)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Training
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} [{batch_idx+1}/{len(trainloader)}] "
                      f"Loss: {running_loss/(batch_idx+1):.3f} Acc: {100.*correct/total:.1f}%")

        dt = time.time() - t0
        print(f"  Epoch {epoch+1} complete: {dt:.1f}s, "
              f"Loss: {running_loss/len(trainloader):.3f}, "
              f"Acc: {100.*correct/total:.1f}%\n")

    print("resnet50_cifar10 complete.")


if __name__ == "__main__":
    main()
