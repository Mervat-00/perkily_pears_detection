# train.py
# Mask R-CNN training — optimized for RTX 3050 Ti Laptop (4GB VRAM)

import torch
import json
import os
import time
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from dataset import CocoDataset, collate_fn
from model import get_model

# ── Ampere GPU optimizations (RTX 3050 Ti) ────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True   # free ~30% speedup on Ampere
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True   # auto-tune kernels for your GPU

# ── Config ────────────────────────────────────────────────────────────
TRAIN_IMG          = "dataset/train/images"
TRAIN_ANN          = "dataset/train/_annotations.coco.json"
VAL_IMG            = "dataset/valid/images"
VAL_ANN            = "dataset/valid/_annotations.coco.json"

NUM_CLASSES        = 4     # background + Middle-Ripe + Ripe + Unripe
NUM_EPOCHS         = 25
BATCH_SIZE         = 2     # max safe for 4GB VRAM
ACCUMULATION_STEPS = 4     # simulates batch size of 8 without extra VRAM
LR                 = 0.005
SAVE_PATH          = "best_pear_model.pth"


def print_vram():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved()  / 1e9
    total     = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {allocated:.2f}GB allocated | "
          f"{reserved:.2f}GB reserved | "
          f"{total:.2f}GB total")


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("  Backbone frozen")


def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("  Backbone unfrozen")


if __name__ == '__main__':

    device = torch.device("cuda")
    print(f"Training on : {torch.cuda.get_device_name(0)}")
    print(f"VRAM total  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Batch size  : {BATCH_SIZE} (effective {BATCH_SIZE * ACCUMULATION_STEPS} with accumulation)")

    # Datasets
    train_dataset = CocoDataset(TRAIN_IMG, TRAIN_ANN, train=True)
    val_dataset   = CocoDataset(VAL_IMG,   VAL_ANN,   train=False)
    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # DataLoaders — num_workers=0 required on Windows
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    # Model
    model = get_model(NUM_CLASSES).to(device)
    freeze_backbone(model)

    # Optimizer — only train unfrozen parameters
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    scaler = GradScaler()

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # Unfreeze backbone after epoch 5 for full fine-tuning
        if epoch == 5:
            unfreeze_backbone(model)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=LR / 10, momentum=0.9, weight_decay=0.0005
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - 5, eta_min=1e-6
            )
            print("  Switched to full fine-tuning mode")

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for i, (images, targets) in enumerate(train_loader):
            images  = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True)
                        for k, v in t.items()} for t in targets]

            with autocast(device_type="cuda"):
                loss_dict = model(images, targets)
                losses    = sum(loss for loss in loss_dict.values())
                losses    = losses / ACCUMULATION_STEPS

            scaler.scale(losses).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += losses.item() * ACCUMULATION_STEPS

            if (i + 1) % 20 == 0:
                print(f"  Batch [{i+1:>3}/{len(train_loader)}] "
                      f"Loss: {losses.item() * ACCUMULATION_STEPS:.4f}", end=" | ")
                print_vram()

        torch.cuda.empty_cache()

        # ── Validate ────────────────────────────────────────────────────
        model.train()  # Mask R-CNN needs train mode to compute val loss
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images  = [img.to(device, non_blocking=True) for img in images]
                targets = [{k: v.to(device, non_blocking=True)
                            for k, v in t.items()} for t in targets]
                with autocast(device_type="cuda"):
                    loss_dict = model(images, targets)
                    val_loss += sum(loss for loss in loss_dict.values()).item()

        torch.cuda.empty_cache()

        avg_train  = train_loss / len(train_loader)
        avg_val    = val_loss   / len(val_loader)
        epoch_time = time.time() - epoch_start
        remaining  = epoch_time * (NUM_EPOCHS - epoch - 1)
        h, rem     = divmod(remaining, 3600)
        m, s       = divmod(rem, 60)

        scheduler.step()

        print(f"\nEpoch [{epoch+1:>2}/{NUM_EPOCHS}] "
              f"Train Loss: {avg_train:.4f}  "
              f"Val Loss: {avg_val:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Epoch time: {epoch_time/60:.1f} min | "
              f"Remaining: {int(h)}h {int(m)}m")
        print_vram()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch':       epoch,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'val_loss':    best_val_loss,
                'num_classes': NUM_CLASSES
            }, SAVE_PATH)
            print(f"  ✅ Best model saved — Val Loss: {best_val_loss:.4f}\n")

    print("Training complete! Model saved as", SAVE_PATH)
