# inference.py
# Run Mask R-CNN model on new pear images

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import json
import os
from collections import Counter

from model import get_model

# ── Category mapping ──────────────────────────────────────────────────
CATEGORIES = {
    0: 'background',
    1: 'Middle-Ripe',
    2: 'Ripe',
    3: 'Unripe'
}

CLASS_COLORS = {
    0: 'white',
    1: 'orange',   # Middle-Ripe
    2: 'green',    # Ripe
    3: 'red'       # Unripe
}

NUM_CLASSES = 4

# ── Transforms — must match training preprocessing ────────────────────
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ── Load model ────────────────────────────────────────────────────────
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: "
      f"{'GPU — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

checkpoint = torch.load("best_pear_model.pth", map_location=device)
model      = get_model(NUM_CLASSES)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()
print(f"Model loaded — trained for {checkpoint['epoch'] + 1} epochs | "
      f"Best Val Loss: {checkpoint['val_loss']:.4f}")


def predict(image_path, threshold=0.5, save_path="prediction.png"):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None, None, None

    image      = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    pred    = predictions[0]
    boxes   = pred['boxes'].cpu().numpy()
    labels  = pred['labels'].cpu().numpy()
    scores  = pred['scores'].cpu().numpy()
    masks   = pred['masks'].cpu().numpy()

    # Filter by confidence threshold
    keep   = scores >= threshold
    boxes  = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks  = masks[keep]

    print(f"\nDetected {len(boxes)} objects (threshold={threshold}):")
    for label, score in zip(labels, scores):
        print(f"  {CATEGORIES.get(label, 'unknown')} — {score:.2%} confidence")

    # Visualize
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, label, score, mask in zip(boxes, labels, scores, masks):
        color    = CLASS_COLORS.get(label, 'white')
        cat_name = CATEGORIES.get(label, 'unknown')

        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5, f"{cat_name} {score:.2%}",
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor='none')
        )

        # Segmentation mask overlay
        mask_bin = mask[0] > 0.5
        colored  = np.zeros((*mask_bin.shape, 4))
        rgba     = plt.cm.colors.to_rgba(color)
        colored[mask_bin] = [rgba[0], rgba[1], rgba[2], 0.45]
        ax.imshow(colored)

    # Summary box
    counts  = Counter(CATEGORIES.get(l, 'unknown') for l in labels)
    summary = "\n".join(f"{name}: {count}" for name, count in counts.items())
    ax.text(5, 5, summary, color='white', fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='black', alpha=0.6, pad=4, edgecolor='none'))

    plt.title(
        f"{os.path.basename(image_path)}  |  "
        f"{len(boxes)} detections  |  threshold={threshold}",
        fontsize=12
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"\nSaved → {save_path}")

    return boxes, labels, scores


def predict_folder(folder_path, threshold=0.5, output_dir="predictions"):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    print(f"Found {len(image_files)} images in {folder_path}")
    for filename in image_files:
        predict(
            image_path = os.path.join(folder_path, filename),
            threshold  = threshold,
            save_path  = os.path.join(output_dir, f"pred_{filename}")
        )


if __name__ == "__main__":
    # Single image
    predict(
        image_path = "dataset/test/images/your_image.jpg",
        threshold  = 0.5,
        save_path  = "prediction.png"
    )

    # Entire test folder (uncomment to use)
    # predict_folder("dataset/test/images", threshold=0.5)
