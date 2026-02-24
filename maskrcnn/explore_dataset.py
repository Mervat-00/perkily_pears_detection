# explore_dataset.py
# Run this FIRST before training — understand your data

import json
import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

TRAIN_ANN = "dataset/train/_annotations.coco.json"
TRAIN_IMG = "dataset/train/images"

coco = COCO(TRAIN_ANN)

# ── Check categories ──────────────────────────────────────────────────
print("Categories:")
for cat in coco.dataset['categories']:
    print(f"  id={cat['id']}  name={cat['name']}")

# ── Dataset size ──────────────────────────────────────────────────────
print(f"\nTotal images     : {len(coco.imgs)}")
print(f"Total annotations: {len(coco.anns)}")

# ── Count per category ────────────────────────────────────────────────
print("\nAnnotations per category:")
for cat in coco.dataset['categories']:
    ann_ids = coco.getAnnIds(catIds=[cat['id']])
    print(f"  {cat['name']}: {len(ann_ids)}")

# ── Visualize sample ──────────────────────────────────────────────────
img_id   = list(coco.imgs.keys())[0]
img_info = coco.imgs[img_id]
img      = Image.open(os.path.join(TRAIN_IMG, img_info['file_name']))

ann_ids = coco.getAnnIds(imgIds=img_id)
anns    = coco.loadAnns(ann_ids)

fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(img)

colors = ['red', 'green', 'blue', 'yellow', 'orange']
for ann in anns:
    x, y, w, h   = ann['bbox']
    cat_name      = coco.cats[ann['category_id']]['name']
    color         = colors[ann['category_id'] % len(colors)]
    rect          = patches.Rectangle((x, y), w, h,
                                       linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y - 5, cat_name, color=color, fontsize=12, fontweight='bold')

    mask          = coco.annToMask(ann)
    colored_mask  = np.zeros((*mask.shape, 4))
    colored_mask[mask == 1] = [1, 0, 0, 0.4]
    ax.imshow(colored_mask)

plt.title(img_info['file_name'])
plt.axis('off')
plt.savefig('sample_annotation.png')
plt.show()
print("Saved sample_annotation.png")
