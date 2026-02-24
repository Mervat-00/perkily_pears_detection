# dataset.py
# Loads COCO segmentation dataset for Mask R-CNN training
# Color-safe augmentation — protects green/yellow/orange ripeness signals

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

# ── ImageNet normalization — must match pretrained ResNet50 ───────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class PearAugmentation:
    """
    Color-safe augmentation for pear maturity detection.
    Protects hue/saturation — changing green→yellow would corrupt labels.
    """
    def __init__(self, train=True):
        self.train = train

    def __call__(self, image):
        if not self.train:
            return image

        # Brightness only — same color, different lighting
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))

        # Slight contrast — doesn't change hue
        if random.random() < 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        # Horizontal flip — pear looks same mirrored
        if random.random() < 0.5:
            image = TF.hflip(image)

        # Small rotation — pears grow at angles
        if random.random() < 0.3:
            image = TF.rotate(image, random.uniform(-15, 15))

        # ❌ Never apply: hue shift, saturation, grayscale
        # These destroy the color signal used to determine ripeness

        return image


class CocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, train=True):
        self.image_dir  = image_dir
        self.train      = train
        self.coco       = COCO(annotation_file)
        self.image_ids  = list(self.coco.imgs.keys())

        # Skip id=0 — Roboflow dataset name, not a real class
        cats             = [c for c in self.coco.dataset['categories'] if c['id'] != 0]
        self.cat_id_map  = {cat['id']: i + 1 for i, cat in enumerate(cats)}

        self.augment   = PearAugmentation(train=train)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        print("Class mapping:")
        for orig_id, new_id in self.cat_id_map.items():
            name = self.coco.cats[orig_id]['name']
            print(f"  {name} (id={orig_id}) → label {new_id}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id   = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image      = Image.open(image_path).convert("RGB")

        ann_ids     = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        masks, boxes, labels, areas = [], [], [], []

        for ann in annotations:
            if ann['category_id'] == 0:
                continue
            if ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
                continue

            mask = self.coco.annToMask(ann)
            masks.append(mask)

            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_map[ann['category_id']])
            areas.append(ann['area'])

        # Apply color-safe augmentation
        image = self.augment(image)
        image = self.to_tensor(image)
        image = self.normalize(image)

        if len(masks) == 0:
            h, w   = image.shape[1], image.shape[2]
            masks  = torch.zeros((0, h, w),  dtype=torch.uint8)
            boxes  = torch.zeros((0, 4),     dtype=torch.float32)
            labels = torch.zeros((0,),        dtype=torch.int64)
            areas  = torch.zeros((0,),        dtype=torch.float32)
        else:
            masks  = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            boxes  = torch.as_tensor(boxes,           dtype=torch.float32)
            labels = torch.as_tensor(labels,          dtype=torch.int64)
            areas  = torch.as_tensor(areas,           dtype=torch.float32)

        target = {
            "masks":    masks,
            "boxes":    boxes,
            "labels":   labels,
            "area":     areas,
            "image_id": torch.tensor([image_id]),
            "iscrowd":  torch.zeros(len(labels), dtype=torch.int64)
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
