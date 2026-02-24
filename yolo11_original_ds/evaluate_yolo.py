# evaluate_yolo.py
# Evaluates YOLOv11 model and saves to the SAME evaluation_log.json
# as Mask R-CNN evaluate.py — so compare.py works for both models

import os
import json
import datetime
import numpy as np
import torch
import cv2
from ultralytics import YOLO

MODEL_PATH  = "runs/pear_yolo11/weights/best.pt"
VAL_IMAGES  = "dataset_yolo/valid/images"
VAL_LABELS  = "dataset_yolo/valid/labels"
LOG_FILE    = "evaluation_log.json"   # same file as evaluate.py

CATEGORIES = {
    0: "Middle-Ripe",
    1: "Ripe",
    2: "Unripe"
}

IOU_THRESHOLD   = 0.5
SCORE_THRESHOLD = 0.5


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def yolo_label_to_box(line, img_w, img_h):
    parts    = list(map(float, line.strip().split()))
    class_id = int(parts[0])
    if len(parts) == 5:
        cx, cy, w, h = parts[1:5]
        x1 = (cx - w / 2) * img_w;  y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w;  y2 = (cy + h / 2) * img_h
    else:
        coords = parts[1:]
        xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
        ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return class_id, [x1, y1, x2, y2]


def evaluate(model, images_dir, labels_dir, split_name):
    stats = {cat_id: {"tp": 0, "fp": 0, "fn": 0} for cat_id in CATEGORIES}

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    print(f"Evaluating on {len(image_files)} {split_name} images...")

    for img_file in image_files:
        img_path   = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        gt_boxes, gt_labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    if line.strip():
                        cat_id, box = yolo_label_to_box(line, img_w, img_h)
                        gt_boxes.append(box)
                        gt_labels.append(cat_id)

        results     = model.predict(img_path, conf=SCORE_THRESHOLD,
                                    iou=IOU_THRESHOLD, verbose=False,
                                    device=0 if torch.cuda.is_available() else "cpu")
        pred        = results[0]
        pred_boxes  = pred.boxes.xyxy.cpu().numpy()            if pred.boxes else np.array([])
        pred_labels = pred.boxes.cls.cpu().numpy().astype(int) if pred.boxes else np.array([])

        matched_gt = set()
        for pb, pl in zip(pred_boxes, pred_labels):
            if pl not in CATEGORIES:
                continue
            best_iou, best_idx = 0, -1
            for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if gl != pl or gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            if best_iou >= IOU_THRESHOLD and best_idx >= 0:
                stats[pl]["tp"] += 1
                matched_gt.add(best_idx)
            else:
                stats[pl]["fp"] += 1

        for gi, gl in enumerate(gt_labels):
            if gi not in matched_gt and gl in CATEGORIES:
                stats[gl]["fn"] += 1

    metrics = {}
    all_p, all_r, all_f1 = [], [], []

    for cat_id, cat_name in CATEGORIES.items():
        tp = stats[cat_id]["tp"]
        fp = stats[cat_id]["fp"]
        fn = stats[cat_id]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)

        metrics[cat_name] = {
            "precision": round(precision, 4),
            "recall"   : round(recall,    4),
            "f1"       : round(f1,        4),
            "tp": tp, "fp": fp, "fn": fn
        }
        all_p.append(precision)
        all_r.append(recall)
        all_f1.append(f1)

    metrics["average"] = {
        "precision": round(float(np.mean(all_p)),  4),
        "recall"   : round(float(np.mean(all_r)),  4),
        "f1"       : round(float(np.mean(all_f1)), 4),
    }
    return metrics


def save_run(model_path, metrics, split_name, notes=""):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            log = json.load(f)
    else:
        log = {"runs": []}

    run = {
        "run_id"    : len(log["runs"]) + 1,
        "timestamp" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": model_path,
        "notes"     : notes,
        "training_info": {
            "epochs_trained": _get_yolo_epochs(),
            "best_val_loss" : _get_yolo_val_loss(),
            "num_classes"   : len(CATEGORIES),
            "model_type"    : "YOLOv11-seg",
            "eval_split"    : split_name
        },
        "eval_settings": {
            "iou_threshold"  : IOU_THRESHOLD,
            "score_threshold": SCORE_THRESHOLD
        },
        "metrics": metrics
    }

    log["runs"].append(run)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nRun #{run['run_id']} saved to {LOG_FILE}")
    return run


def _get_yolo_epochs():
    csv = "runs/pear_yolo11/results.csv"
    if os.path.exists(csv):
        with open(csv) as f:
            return len(f.readlines()) - 1
    return "unknown"


def _get_yolo_val_loss():
    csv = "runs/pear_yolo11/results.csv"
    if os.path.exists(csv):
        try:
            import csv as csv_mod
            with open(csv) as f:
                reader = csv_mod.DictReader(f)
                rows   = list(reader)
            col = [k for k in rows[0].keys() if "val" in k.lower() and "box" in k.lower()]
            if col:
                losses = [float(r[col[0]]) for r in rows if r[col[0]].strip()]
                return round(min(losses), 4)
        except Exception:
            pass
    return "unknown"


def print_results(run):
    split = run['training_info'].get('eval_split', 'valid')
    print("\n" + "=" * 60)
    print(f"  Run #{run['run_id']}  |  {run['timestamp']}")
    print(f"  Checkpoint : {run['checkpoint']}")
    print(f"  Model type : {run['training_info'].get('model_type', 'unknown')}")
    print(f"  Eval split : {split}")
    print(f"  Epochs     : {run['training_info']['epochs_trained']}")
    print(f"  Val Loss   : {run['training_info']['best_val_loss']}")
    if run['notes']:
        print(f"  Notes      : {run['notes']}")
    print("=" * 60)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for class_name, m in run['metrics'].items():
        print(f"{class_name:<15} {m['precision']:>10.2%} "
              f"{m['recall']:>10.2%} {m['f1']:>10.2%}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=MODEL_PATH)
    parser.add_argument('--split',      default='valid', choices=['valid', 'train'])
    parser.add_argument('--notes',      default='')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Model not found: {args.checkpoint}")
        exit()

    images_dir = f"dataset_yolo/{args.split}/images"
    labels_dir = f"dataset_yolo/{args.split}/labels"

    print(f"Loading    : {args.checkpoint}")
    print(f"Evaluating : {args.split} split")
    print(f"Device     : {'GPU — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    model   = YOLO(args.checkpoint)
    metrics = evaluate(model, images_dir, labels_dir, split_name=args.split)
    run     = save_run(args.checkpoint, metrics, split_name=args.split, notes=args.notes)
    print_results(run)
