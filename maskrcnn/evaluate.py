# evaluate.py
# Evaluates Mask R-CNN model — saves results to evaluation_log.json
# Run after training: python evaluate.py --checkpoint best_pear_model.pth --notes "25 epochs"

import torch
import json
import os
import datetime
import numpy as np
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from dataset import CocoDataset, collate_fn
from model import get_model

# ── Config ────────────────────────────────────────────────────────────
CATEGORIES = {
    1: 'Middle-Ripe',
    2: 'Ripe',
    3: 'Unripe'
}

NUM_CLASSES     = 4
IOU_THRESHOLD   = 0.5
SCORE_THRESHOLD = 0.5
LOG_FILE        = "evaluation_log.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model      = get_model(NUM_CLASSES)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate(model, data_loader):
    results = {cat_id: {'tp': 0, 'fp': 0, 'fn': 0} for cat_id in CATEGORIES}

    with torch.no_grad():
        for images, targets in data_loader:
            images  = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                keep        = output['scores'] >= SCORE_THRESHOLD
                pred_boxes  = output['boxes'][keep]
                pred_labels = output['labels'][keep]

                gt_boxes  = target['boxes'].to(device)
                gt_labels = target['labels'].to(device)
                matched_gt = set()

                for pb, pl in zip(pred_boxes, pred_labels):
                    pl = pl.item()
                    if pl not in CATEGORIES:
                        continue
                    if len(gt_boxes) == 0:
                        results[pl]['fp'] += 1
                        continue

                    ious     = box_iou(pb.unsqueeze(0), gt_boxes)[0]
                    best_iou = ious.max().item()
                    best_idx = ious.argmax().item()

                    if (best_iou >= IOU_THRESHOLD and
                            best_idx not in matched_gt and
                            gt_labels[best_idx].item() == pl):
                        results[pl]['tp'] += 1
                        matched_gt.add(best_idx)
                    else:
                        results[pl]['fp'] += 1

                for gi, gl in enumerate(gt_labels):
                    gl = gl.item()
                    if gl in CATEGORIES and gi not in matched_gt:
                        results[gl]['fn'] += 1

    metrics = {}
    all_p, all_r, all_f1 = [], [], []

    for cat_id, cat_name in CATEGORIES.items():
        tp = results[cat_id]['tp']
        fp = results[cat_id]['fp']
        fn = results[cat_id]['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)

        metrics[cat_name] = {
            'precision': round(precision, 4),
            'recall':    round(recall,    4),
            'f1':        round(f1,        4),
            'tp': tp, 'fp': fp, 'fn': fn
        }
        all_p.append(precision)
        all_r.append(recall)
        all_f1.append(f1)

    metrics['average'] = {
        'precision': round(np.mean(all_p),  4),
        'recall':    round(np.mean(all_r),  4),
        'f1':        round(np.mean(all_f1), 4),
    }

    return metrics


def save_run(checkpoint_path, metrics, checkpoint, notes=""):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            log = json.load(f)
    else:
        log = {"runs": []}

    run = {
        "run_id":    len(log["runs"]) + 1,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": checkpoint_path,
        "notes":     notes,
        "training_info": {
            "epochs_trained": checkpoint.get('epoch', 'unknown') + 1,
            "best_val_loss":  round(checkpoint.get('val_loss', 0), 4),
            "num_classes":    checkpoint.get('num_classes', NUM_CLASSES),
            "model_type":     "MaskRCNN-ResNet50-FPN-V2"
        },
        "eval_settings": {
            "iou_threshold":   IOU_THRESHOLD,
            "score_threshold": SCORE_THRESHOLD
        },
        "metrics": metrics
    }

    log["runs"].append(run)
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nRun #{run['run_id']} saved to {LOG_FILE}")
    return run


def print_results(run):
    print("\n" + "=" * 60)
    print(f"  Run #{run['run_id']}  |  {run['timestamp']}")
    print(f"  Checkpoint : {run['checkpoint']}")
    print(f"  Model type : {run['training_info'].get('model_type', 'unknown')}")
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
    parser.add_argument('--checkpoint', default='best_pear_model.pth')
    parser.add_argument('--notes',      default='')
    args = parser.parse_args()

    print(f"Loading: {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint)

    test_dataset = CocoDataset(
        "dataset/test/images",
        "dataset/test/_annotations.coco.json",
        train=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    print(f"Evaluating on {len(test_dataset)} test images...")

    metrics = evaluate(model, test_loader)
    run     = save_run(args.checkpoint, metrics, checkpoint, notes=args.notes)
    print_results(run)
