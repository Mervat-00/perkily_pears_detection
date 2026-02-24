# train_yolo.py
# YOLOv11 segmentation training — optimized for RTX 3050 Ti (4GB VRAM)

import torch
import os
import json
import datetime
from ultralytics import YOLO

# ── GPU Check ─────────────────────────────────────────────────────────
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Config — Tuned for RTX 3050 Ti (4GB VRAM) ─────────────────────────
CONFIG = {
    # Model size options:
    # yolo11n-seg.pt → nano    (2.9M params)  fastest
    # yolo11s-seg.pt → small   (10.1M params) ✅ good for 4GB VRAM
    # yolo11m-seg.pt → medium  (22.4M params) needs ~6GB VRAM
    "model"        : "yolo11s-seg.pt",

    "data"         : "dataset_yolo/data.yaml",
    "epochs"       : 100,
    "patience"     : 20,       # early stopping
    "batch"        : 8,        # safe for 4GB VRAM
    "imgsz"        : 640,

    # Optimizer
    "optimizer"    : "SGD",
    "lr0"          : 0.01,
    "lrf"          : 0.01,
    "momentum"     : 0.937,
    "weight_decay" : 0.0005,

    # ── Color-safe augmentation for maturity detection ─────────────────
    "hsv_h"        : 0.0,    # ← OFF: hue shift destroys green→yellow signal
    "hsv_s"        : 0.0,    # ← OFF: saturation destroys ripeness colors
    "hsv_v"        : 0.3,    # ← ON:  brightness safe (same color, diff light)
    "flipud"       : 0.0,    # ← OFF: pears don't grow upside down
    "fliplr"       : 0.5,    # ← ON:  pear looks same mirrored
    "degrees"      : 10.0,   # small rotation — pears grow at angles
    "translate"    : 0.1,
    "scale"        : 0.3,
    "shear"        : 0.0,
    "perspective"  : 0.0,
    "mosaic"       : 0.5,    # combines 4 images — helps small dataset
    "mixup"        : 0.0,    # ← OFF: blends images, bad for color labels
    "copy_paste"   : 0.3,    # copies rare Middle-Ripe into other images

    # Hardware
    "device"       : 0,      # GPU 0
    "workers"      : 0,      # 0 for Windows
    "amp"          : True,   # mixed precision — saves VRAM

    # Output
    "project"      : "runs",
    "name"         : "pear_yolo11",
    "save"         : True,
    "save_period"  : 10,
    "plots"        : True,
    "verbose"      : True,
}


def save_training_info(config, results):
    log_file = "yolo_training_log.json"

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log = json.load(f)
    else:
        log = {"runs": []}

    run = {
        "run_id"    : len(log["runs"]) + 1,
        "timestamp" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model"     : config["model"],
        "config"    : {k: v for k, v in config.items()
                       if k not in ["device", "workers"]},
        "results"   : {
            "best_mAP50"    : float(results.results_dict.get("metrics/mAP50(B)",     0)),
            "best_mAP50_95" : float(results.results_dict.get("metrics/mAP50-95(B)",  0)),
            "seg_mAP50"     : float(results.results_dict.get("metrics/mAP50(M)",     0)),
            "seg_mAP50_95"  : float(results.results_dict.get("metrics/mAP50-95(M)",  0)),
            "precision"     : float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall"        : float(results.results_dict.get("metrics/recall(B)",    0)),
            "best_checkpoint": str(results.save_dir / "weights" / "best.pt")
        }
    }

    log["runs"].append(run)
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining info saved to {log_file}")
    return run


if __name__ == '__main__':

    if not os.path.exists(CONFIG["data"]):
        print("data.yaml not found! Run coco_to_yolo.py first.")
        exit()

    print(f"\nLoading model: {CONFIG['model']}")
    model = YOLO(CONFIG["model"])

    print(f"\nStarting training...")
    print(f"Epochs    : {CONFIG['epochs']}")
    print(f"Batch     : {CONFIG['batch']}")
    print(f"Image size: {CONFIG['imgsz']}")
    print(f"HSV hue   : {CONFIG['hsv_h']} ← disabled (protects color signal)")
    print(f"HSV sat   : {CONFIG['hsv_s']} ← disabled (protects color signal)")
    print(f"HSV val   : {CONFIG['hsv_v']} ← brightness only\n")

    # ── Train ──────────────────────────────────────────────────────────
    results = model.train(**CONFIG)

    # ── Best Validation Results (from training) ────────────────────────
    print("\n" + "=" * 60)
    print("BEST VALIDATION RESULTS  (from training)")
    print("=" * 60)
    print(f"Box mAP@50    : {results.results_dict.get('metrics/mAP50(B)',     0):.4f}")
    print(f"Box mAP@50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"Mask mAP@50   : {results.results_dict.get('metrics/mAP50(M)',     0):.4f}")
    print(f"Mask mAP@50-95: {results.results_dict.get('metrics/mAP50-95(M)', 0):.4f}")
    print(f"Precision     : {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall        : {results.results_dict.get('metrics/recall(B)',    0):.4f}")
    print(f"\nEpoch log (every epoch): {results.save_dir}/results.csv")
    print(f"Training curves        : {results.save_dir}/results.png")

    # ── Save to log ────────────────────────────────────────────────────
    run = save_training_info(CONFIG, results)

    # ── Test Set Results ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET RESULTS  (never seen during training)")
    print("=" * 60)
    best_model   = YOLO(f"{results.save_dir}/weights/best.pt")
    test_results = best_model.val(
        data   = CONFIG["data"],
        split  = "test",
        imgsz  = CONFIG["imgsz"],
        device = CONFIG["device"]
    )
    print(f"Box mAP@50    : {test_results.results_dict.get('metrics/mAP50(B)',     0):.4f}")
    print(f"Box mAP@50-95 : {test_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"Mask mAP@50   : {test_results.results_dict.get('metrics/mAP50(M)',     0):.4f}")
    print(f"Precision     : {test_results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall        : {test_results.results_dict.get('metrics/recall(B)',    0):.4f}")
    print(f"\nBest weights: {results.save_dir}/weights/best.pt")
