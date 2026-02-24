# compare_models.py
# Compares YOLOv11 training results vs Mask R-CNN evaluation results

import json
import os

MASKRCNN_LOG = "evaluation_log.json"
YOLO_LOG     = "yolo_training_log.json"


def compare():
    print("\n" + "=" * 70)
    print("  YOLOv11  vs  Mask R-CNN  — MODEL COMPARISON")
    print("=" * 70)

    # Load Mask R-CNN results
    if os.path.exists(MASKRCNN_LOG):
        with open(MASKRCNN_LOG) as f:
            rcnn_log = json.load(f)
        rcnn_runs = [r for r in rcnn_log['runs']
                     if r['training_info'].get('model_type', '') == 'MaskRCNN-ResNet50-FPN-V2']
        if rcnn_runs:
            best_rcnn = max(rcnn_runs, key=lambda r: r['metrics']['average']['f1'])
            rcnn_f1   = best_rcnn['metrics']['average']['f1']
            rcnn_p    = best_rcnn['metrics']['average']['precision']
            rcnn_r    = best_rcnn['metrics']['average']['recall']
            rcnn_ep   = best_rcnn['training_info']['epochs_trained']
        else:
            rcnn_f1 = rcnn_p = rcnn_r = rcnn_ep = 0
    else:
        print("No Mask R-CNN log found — run evaluate.py first")
        rcnn_f1 = rcnn_p = rcnn_r = rcnn_ep = 0

    # Load YOLO results
    if os.path.exists(YOLO_LOG):
        with open(YOLO_LOG) as f:
            yolo_log = json.load(f)
        best_yolo    = max(yolo_log['runs'],
                           key=lambda r: r['results']['seg_mAP50'])
        yolo_map50   = best_yolo['results']['seg_mAP50']
        yolo_map5095 = best_yolo['results']['seg_mAP50_95']
        yolo_p       = best_yolo['results']['precision']
        yolo_r       = best_yolo['results']['recall']
        yolo_ep      = best_yolo['config']['epochs']
        yolo_model   = best_yolo['model']
    else:
        print("No YOLO log found — run train_yolo.py first")
        yolo_map50 = yolo_map5095 = yolo_p = yolo_r = yolo_ep = 0
        yolo_model = "N/A"

    print(f"\n{'Metric':<25} {'Mask R-CNN':>15} {'YOLOv11':>15}")
    print("-" * 55)
    print(f"{'Model':<25} {'ResNet50 FPN V2':>15} {yolo_model:>15}")
    print(f"{'Epochs':<25} {str(rcnn_ep):>15} {str(yolo_ep):>15}")
    print(f"{'Precision':<25} {rcnn_p:>15.2%} {yolo_p:>15.2%}")
    print(f"{'Recall':<25} {rcnn_r:>15.2%} {yolo_r:>15.2%}")
    print(f"{'F1 / Mask mAP@50':<25} {rcnn_f1:>15.2%} {yolo_map50:>15.2%}")
    print(f"{'mAP@50-95':<25} {'N/A':>15} {yolo_map5095:>15.2%}")
    print("=" * 55)

    print("\nVerdict:")
    if yolo_map50 > rcnn_f1:
        diff = (yolo_map50 - rcnn_f1) * 100
        print(f"  ✅ YOLOv11 wins by {diff:.1f}%")
        print(f"  → Use YOLOv11 (inference_yolo.py)")
    else:
        diff = (rcnn_f1 - yolo_map50) * 100
        print(f"  ✅ Mask R-CNN wins by {diff:.1f}%")
        print(f"  → Use Mask R-CNN (inference.py)")

    print("\nSpeed:")
    print("  YOLOv11    → ~5ms  per image  (real-time)")
    print("  Mask R-CNN → ~100ms per image (slower)")


if __name__ == '__main__':
    compare()
