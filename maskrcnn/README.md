# Perky Pear Maturity Detection
# Python scripts for training and evaluating pear maturity detection models

## Files

| File                  | Purpose                                      | Run order |
|-----------------------|----------------------------------------------|-----------|
| explore_dataset.py    | Visualize COCO dataset before training       | 1st       |
| dataset.py            | COCO dataset loader for Mask R-CNN           | (import)  |
| model.py              | Mask R-CNN model definition                  | (import)  |
| train.py              | Train Mask R-CNN on your dataset             | 2nd       |
| inference.py          | Run Mask R-CNN on new images                 | after training |
| evaluate.py           | Evaluate Mask R-CNN, save to log             | after training |
| compare.py            | Compare all runs from evaluation_log.json    | anytime   |
| coco_to_yolo.py       | Convert COCO format → YOLO format            | before YOLO |
| train_yolo.py         | Train YOLOv11 segmentation model             | after convert |
| inference_yolo.py     | Run YOLOv11 on new images                    | after training |
| evaluate_yolo.py      | Evaluate YOLOv11, save to same log           | after training |
| compare_models.py     | Compare YOLOv11 vs Mask R-CNN                | anytime   |

## Workflow

### Mask R-CNN
python explore_dataset.py
python train.py
python evaluate.py --checkpoint best_pear_model.pth --notes "25 epochs"
python inference.py

### YOLOv11
python coco_to_yolo.py
python train_yolo.py
python evaluate_yolo.py --checkpoint runs/pear_yolo11/weights/best.pt --notes "100 epochs"
python inference_yolo.py

### Compare everything
python compare.py          # all Mask R-CNN runs
python compare_models.py   # YOLO vs Mask R-CNN

## Classes
- 0: Background (Mask R-CNN only)
- 1 / 0: Middle-Ripe
- 2 / 1: Ripe
- 3 / 2: Unripe

## Requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pycocotools opencv-python matplotlib pillow ultralytics
