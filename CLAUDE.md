# CLAUDE.md — Perky Pear Maturity Detection Project
# This file is the complete guide for Claude Code to understand, validate,
# organize, and verify the entire project works correctly end-to-end.

---

## PROJECT OVERVIEW

This project detects pear maturity (Unripe / Middle-Ripe / Ripe) from photos
using instance segmentation. Two models are implemented:

1. **Mask R-CNN** (ResNet50 FPN V2) — PyTorch, trained on COCO segmentation format
2. **YOLOv11-seg** (Ultralytics) — trained on converted YOLO format

Both models save evaluation results to the **same** `evaluation_log.json`
so they can be compared directly.

---

## EXPECTED FOLDER STRUCTURE

Claude Code must verify this structure exists. If any folder is missing, create it.

```
prekly pear maturity detection\
│
├── CLAUDE.md                          ← this file
│
├── maskrcnn\                          ← Mask R-CNN project
│   ├── dataset\
│   │   ├── train\
│   │   │   ├── images\                ← .jpg/.png training images
│   │   │   └── _annotations.coco.json
│   │   ├── valid\
│   │   │   ├── images\
│   │   │   └── _annotations.coco.json
│   │   └── test\
│   │       ├── images\
│   │       └── _annotations.coco.json
│   ├── explore_dataset.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── compare.py
│   └── README.md
│
└── yolo11_original_ds\                ← YOLOv11 project
    ├── dataset\                       ← original COCO format (input)
    │   ├── train\images\
    │   ├── valid\images\
    │   └── test\images\
    ├── dataset_yolo\                  ← converted YOLO format (output of coco_to_yolo.py)
    │   ├── train\
    │   │   ├── images\
    │   │   └── labels\
    │   ├── valid\
    │   │   ├── images\
    │   │   └── labels\
    │   ├── test\
    │   │   ├── images\
    │   │   └── labels\
    │   └── data.yaml
    ├── runs\
    │   └── pear_yolo11\
    │       └── weights\
    │           ├── best.pt
    │           └── last.pt
    ├── coco_to_yolo.py
    ├── train_yolo.py
    ├── inference_yolo.py
    ├── evaluate_yolo.py
    └── compare_models.py
```

---

## DATASET INFORMATION

- **Format**: COCO segmentation (JSON with polygon annotations)
- **Source**: Roboflow — "pricky-maturity-hbUM" dataset
- **Total images**: 1179 training images
- **Annotation file**: `_annotations.coco.json`

### Classes (IMPORTANT — Read carefully)

The COCO JSON has 4 category entries but only 3 are real classes:

```json
id=0  name="pricky-maturity-hbUM"  → SKIP THIS — it is the Roboflow dataset name, NOT a class
id=1  name="Middle-Ripe"           → real class — 363 annotations
id=2  name="Ripe"                  → real class — 1050 annotations (dominant)
id=3  name="Unripe"                → real class — 447 annotations
```

### Class Mapping by Model

| Real Class    | COCO id | Mask R-CNN label | YOLO class id |
|---------------|---------|------------------|---------------|
| background    | —       | 0 (reserved)     | — (automatic) |
| Middle-Ripe   | 1       | 1                | 0             |
| Ripe          | 2       | 2                | 1             |
| Unripe        | 3       | 3                | 2             |

### Class Imbalance Warning
Ripe has 1050 samples vs 363 for Middle-Ripe. Model will likely score lowest
on Middle-Ripe. This is expected and documented.

---

## FILE DESCRIPTIONS AND VALIDATION RULES

Claude Code must read each file and validate all rules listed below.

---

### maskrcnn/explore_dataset.py

**Purpose**: Explore COCO dataset before training. Run first.

**Validation rules**:
- [ ] `TRAIN_ANN` path points to `dataset/train/_annotations.coco.json`
- [ ] `TRAIN_IMG` path points to `dataset/train/images`
- [ ] Uses `pycocotools.coco.COCO` to load annotations
- [ ] Prints all categories including id=0
- [ ] Prints total images and annotations
- [ ] Prints annotation count per category
- [ ] Visualizes one sample image with masks and bounding boxes
- [ ] Saves output as `sample_annotation.png`

**Expected output when run**:
```
Categories:
  id=0  name=pricky-maturity-hbUM
  id=1  name=Middle-Ripe
  id=2  name=Ripe
  id=3  name=Unripe

Total images     : 1179
Total annotations: 1860

Annotations per category:
  pricky-maturity-hbUM: 0
  Middle-Ripe: 363
  Ripe: 1050
  Unripe: 447
```

---

### maskrcnn/dataset.py

**Purpose**: PyTorch Dataset class for loading COCO segmentation data.

**Validation rules**:
- [ ] Imports: `torch`, `numpy`, `pycocotools.coco.COCO`, `PIL.Image`, `torchvision.transforms`
- [ ] `IMAGENET_MEAN = [0.485, 0.456, 0.406]` — must match these exact values
- [ ] `IMAGENET_STD  = [0.229, 0.224, 0.225]` — must match these exact values
- [ ] Class `PearAugmentation` exists with color-safe transforms only:
  - [ ] Brightness adjustment allowed ✅
  - [ ] Contrast adjustment allowed ✅
  - [ ] Horizontal flip allowed ✅
  - [ ] Small rotation allowed ✅
  - [ ] NO hue shift ❌ (would change green→yellow = wrong label)
  - [ ] NO saturation change ❌ (destroys ripeness color signal)
  - [ ] NO grayscale ❌ (removes all color information)
- [ ] Class `CocoDataset` skips `category_id == 0` (dataset name, not a class)
- [ ] `cat_id_map` maps `{1:1, 2:2, 3:3}` (COCO id → model label)
- [ ] `annToMask()` used to convert polygon → binary mask
- [ ] Bbox converted from `[x, y, w, h]` → `[x1, y1, x2, y2]`
- [ ] Empty annotation handling: returns zero tensors, does NOT crash
- [ ] `ToTensor()` applied before `Normalize()`
- [ ] `collate_fn` returns `tuple(zip(*batch))` — required for variable object counts

**Critical check**: Normalization must be applied to BOTH train and val/test.
If normalization is missing from inference, predictions will be wrong.

---

### maskrcnn/model.py

**Purpose**: Defines Mask R-CNN architecture with custom heads.

**Validation rules**:
- [ ] Uses `maskrcnn_resnet50_fpn_v2` with `MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT`
- [ ] Box predictor head replaced with `FastRCNNPredictor(in_features, num_classes)`
- [ ] Mask predictor head replaced with `MaskRCNNPredictor(in_channels, 256, num_classes)`
- [ ] Function signature: `get_model(num_classes)` — takes num_classes as argument
- [ ] `num_classes = 4` (background + 3 pear classes) — validate this is passed correctly

**Architecture diagram for reference**:
```
Input image (3, H, W)
       ↓
ResNet50 + FPN backbone  [PRETRAINED — weights kept]
       ↓
Region Proposal Network  [PRETRAINED — weights kept]
       ↓
ROI Align
       ↓
Box Predictor Head       [REPLACED — trained for 4 classes]
Mask Predictor Head      [REPLACED — trained for 4 classes]
       ↓
Output: boxes, labels, scores, masks
```

---

### maskrcnn/train.py

**Purpose**: Main training loop for Mask R-CNN.

**Validation rules**:
- [ ] TF32 enabled: `torch.backends.cuda.matmul.allow_tf32 = True`
- [ ] TF32 enabled: `torch.backends.cudnn.allow_tf32 = True`
- [ ] `torch.backends.cudnn.benchmark = True`
- [ ] `NUM_CLASSES = 4`
- [ ] `BATCH_SIZE = 2` (safe for 4GB VRAM)
- [ ] `ACCUMULATION_STEPS = 4` (simulates batch size of 8)
- [ ] `num_workers = 0` — REQUIRED on Windows (multiprocessing fix)
- [ ] `persistent_workers = False` when `num_workers = 0`
- [ ] All training code inside `if __name__ == '__main__':` block — REQUIRED on Windows
- [ ] Backbone frozen for first 5 epochs, unfrozen at epoch 5
- [ ] When backbone unfrozen: LR reduced to `LR / 10`
- [ ] Mixed precision: `GradScaler()` and `autocast(device_type="cuda")`
- [ ] Gradient clipping: `clip_grad_norm_(model.parameters(), max_norm=5.0)`
- [ ] `non_blocking=True` on `.to(device)` calls
- [ ] `torch.cuda.empty_cache()` called after train and val phases
- [ ] Validation uses `model.train()` mode (Mask R-CNN needs this for loss computation)
- [ ] Saves checkpoint dict with keys: `epoch`, `model`, `optimizer`, `val_loss`, `num_classes`
- [ ] Saves only when val_loss improves (best model tracking)
- [ ] Prints epoch time and estimated remaining time
- [ ] `CosineAnnealingLR` scheduler used

**Common bugs to check**:
- `persistent_workers=True` with `num_workers=0` → will crash on Windows
- Missing `if __name__ == '__main__':` → RuntimeError about bootstrapping
- Using `model.eval()` for validation → Mask R-CNN won't compute losses in eval mode

---

### maskrcnn/inference.py

**Purpose**: Run trained Mask R-CNN on new images.

**Validation rules**:
- [ ] Loads checkpoint from `best_pear_model.pth`
- [ ] `transform` includes BOTH `T.ToTensor()` AND `T.Normalize(mean, std)`
- [ ] Same normalization values as `dataset.py`: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
- [ ] Model set to `.eval()` mode before inference
- [ ] `torch.no_grad()` context used
- [ ] Confidence threshold filtering applied
- [ ] `CATEGORIES` dict: `{0:'background', 1:'Middle-Ripe', 2:'Ripe', 3:'Unripe'}`
- [ ] `CLASS_COLORS` dict: `{1:'orange', 2:'green', 3:'red'}`
- [ ] Bounding boxes drawn with correct class colors
- [ ] Segmentation masks overlaid semi-transparently
- [ ] Summary count box shown in corner of image
- [ ] `predict_folder()` function exists for batch processing
- [ ] Both functions check if image file exists before processing

---

### maskrcnn/evaluate.py

**Purpose**: Evaluate Mask R-CNN and save results to `evaluation_log.json`.

**Validation rules**:
- [ ] `LOG_FILE = "evaluation_log.json"` — must match exactly
- [ ] `IOU_THRESHOLD = 0.5`
- [ ] `SCORE_THRESHOLD = 0.5`
- [ ] Computes TP, FP, FN per class
- [ ] IoU computed using `torchvision.ops.box_iou`
- [ ] Precision, Recall, F1 computed per class
- [ ] Average metrics computed across all classes
- [ ] `save_run()` function appends to existing log (does NOT overwrite)
- [ ] Saved run structure must have these exact keys:
  ```json
  {
    "run_id": int,
    "timestamp": "YYYY-MM-DD HH:MM:SS",
    "checkpoint": "path/to/model.pth",
    "notes": "string",
    "training_info": {
      "epochs_trained": int,
      "best_val_loss": float,
      "num_classes": int,
      "model_type": "MaskRCNN-ResNet50-FPN-V2"
    },
    "eval_settings": {
      "iou_threshold": 0.5,
      "score_threshold": 0.5
    },
    "metrics": {
      "Middle-Ripe": {"precision": float, "recall": float, "f1": float, "tp": int, "fp": int, "fn": int},
      "Ripe":        {"precision": float, "recall": float, "f1": float, "tp": int, "fp": int, "fn": int},
      "Unripe":      {"precision": float, "recall": float, "f1": float, "tp": int, "fp": int, "fn": int},
      "average":     {"precision": float, "recall": float, "f1": float}
    }
  }
  ```
- [ ] `--checkpoint` and `--notes` CLI arguments work
- [ ] `print_results()` prints table with Precision / Recall / F1 per class

---

### maskrcnn/compare.py

**Purpose**: Compare ALL runs from `evaluation_log.json` (both models).

**Validation rules**:
- [ ] Reads from `evaluation_log.json`
- [ ] Shows all runs in one table: run_id, date, model_type, epochs, val_loss, F1, precision, recall
- [ ] Marks best run with ⭐
- [ ] Shows per-class F1 breakdown of best run
- [ ] Shows F1 per class across all runs in a matrix
- [ ] Handles mixed runs (Mask R-CNN + YOLOv11) in same log file

---

### yolo11_original_ds/coco_to_yolo.py

**Purpose**: Convert COCO segmentation JSON → YOLO segmentation .txt labels.

**Validation rules**:
- [ ] Input paths point to `dataset/` (COCO format)
- [ ] Output path is `dataset_yolo/`
- [ ] Skips `category_id == 0` (dataset name)
- [ ] YOLO class IDs are 0-indexed: `{1→0, 2→1, 3→2}`
- [ ] Polygon coordinates normalized to [0, 1] by dividing by image width/height
- [ ] Coordinates clamped to [0.0, 1.0] to handle edge annotations
- [ ] Fallback to bbox if no segmentation polygon exists
- [ ] Creates `.txt` label file even for images with no annotations (empty file)
- [ ] Copies images to `dataset_yolo/{split}/images/`
- [ ] Creates `dataset_yolo/data.yaml` with correct format:
  ```yaml
  path  : /absolute/path/to/dataset_yolo
  train : train/images
  val   : valid/images
  test  : test/images
  nc    : 3
  names:
    0: Middle-Ripe
    1: Ripe
    2: Unripe
  ```
- [ ] `data.yaml` uses ABSOLUTE path (not relative) for `path` field

**Critical check**: YOLO label files use segmentation format (polygon), not detection format:
```
# Detection format (5 values):
class_id cx cy w h

# Segmentation format (5+ values):
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

---

### yolo11_original_ds/train_yolo.py

**Purpose**: Train YOLOv11 segmentation model.

**Validation rules**:
- [ ] `"model": "yolo11s-seg.pt"` — small model for 4GB VRAM
- [ ] `"data"` points to `dataset_yolo/data.yaml`
- [ ] `"workers": 0` — REQUIRED on Windows
- [ ] `"device": 0` — GPU
- [ ] `"amp": True` — mixed precision for VRAM savings
- [ ] Color-safe augmentation settings:
  - [ ] `"hsv_h": 0.0` — hue shift OFF
  - [ ] `"hsv_s": 0.0` — saturation OFF
  - [ ] `"hsv_v": 0.3` — brightness only
  - [ ] `"flipud": 0.0` — vertical flip OFF
  - [ ] `"mixup": 0.0` — mixup OFF
- [ ] `"project": "runs"` and `"name": "pear_yolo11"`
- [ ] `save_training_info()` saves to `yolo_training_log.json`
- [ ] After training, prints BEST VALIDATION RESULTS clearly labeled
- [ ] After training, runs `best_model.val(split="test")` and prints TEST SET RESULTS
- [ ] All code inside `if __name__ == '__main__':` block
- [ ] Prints path to `results.csv` and `results.png` after training

---

### yolo11_original_ds/inference_yolo.py

**Purpose**: Run trained YOLOv11 on new images using OpenCV.

**Validation rules**:
- [ ] `MODEL_PATH = "runs/pear_yolo11/weights/best.pt"`
- [ ] Uses `YOLO(MODEL_PATH)` to load model
- [ ] `CLASS_COLORS` uses BGR format (OpenCV): `{0:(0,165,255), 1:(0,200,0), 2:(0,0,220)}`
- [ ] `CATEGORIES`: `{0:"Middle-Ripe", 1:"Ripe", 2:"Unripe"}`
- [ ] Mask resized to image dimensions using `cv2.resize`
- [ ] Overlay applied with `cv2.addWeighted`
- [ ] Label text drawn with background rectangle
- [ ] Summary text drawn in top-left corner
- [ ] `predict_folder()` function exists
- [ ] Checks if image exists before processing

---

### yolo11_original_ds/evaluate_yolo.py

**Purpose**: Evaluate YOLOv11 — saves to SAME `evaluation_log.json` as evaluate.py.

**Validation rules**:
- [ ] `LOG_FILE = "evaluation_log.json"` — MUST be identical to maskrcnn/evaluate.py
- [ ] `IOU_THRESHOLD = 0.5` — same as Mask R-CNN
- [ ] `SCORE_THRESHOLD = 0.5` — same as Mask R-CNN
- [ ] `CATEGORIES = {0:"Middle-Ripe", 1:"Ripe", 2:"Unripe"}` (YOLO 0-indexed)
- [ ] `yolo_label_to_box()` handles BOTH detection format (5 values) and segmentation format (5+ values)
- [ ] Saved run has `"model_type": "YOLOv11-seg"` in `training_info`
- [ ] Saved run has `"eval_split"` field showing which split was evaluated
- [ ] JSON structure identical to evaluate.py output (same keys, same nesting)
- [ ] `--split` argument accepts `valid` or `train`
- [ ] `_get_yolo_epochs()` reads from `runs/pear_yolo11/results.csv`
- [ ] `_get_yolo_val_loss()` reads minimum box loss from `runs/pear_yolo11/results.csv`
- [ ] `print_results()` output format identical to maskrcnn/evaluate.py

---

### yolo11_original_ds/compare_models.py

**Purpose**: Compare best YOLOv11 vs best Mask R-CNN run.

**Validation rules**:
- [ ] Reads `yolo_training_log.json` for YOLO results
- [ ] Reads `evaluation_log.json` for Mask R-CNN results
- [ ] Filters Mask R-CNN runs by `model_type == "MaskRCNN-ResNet50-FPN-V2"`
- [ ] Shows side-by-side table: precision, recall, F1/mAP50, mAP50-95
- [ ] Prints verdict: which model wins and by how much
- [ ] Prints speed comparison note

---

## CROSS-FILE DEPENDENCY VALIDATION

Claude Code must check these dependencies are consistent across ALL files:

### 1 — Shared Log File
Both evaluate.py and evaluate_yolo.py must write to the same file:
```python
# evaluate.py (maskrcnn folder)
LOG_FILE = "evaluation_log.json"

# evaluate_yolo.py (yolo11_original_ds folder)
LOG_FILE = "evaluation_log.json"
```
✅ Check: both use identical filename

### 2 — Normalization Consistency
`dataset.py` and `inference.py` must use identical normalization:
```python
# dataset.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# inference.py
T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
✅ Check: values identical in both files

### 3 — NUM_CLASSES Consistency
```python
# model.py
get_model(num_classes)  # accepts as parameter

# train.py
NUM_CLASSES = 4
model = get_model(NUM_CLASSES)

# inference.py
NUM_CLASSES = 4
model = get_model(NUM_CLASSES)

# evaluate.py
NUM_CLASSES = 4
model = get_model(NUM_CLASSES)
```
✅ Check: all use 4

### 4 — Category Mapping Consistency
```python
# evaluate.py
CATEGORIES = {1: 'Middle-Ripe', 2: 'Ripe', 3: 'Unripe'}

# inference.py
CATEGORIES = {0: 'background', 1: 'Middle-Ripe', 2: 'Ripe', 3: 'Unripe'}
```
Note: inference.py includes background (0), evaluate.py starts from 1 (correct)

### 5 — YOLO Class IDs Consistency
```python
# coco_to_yolo.py
cat_id_map = {1:0, 2:1, 3:2}   # Middle-Ripe=0, Ripe=1, Unripe=2

# inference_yolo.py
CATEGORIES = {0:"Middle-Ripe", 1:"Ripe", 2:"Unripe"}

# evaluate_yolo.py
CATEGORIES = {0:"Middle-Ripe", 1:"Ripe", 2:"Unripe"}
```
✅ Check: all YOLO files use 0-indexed IDs consistently

### 6 — Model Checkpoint Path Consistency
```python
# train.py
SAVE_PATH = "best_pear_model.pth"

# inference.py
torch.load("best_pear_model.pth")

# evaluate.py (default argument)
parser.add_argument('--checkpoint', default='best_pear_model.pth')
```
✅ Check: all three use same filename

### 7 — YOLO Checkpoint Path Consistency
```python
# train_yolo.py — saves to:
"runs/pear_yolo11/weights/best.pt"

# inference_yolo.py
MODEL_PATH = "runs/pear_yolo11/weights/best.pt"

# evaluate_yolo.py
MODEL_PATH = "runs/pear_yolo11/weights/best.pt"
```
✅ Check: all three use same path

### 8 — Windows-Specific Requirements (ALL files must follow)
- [ ] `num_workers = 0` in all DataLoader calls
- [ ] `persistent_workers = False` when `num_workers = 0`
- [ ] All entry-point code inside `if __name__ == '__main__':` in train.py and train_yolo.py

---

## VALIDATION CHECKLIST

When asked to validate the project, Claude Code must run through this list:

### Step 1 — Folder Structure
```
□ maskrcnn/ folder exists
□ maskrcnn/dataset/train/images/ exists
□ maskrcnn/dataset/valid/images/ exists
□ maskrcnn/dataset/test/images/ exists
□ maskrcnn/dataset/train/_annotations.coco.json exists
□ maskrcnn/dataset/valid/_annotations.coco.json exists
□ maskrcnn/dataset/test/_annotations.coco.json exists
□ yolo11_original_ds/dataset/ exists
□ yolo11_original_ds/dataset_yolo/data.yaml exists (after coco_to_yolo.py)
□ All 13 Python files present in correct locations
```

### Step 2 — Import Validation
Check each file can be imported without errors:
```python
# Run in maskrcnn\ directory
python -c "from dataset import CocoDataset, collate_fn; print('dataset.py OK')"
python -c "from model import get_model; print('model.py OK')"
python -c "import train; print('train.py OK')"
```

### Step 3 — Dataset Validation
```python
# Run in maskrcnn\ directory
python explore_dataset.py
# Expected: prints category list, image count, annotation count
# Expected: saves sample_annotation.png
```

### Step 4 — Model Creation Validation
```python
# Run in maskrcnn\ directory
python -c "
from model import get_model
import torch
model = get_model(4)
print('Model created OK')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
# Expected: ~44 million parameters
```

### Step 5 — COCO Annotation Validation
```python
# Run this to verify annotation file is valid
python -c "
import json
for split in ['train', 'valid', 'test']:
    with open(f'dataset/{split}/_annotations.coco.json') as f:
        data = json.load(f)
    cats = [c for c in data['categories'] if c['id'] != 0]
    print(f'{split}: {len(data[\"images\"])} images, {len(data[\"annotations\"])} annotations, {len(cats)} classes')
"
```

### Step 6 — YOLO Data Validation
```python
# Run in yolo11_original_ds\ directory
python -c "
import os
for split in ['train', 'valid', 'test']:
    imgs   = len(os.listdir(f'dataset_yolo/{split}/images'))
    labels = len(os.listdir(f'dataset_yolo/{split}/labels'))
    print(f'{split}: {imgs} images, {labels} labels')
    assert imgs == labels, f'MISMATCH in {split}!'
print('YOLO dataset OK')
"
```

### Step 7 — data.yaml Validation
```python
# Run in yolo11_original_ds\ directory
python -c "
with open('dataset_yolo/data.yaml') as f:
    print(f.read())
# Verify: nc=3, names has Middle-Ripe/Ripe/Unripe, path is absolute
"
```

### Step 8 — GPU Validation
```python
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
print('PyTorch:', torch.__version__)
"
# Expected: CUDA True, RTX 3050 Ti, ~4GB, 2.6.0+cu124
```

### Step 9 — Evaluation Log Compatibility
After running both models, verify the log is compatible:
```python
python -c "
import json
with open('evaluation_log.json') as f:
    log = json.load(f)
for run in log['runs']:
    t = run['training_info']
    m = run['metrics']['average']
    print(f\"Run #{run['run_id']} | {t.get('model_type','?')} | F1={m['f1']:.2%} | {run['notes']}\")
"
```

---

## COMMON ERRORS AND FIXES

Claude Code must know these fixes:

### Error: RuntimeError bootstrapping multiprocessing
```
Fix: Set num_workers=0 in all DataLoader calls
     Wrap all training code in if __name__ == '__main__':
```

### Error: CUDA out of memory
```
Fix: Reduce BATCH_SIZE from 2 to 1
     Add torch.cuda.set_per_process_memory_fraction(0.85)
     Add T.Resize((480, 480)) to transforms
```

### Error: pycocotools not found
```
Fix: pip install pycocotools
     If that fails: pip install pycocotools-win
```

### Error: Model predictions all wrong class
```
Fix: Check normalization is applied in inference.py
     Verify IMAGENET_MEAN and IMAGENET_STD match dataset.py
```

### Error: Middle-Ripe never detected
```
Fix: Lower threshold from 0.5 to 0.3
     This class has only 363 samples — model is less confident
```

### Error: data.yaml path not found
```
Fix: Make sure path in data.yaml is ABSOLUTE, not relative
     Re-run coco_to_yolo.py to regenerate it
```

### Error: evaluation_log.json runs mixed up
```
Fix: Each run has model_type field to distinguish them
     compare.py filters by model_type automatically
```

### Error: YOLOv11 label count != image count
```
Fix: Re-run coco_to_yolo.py
     Check that all images in dataset/ have corresponding _annotations.coco.json entries
```

---

## RUN ORDER

### First Time Setup
```powershell
# 1. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pycocotools opencv-python matplotlib pillow ultralytics

# 2. Explore dataset (run from maskrcnn\)
cd maskrcnn
python explore_dataset.py

# 3. Convert dataset for YOLO (run from yolo11_original_ds\)
cd ..\yolo11_original_ds
python coco_to_yolo.py
```

### Train Mask R-CNN
```powershell
cd maskrcnn
python train.py
# Saves: best_pear_model.pth
# Time: ~6-8 hours on RTX 3050 Ti
```

### Train YOLOv11
```powershell
cd yolo11_original_ds
python train_yolo.py
# Saves: runs/pear_yolo11/weights/best.pt
# Time: ~2-3 hours on RTX 3050 Ti
```

### Evaluate and Compare
```powershell
# Mask R-CNN evaluation
cd maskrcnn
python evaluate.py --checkpoint best_pear_model.pth --notes "25 epochs lr=0.005"

# YOLOv11 evaluation
cd ..\yolo11_original_ds
python evaluate_yolo.py --checkpoint runs/pear_yolo11/weights/best.pt --notes "100 epochs yolo11s"

# Compare all runs (run from either folder, both read same log)
python ..\maskrcnn\compare.py

# Compare models head-to-head
python compare_models.py
```

### Run Inference
```powershell
# Mask R-CNN
cd maskrcnn
python inference.py    # edit image_path inside file first

# YOLOv11
cd ..\yolo11_original_ds
python inference_yolo.py    # edit image_path inside file first
```

---

## HARDWARE PROFILE

```
GPU  : NVIDIA GeForce RTX 3050 Ti Laptop
VRAM : 4 GB
CUDA : 12.4 (driver supports up to 13.0)
OS   : Windows
```

All code is tuned for this hardware. Key settings that depend on this:
- `BATCH_SIZE = 2` in train.py
- `num_workers = 0` everywhere (Windows)
- `yolo11s-seg.pt` not medium/large (VRAM)
- `amp = True` everywhere (saves VRAM)
- `batch = 8` in train_yolo.py (YOLO is more memory efficient)

---

## WHAT CLAUDE CODE SHOULD DO WHEN ASKED TO VALIDATE

1. Read this CLAUDE.md file completely
2. Check folder structure matches EXPECTED FOLDER STRUCTURE section
3. Read each Python file and validate against its rules in FILE DESCRIPTIONS section
4. Run CROSS-FILE DEPENDENCY VALIDATION checks
5. Run VALIDATION CHECKLIST steps 1-9
6. Report any issues found with exact file names and line numbers
7. Suggest fixes from COMMON ERRORS section when applicable
8. Confirm which files pass all checks and which need fixes
9. Never modify working files without explicit instruction
10. If a file is missing, report it and ask before creating it

---

## NOTES FOR CLAUDE CODE

- This project uses TWO separate model frameworks (PyTorch + Ultralytics)
- They share ONE evaluation log file — this is intentional for comparison
- The dataset has a class imbalance — Middle-Ripe will always score lower
- Color features are critical — never suggest hue/saturation augmentation
- All paths in Python files use forward slashes but Windows uses backslash — both work in Python
- The `id=0` category bug in COCO JSON is intentional from Roboflow — all files handle it
- YOLOv11 results are in `yolo_training_log.json`, Mask R-CNN in `evaluation_log.json`
  but both also write per-class metrics to `evaluation_log.json` for unified comparison
