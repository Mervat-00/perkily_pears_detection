# inference_yolo.py
# Run YOLOv11 model on new pear images

import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

MODEL_PATH = "runs/pear_yolo11/weights/best.pt"
THRESHOLD  = 0.5

CATEGORIES = {
    0: "Middle-Ripe",
    1: "Ripe",
    2: "Unripe"
}

# BGR colors for OpenCV
CLASS_COLORS = {
    0: (0,   165, 255),   # orange — Middle-Ripe
    1: (0,   200,   0),   # green  — Ripe
    2: (0,     0, 220),   # red    — Unripe
}

device = 0 if torch.cuda.is_available() else "cpu"
model  = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")


def predict(image_path, threshold=THRESHOLD, save_path="prediction_yolo.jpg"):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    results = model.predict(
        source  = image_path,
        conf    = threshold,
        iou     = 0.5,
        device  = device,
        imgsz   = 640,
        verbose = False
    )

    result       = results[0]
    image        = cv2.imread(image_path)
    class_counts = {name: 0 for name in CATEGORIES.values()}

    if result.masks is not None:
        masks  = result.masks.data.cpu().numpy()
        boxes  = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()

        for mask, box, label, score in zip(masks, boxes, labels, scores):
            color    = CLASS_COLORS.get(label, (255, 255, 255))
            cat_name = CATEGORIES.get(label, "unknown")
            class_counts[cat_name] += 1

            # Segmentation mask overlay
            h, w         = image.shape[:2]
            mask_resized = cv2.resize(mask.astype(np.uint8), (w, h),
                                      interpolation=cv2.INTER_NEAREST)
            overlay      = image.copy()
            overlay[mask_resized == 1] = color
            image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

            # Bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Label
            label_text  = f"{cat_name} {score:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label_text, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Summary box
    total    = sum(class_counts.values())
    summary  = [f"Total: {total}"] + [
        f"{name}: {count}" for name, count in class_counts.items() if count > 0
    ]
    y_offset = 15
    for line in summary:
        cv2.putText(image, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 25

    cv2.imwrite(save_path, image)
    print(f"\nDetected {total} pears:")
    for name, count in class_counts.items():
        if count > 0:
            print(f"  {name}: {count}")
    print(f"Saved → {save_path}")


def predict_folder(folder_path, threshold=THRESHOLD, output_dir="predictions_yolo"):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    print(f"Found {len(image_files)} images")
    for filename in image_files:
        predict(
            image_path = os.path.join(folder_path, filename),
            threshold  = threshold,
            save_path  = os.path.join(output_dir, f"pred_{filename}")
        )


if __name__ == '__main__':
    # Single image
    predict(
        image_path = "dataset/test/images/your_image.jpg",
        threshold  = 0.5,
        save_path  = "prediction_yolo.jpg"
    )

    # Entire folder (uncomment to use)
    # predict_folder("dataset/test/images", threshold=0.5)
