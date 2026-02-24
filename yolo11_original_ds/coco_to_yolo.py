# coco_to_yolo.py
# Converts Roboflow COCO segmentation format → YOLO segmentation format
# Run this BEFORE train_yolo.py

import json
import os
import shutil

DATASET_SPLITS = {
    "train": {
        "images": "dataset/train/images",
        "ann":    "dataset/train/_annotations.coco.json"
    },
    "valid": {
        "images": "dataset/valid/images",
        "ann":    "dataset/valid/_annotations.coco.json"
    },
    "test": {
        "images": "dataset/test/images",
        "ann":    "dataset/test/_annotations.coco.json"
    }
}

OUTPUT_DIR = "dataset_yolo"


def coco_to_yolo(split_name, image_dir, ann_file, output_dir):
    print(f"\nConverting {split_name}...")

    out_img_dir = os.path.join(output_dir, split_name, "images")
    out_lbl_dir = os.path.join(output_dir, split_name, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    with open(ann_file) as f:
        coco = json.load(f)

    images      = {img['id']: img for img in coco['images']}
    valid_cats  = [c for c in coco['categories'] if c['id'] != 0]
    cat_id_map  = {cat['id']: i for i, cat in enumerate(valid_cats)}

    print(f"  Class mapping (YOLO 0-indexed):")
    for cat in valid_cats:
        print(f"    {cat['name']} (coco_id={cat['id']}) → yolo_class={cat_id_map[cat['id']]}")

    ann_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)

    converted, skipped = 0, 0

    for img_id, img_info in images.items():
        filename = img_info['file_name']
        W        = img_info['width']
        H        = img_info['height']

        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join(out_img_dir, filename)

        if not os.path.exists(src_path):
            skipped += 1
            continue

        shutil.copy2(src_path, dst_path)

        anns  = ann_by_image.get(img_id, [])
        lines = []

        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in cat_id_map:
                continue

            yolo_class = cat_id_map[cat_id]

            if ann.get('segmentation') and len(ann['segmentation']) > 0:
                polygon    = ann['segmentation'][0]
                normalized = []
                for i, coord in enumerate(polygon):
                    if i % 2 == 0:
                        normalized.append(round(coord / W, 6))
                    else:
                        normalized.append(round(coord / H, 6))
                normalized  = [max(0.0, min(1.0, c)) for c in normalized]
                coords_str  = " ".join(map(str, normalized))
                lines.append(f"{yolo_class} {coords_str}")
            else:
                x, y, w, h  = ann['bbox']
                x_center    = (x + w / 2) / W
                y_center    = (y + h / 2) / H
                w_norm      = w / W
                h_norm      = h / H
                lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} "
                              f"{w_norm:.6f} {h_norm:.6f}")

        label_name = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(out_lbl_dir, label_name), 'w') as f:
            f.write("\n".join(lines))

        converted += 1

    print(f"  Converted: {converted} | Skipped: {skipped}")


def create_yaml(output_dir, categories):
    yaml_content = f"""# Pear Maturity Detection — YOLOv11 Dataset Config

path  : {os.path.abspath(output_dir)}
train : train/images
val   : valid/images
test  : test/images

nc: {len(categories)}

names:
"""
    for i, cat in enumerate(categories):
        yaml_content += f"  {i}: {cat['name']}\n"

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\ndata.yaml created at: {yaml_path}")
    return yaml_path


if __name__ == '__main__':
    with open(DATASET_SPLITS['train']['ann']) as f:
        coco_data = json.load(f)

    valid_categories = [c for c in coco_data['categories'] if c['id'] != 0]
    print(f"Found {len(valid_categories)} classes: "
          f"{[c['name'] for c in valid_categories]}")

    for split_name, paths in DATASET_SPLITS.items():
        coco_to_yolo(split_name, paths['images'], paths['ann'], OUTPUT_DIR)

    create_yaml(OUTPUT_DIR, valid_categories)
    print("\nConversion complete! Run train_yolo.py next.")
