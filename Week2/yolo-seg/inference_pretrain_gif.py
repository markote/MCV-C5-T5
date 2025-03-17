#!/usr/bin/env python3
import os
import glob
import cv2
import random
import imageio
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# 1. Color cache for consistent class colors
# --------------------------------------------------
color_cache = {}
def get_color_for_class(cls_id: int):
    """
    Returns a unique (B, G, R) color for each class ID.
    """
    cls_id = cls_id.item() if hasattr(cls_id, "item") else cls_id
    if cls_id not in color_cache:
        random.seed(cls_id)
        color = tuple(random.randint(0, 255) for _ in range(3))
        color_cache[cls_id] = color
    return color_cache[cls_id]

# --------------------------------------------------
# 2. Model and directories
# --------------------------------------------------
yl = "11l-seg"
eval_images_dir = "/mnt/dataset/0018"
prediction_dir = f"prediction_{yl}"

Path(eval_images_dir.replace("my_datasets/KITTI_MOTS_YOLO", prediction_dir)).mkdir(parents=True, exist_ok=True)

print(f"Loading model: yolo{yl}.pt")
yln = f"yolo{yl}.pt"
model = YOLO(yln)

# Override the internal class names if desired
model.model.names = {
    0: "person",
    1: "car",
    **{i: f"class_{i}" for i in range(2, 80)}
}

# --------------------------------------------------
# 3. Gather images
# --------------------------------------------------
image_paths = glob.glob(os.path.join(eval_images_dir, "*.png"))
image_paths.sort()
if not image_paths:
    print(f"No images found in {eval_images_dir}")
    exit(1)

frames = []

# --------------------------------------------------
# 4. Inference and drawing
# --------------------------------------------------
for image_path in image_paths:
    img = cv2.imread(image_path)

    # Run YOLO inference (segmentation model)
    results = model.predict(image_path, device='cuda:0')

    # Mapping from COCO classes to KITTI-MOTS classes
    coco2kittimots = {2.0: 1.0, 5.0: 1.0, 7.0: 1.0, 0.0: 0.0}

    for result in results:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # A) BOUNDING BOXES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        boxes = result.boxes.xyxy.cpu().numpy()         # (N, 4)
        confs = result.boxes.conf.cpu().numpy()           # (N,)
        clss = result.boxes.cls.cpu().numpy().astype(int) # (N,)

        # Remap classes if they appear in coco2kittimots
        for i, c in enumerate(clss):
            if c in coco2kittimots:
                clss[i] = coco2kittimots[c]
            else:
                clss[i] = 79.0  # placeholder for "unused" class

        # Draw bounding boxes and labels
        for box, conf, cls_id in zip(boxes, confs, clss):
            if cls_id in coco2kittimots.values():
                x1, y1, x2, y2 = map(int, box)
                label_text = f"{model.model.names.get(cls_id, 'unknown')} {conf:.2f}"
                color = get_color_for_class(cls_id)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=1)

                # Draw label background
                (tw, th), base = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1), (x1 + tw, y1 - th - base), color, -1)
                # Draw label text
                cv2.putText(img, label_text, (x1, y1 - base), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 0), 1)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # B) SEGMENTATION MASKS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # shape: (N, h, w)
            # Loop over each detection (using the same order as boxes)
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, clss)):
                if cls_id in coco2kittimots.values() and i < len(masks):
                    # Convert mask to boolean at its native resolution
                    mask = masks[i] > 0.5
                    # Resize mask to match the image dimensions
                    mask_resized = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    color = get_color_for_class(cls_id)
     
                    masked_image = img.copy()
                    masked_image = np.where(np.stack([mask_resized,mask_resized,mask_resized], axis=-1),
                          color,
                          masked_image)
                    masked_image = masked_image.astype(np.uint8)
                    img = cv2.addWeighted(img, 0.3, masked_image, 0.7, 0)

    print(f"Processed: {image_path}")

    # Convert BGR -> RGB for imageio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (int(img.shape[1]*2/3), int(img.shape[0]*2/3)))
    frames.append(img_rgb)

# --------------------------------------------------
# 5. Save as GIF
# --------------------------------------------------
output_gif_path = "predictions_pretrain.gif"
imageio.mimsave(output_gif_path, frames, fps=20)
print(f"GIF saved to '{output_gif_path}'")