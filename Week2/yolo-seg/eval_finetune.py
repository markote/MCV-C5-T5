from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
import torch


model = YOLO("/mnt/home/MCV-C5-T5/Week2/yolo-seg/runs/segment/kitti_mots_finetune/weights/best.pt")
metrics = model.val(data="kitti_mots.yaml")
print(metrics.mask.map)  # map50-95
