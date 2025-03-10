from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
import torch


model = YOLO("/mnt/home/MCV-C5/yolo/runs/detect/kitti_mots_finetune/weights/best.pt")
metrics = model.val(data="kitti_mots_finetune.yaml")
print(metrics.box.map)  # map50-95
