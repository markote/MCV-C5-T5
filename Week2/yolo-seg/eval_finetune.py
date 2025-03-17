from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
import torch


model = YOLO("/mnt/home/MCV-C5-T5/Week2/yolo-seg/runs/segment/kitti_mots_finetune2/weights/best.pt")
metrics = model.val(data="kitti_mots.yaml", workers=64, batch= 32*8, device = [0,1,2,3,4,5,6,7])
print(metrics.box.map)  # map50-95(B)
print(metrics.box.map50)  # map50(B)
print(metrics.box.map75)  # map75(B)
print(metrics.box.maps)  # a list contains map50-95(B) of each category
print(metrics.seg.map)  # map50-95(M)
print(metrics.seg.map50)  # map50(M)
print(metrics.seg.map75)  # map75(M)
print(metrics.seg.maps)  # a list contains map50-95(M) of each category
