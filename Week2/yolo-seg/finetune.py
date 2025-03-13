from ultralytics import YOLO
import wandb
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

# wandb.init(project="YOLO")
# Load a pretrained YOLO model
model = YOLO("yolo11l-seg.pt")

# Train on the new dataset
model.train(
    data='kitti_mots_finetune.yaml',
    pretrained = True,
    epochs=30,
    imgsz=640,
    workers=64,
    batch= 32*8,
    name='kitti_mots_finetune',
    plots = True,
    device = [0,1,2,3,4,5,6,7]
)
