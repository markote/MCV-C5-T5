from ultralytics import YOLO
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

# Load a pretrained YOLO model
model = YOLO("yolo11l-seg.pt")

# Train on the new dataset
model.train(
    data='kitti_mots.yaml',
    pretrained = True,
    epochs=30,
    imgsz=640,
    workers=64,
    batch= 32*8,
    name='kitti_mots_finetune',
    plots = True,
    lr0= 0.01319,
    lrf= 0.00991,
    momentum= 0.95925,
    weight_decay= 0.0005,
    warmup_epochs= 3.59846,
    warmup_momentum= 0.86249,
    box= 15.36496,
    cls= 0.25358,
    dfl= 2.15886,
    hsv_h= 0.01385,
    hsv_s= 0.9,
    hsv_v= 0.50318,
    degrees= 0.0,
    translate= 0.10158,
    scale= 0.37631,
    shear= 0.0,
    perspective= 0.0,
    flipud= 0.0,
    fliplr= 0.52409,
    bgr= 0.0,
    mosaic= 0.82038,
    mixup= 0.0,
    copy_paste= 0.0,
    device = [0,1,2,3,4,5,6,7]
)
