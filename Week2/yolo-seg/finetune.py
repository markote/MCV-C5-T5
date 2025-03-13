from ultralytics import YOLO
import wandb
wandb.login(key="")
wandb.init(project="YOLO")
# Load a pretrained YOLO model
model = YOLO("yolo11l-seg.pt")

# Train on the new dataset
model.train(
    data='kitti_mots_finetune.yaml',
    pretrained = True,
    epochs=30,
    imgsz=640,
    workers=32,
    batch= 8*3,
    name='kitti_mots_finetune',
    plots = True,
    device = [0,1,2]
)
