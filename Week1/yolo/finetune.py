from ultralytics import YOLO
import wandb
wandb.login(key="")
wandb.init(project="YOLO")
# Load a pretrained YOLO model
model = YOLO("yolo11x.pt")

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
    lr0=0.01232,
    lrf=0.01262,
    momentum=0.85567,
    weight_decay=0.00032,
    warmup_epochs=2.38886,
    warmup_momentum=0.36637,
    box=9.58948,
    cls=0.31725,
    dfl=1.35851,
    hsv_h=0.,#0.01567,
    hsv_s=0.,#0.75726,
    hsv_v=0.,#0.31022,
    degrees=0.0,
    translate=0.,#0.09849,
    scale=0.,#0.6056,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.,#0.34339,
    bgr=0.0,
    mosaic=0.,#1.0,
    mixup=0.,
    copy_paste=0.,
    device = [0,1,2,3,4,5,6,7]
)
