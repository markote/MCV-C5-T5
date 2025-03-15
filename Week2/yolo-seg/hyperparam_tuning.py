from ultralytics import YOLO
# Initialize the YOLO model
model = YOLO("yolo11l-seg.pt")

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="/mnt/home/MCV-C5-T5/Week2/yolo-seg/kitti_mots_finetune.yaml",
    epochs=30,
    iterations=100,
    val=False,
    plots=False,
    save=False,
    workers=64,
    batch= 32*8,
    device = [0,1,2,3,4,5,6,7] 
)