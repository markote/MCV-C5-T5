from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import os
import torch
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader

# ----- MODIFY THESE VARIABLES -----
DATASET_NAME = "c03_10"
DATASET_PATH = "./dataset"
TRAIN_JSON = "train_gt_final.json"
VAL_JSON = "eval_gt_final.json"
IMAGES_PATH = os.path.join(DATASET_PATH, "dataset")

# ----- REGISTER DATASET -----
register_coco_instances(DATASET_NAME + "_train", {}, TRAIN_JSON, IMAGES_PATH)
register_coco_instances(DATASET_NAME + "_val", {}, VAL_JSON, IMAGES_PATH)

augs = [
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),   
    T.RandomRotation(angle=[-30, 30]),  
    T.RandomApply(T.Resize((800, 800)), prob=0.5),
    T.RandomBrightness(0.8, 1.2),  
    T.RandomContrast(0.8, 1.2),    
    T.RandomSaturation(0.5, 1.5),  

]

# ----- CUSTOM DATASET MAPPER -----
class AlbumentationsMapper(DatasetMapper):
    def __init__(self, is_train: bool, augmentations=None, image_format="BGR"):
        super().__init__(is_train=is_train, augmentations=augmentations, image_format=image_format)

# ----- CONFIGURE MODEL -----
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = (DATASET_NAME + "_train",)
cfg.DATASETS.TEST = (DATASET_NAME + "_val",)
cfg.DATALOADER.NUM_WORKERS = 4

# Load COCO-pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  

# Training settings
cfg.SOLVER.IMS_PER_BATCH = 32
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.MAX_ITER = 2500
cfg.SOLVER.STEPS = []

# Set batch size for evaluation
cfg.TEST.EVAL_PERIOD = 20000  

# ----- TRAINER CONFIGURATION -----
class TrainerWithAlbumentations(DefaultTrainer):
    def build_train_loader(self, cfg):
        return build_detection_train_loader(cfg, mapper=AlbumentationsMapper(is_train=True, augmentations=augs))

trainer = TrainerWithAlbumentations(cfg)
trainer.resume_or_load(resume=False)

# Freeze the backbone
model = trainer.model
for param in model.backbone.parameters():
    param.requires_grad = False

# Fine-tune ROI heads
for param in model.roi_heads.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer.model.to(device)

# ----- EVALUATOR SETUP -----
evaluator = COCOEvaluator(DATASET_NAME + "_val", cfg, False, output_dir="./output/")

# ----- TRAIN MODEL -----
trainer.train()

# ----- EVALUATE THE MODEL -----
trainer.test(cfg, model, evaluators=[evaluator])

# ----- SAVE MODEL -----
torch.save(cfg, "faster_rcnn_finetuned.pth")
print("Model saved!")