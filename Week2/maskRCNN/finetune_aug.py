from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader
import os
import torch

# ----- MODIFY THESE VARIABLES -----
DATASET_NAME = "c03_10"
DATASET_PATH = "./dataset"
TRAIN_JSON = "train_gt2.json"
VAL_JSON = "eval_gt2.json"
IMAGES_PATH = os.path.join(DATASET_PATH, "dataset")

# ----- REGISTER DATASET -----
register_coco_instances(DATASET_NAME + "_train", {}, TRAIN_JSON, IMAGES_PATH)
register_coco_instances(DATASET_NAME + "_val", {}, VAL_JSON, IMAGES_PATH)

# ----- AUGMENTATION -----
augs = [
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),   
    T.RandomRotation(angle=[-30, 30]),  
    T.RandomApply(T.Resize((800, 800)), prob=0.5),
    T.RandomBrightness(0.8, 1.2),  
    T.RandomContrast(0.8, 1.2),    
    T.RandomSaturation(0.5, 1.5)
]

# ----- CUSTOM DATASET MAPPER (WITH MASK AUGMENTATION) -----
class CustomDatasetMapper(DatasetMapper):
    def __init__(self, is_train: bool):
        super().__init__(
            is_train=is_train, 
            augmentations=augs, 
            use_instance_mask=True,  # Ensures masks are used
            image_format="BGR"
        )

# ----- CONFIGURE MODEL -----
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = (DATASET_NAME + "_train",)
cfg.DATASETS.TEST = (DATASET_NAME + "_val",)
cfg.DATALOADER.NUM_WORKERS = 4

# Load COCO-pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Car and Pedestrian

# Training settings
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.MAX_ITER = 2500
cfg.SOLVER.STEPS = []

# Evaluation period
cfg.TEST.EVAL_PERIOD = 500  

# ----- TRAINER CONFIGURATION -----
class TrainerWithMaskAug(DefaultTrainer):
    def build_train_loader(self, cfg):
        return build_detection_train_loader(cfg, mapper=CustomDatasetMapper(is_train=True))

trainer = TrainerWithMaskAug(cfg)
trainer.resume_or_load(resume=False)

# Freeze the backbone
model = trainer.model
for param in model.backbone.parameters():
    param.requires_grad = False

# Fine-tune ROI heads (including mask heads)
for param in model.roi_heads.parameters():
    param.requires_grad = True

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer.model.to(device)

# ----- EVALUATOR SETUP -----
evaluator = COCOEvaluator(DATASET_NAME + "_val", cfg, False, output_dir="./output_finet/")

# ----- TRAIN MODEL -----
trainer.train()

# ----- EVALUATE THE MODEL -----
trainer.test(cfg, model, evaluators=[evaluator])

# ----- SAVE MODEL -----
torch.save(cfg, "mask_rcnn_finetuned.pth")
print("Model saved!")