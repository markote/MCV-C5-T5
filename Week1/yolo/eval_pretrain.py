from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
import torch

class ModifiedYOLOValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        #map class yolo to kitti
        coco2kittimots = {2.:1. , 5.:1., 7.:1., 0:0.}
        predcls = predn[:, 5].cpu()
        for i,c in enumerate(predcls):
            if c.item() in coco2kittimots.keys():
                predn[i, 5] = coco2kittimots[c.item()]
            else:
                predn[i, 5] = 79.
        predn = torch.Tensor(predn)
        return predn

model = YOLO("yolo11x.pt")
model.model.names = {
    0: "person",
    1: "car",
    **{i: f"class_{i}" for i in range(2, 80)}
}

metrics = model.val(data="kitti_mots_pretrain.yaml", validator = ModifiedYOLOValidator)
print(metrics.box.map)  # map50-95