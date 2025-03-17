from ultralytics import YOLO
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import ops
import torch

class ModifiedYOLOSegValidator(SegmentationValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

    def _prepare_pred(self, pred, pbatch, proto):
        """
        Prepare predictions for evaluation by processing bounding boxes and masks.

        Args:
            pred (torch.Tensor): Raw predictions from the model.
            pbatch (Dict): Prepared batch data.
            proto (torch.Tensor): Prototype masks for segmentation.

        Returns:
            predn (torch.Tensor): Processed bounding box predictions.
            pred_masks (torch.Tensor): Processed mask predictions.
        """
        predn, pred_masks = super()._prepare_pred(pred, pbatch, proto)
        coco2kittimots = {2.:1., 0:0.}
        predcls = predn[:, 5].cpu()
        for i,c in enumerate(predcls):
            if c.item() in coco2kittimots.keys():
                predn[i, 5] = coco2kittimots[c.item()]
            else:
                predn[i, 5] = 79.
        predn = torch.Tensor(predn)
        return predn, pred_masks

model = YOLO("yolo11l-seg.pt")
model.model.names = {
    0: "person",
    1: "car",
    **{i: f"class_{i}" for i in range(2, 80)}
}

# metrics = model.val(data="kitti_mots.yaml", validator = ModifiedYOLOSegValidator)
metrics = model.val(data="kitti_mots.yaml", validator = ModifiedYOLOSegValidator, workers=64, batch= 32*8, device = [0,1,2,3,4,5,6,7])
print(metrics.box.map)  # map50-95(B)
print(metrics.box.map50)  # map50(B)
print(metrics.box.map75)  # map75(B)
print(metrics.box.maps)  # a list contains map50-95(B) of each category
print(metrics.seg.map)  # map50-95(M)
print(metrics.seg.map50)  # map50(M)
print(metrics.seg.map75)  # map75(M)
print(metrics.seg.maps)  # a list contains map50-95(M) of each category