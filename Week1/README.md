# Week 1

## Object detection
### Fast R CNN

finetune.py: Fine-tunes the model without augmentation.

finetune_aug.py: Fine-tunes the model with configurable augmentation.

coco_format_inference.py: Runs inference using a fine-tuned model and maps the classes to custom dataset labels.


### YOLO
### DETR
* To execute the DETR inference, from the pretrained COCO dataset, execute this command:
```
python inference.py --metric --draw
```

* The finetuning is done exectuing this script:
```
python finetune.py --metric --draw
```

* Once fine tuned we can execute the following script tocompute metrics and draw the boxes of the prediction:
```
python inference_from_finetune_drawings.py --metric #metric flag for bb of the gt
python inference_from_finetune_metric.py
```
