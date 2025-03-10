# Week 1
## Presentation
* The slides of the project can be found [here](https://docs.google.com/presentation/d/1SPHVbv9CpBlOdyE1i5LMe4Scj5LjWOr8vCCc4I8aot4/edit?usp=sharing).
## Object detection
### Fast R CNN

* finetune.py: Fine-tunes the model without augmentation.

* finetune_aug.py: Fine-tunes the model with configurable augmentation.

* coco_format_inference.py: Runs inference using a fine-tuned model and maps the classes to custom dataset labels.


### YOLO
Go to the folder "yolo" and use the following command.

* Prepare dataset format for YOLO

```
python prepare_yolo_dataset.py
```

* To execute the YOLO inference, from the pretrained COCO dataset, execute this command:
```
python inference_pretrain.py # 1 image
or
python inference_pretrain_gif.py # produce GIF
```

* Eval the pretrained model:

```
python eval_pretrain.py
```

* The finetuning is done exectuing this script:
```
python finetune.py
```
or we can run the hyperparameter tuning by:
```
python hyperparam_tuning.py
```

* Once fine tuned we can execute the following script to compute metrics:

```
python eval_finetune.py
```

* We also can run inference on finetuned model to produce GIF

```
python inference_finetune_gif.py # produce GIF
```  

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
