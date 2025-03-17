# Week 2

## Presentation

- The slides of the project can be found [here](https://docs.google.com/presentation/d/1_Z6B1suaqjK-kdclQK2QIehBfXtvoAQn_MPRYKhfwC4/edit#slide=id.g33e8f63fe75_0_826).

## Object detection

### Mask R CNN

* finetune.py: Fine-tunes the model without augmentation.

* finetune_aug.py: Fine-tunes the model with configurable augmentation.

* coco_format_inference.py: Runs inference using a fine-tuned model and maps the classes to custom dataset labels.

* test_map.py: evaluation

### YOLO-SEG

Go to the folder "yolo-seg" and use the following command.

* Prepare dataset format for YOLO

```
python prepare_yolo_dataset.py
```

* Check the produced annotation:
```
python check_yolo_mask.py
```

* To execute the YOLO inference, from the pretrained COCO dataset, execute this command:
```
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

### Mask2Former

- To execute the Mask2Former inference, from the pretrained COCO dataset, execute this command:

```
python inference.py --metric --draw
```

- The finetuning is done exectuing this script:

```
python finetune.py
```

- Once fine tuned we can execute the following script to compute metrics and draw the boxes of the predictions:

```
python inference.py --draw --metric --finetune --json_output "inference_finetune_mask2former.json" --checkpoint "finetune_mask2former" --output_drawings "./finetune_eval/"
```

## Domain shift finetuning

Domain Shift Fine-tuning for Week 2 was done at the following [colab](https://colab.research.google.com/drive/19s8xXQVHBhcCaD2LrAUYnrAQzIym8kEJ?usp=sharing)

- [Drive Folder](https://drive.google.com/drive/folders/1vQn0S-rXAk5gnBBx8cXdHkwzjXkMJdLE?usp=sharing)
