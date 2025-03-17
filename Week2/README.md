# Week 1
## Presentation
* The slides of the project can be found [here](https://docs.google.com/presentation/d/1_Z6B1suaqjK-kdclQK2QIehBfXtvoAQn_MPRYKhfwC4/edit#slide=id.g33e8f63fe75_0_1026).
## Object detection
### Fast R CNN



### YOLO


### Mask2Former
* To execute the Mask2Former inference, from the pretrained COCO dataset, execute this command:
```
python inference.py --metric --draw
```

* The finetuning is done exectuing this script:
```
python finetune.py
```

* Once fine tuned we can execute the following script to compute metrics and draw the boxes of the predictions:
```
python inference.py --draw --metric --finetune --json_output "inference_finetune_mask2former.json" --checkpoint "finetune_mask2former" --output_drawings "./finetune_eval/"
```
## Domain shift finetuning
Domain Shift Fine-tuning for Week 2 was done at the following [colab](?¿)

