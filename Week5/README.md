# Week 5

## Presentation

[Google Slides](https://docs.google.com/presentation/d/13SFHbBjlSCsLyBiQGgJxQK16i4kEBDKHg0LOgH-Mp_w/edit?usp=sharing)

## Code

### Generate prompts and new dataset

* prompt_generator.py: Generates the prompts by obtaining the titles of the discarded images and adding two random condition permutations.

* finetune_aug.py: Fine-tunes the model with configurable augmentation.

* coco_format_inference.py: Runs inference using a fine-tuned model and maps the classes to custom dataset labels.

* test_map.py: evaluation

### Image generationn with diffusion models
* All theexperimentation, and generation was done on the following [jupyter notebook](./stable-diffusion.ipynb)

### Execute image captioning models
* For executing the QWEN pretrained MLLM execute the code of [QWEN](../Week4/DeepSeek/multimodal_inference.py), which as been modified accordingly.

* For finetunning LLama model execute the train_lora.py script, right now configure for Llama 1B with prompt. Modify the configuration variable at the beggining of the script for Llama 3B.

