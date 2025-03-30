import torch
from transformers import ViTImageProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from unsloth import FastLanguageModel
import wandb
import time
import torch.nn as nn
import evaluate

# Initialize WandB
wandb.init(project="unsloth_vision_text", name=f"run_{time.strftime('%Y%m%d_%H%M%S')}")

# Step 1: Pre-Extract Vision Features with Best ViT Checkpoint
vit_model_name = "nlpconnect/vit-gpt2-image-captioning"
vit_checkpoint_path = "./LORA/best_vitgpt2.pt"
processor = ViTImageProcessor.from_pretrained(vit_model_name)
vit_model = VisionEncoderDecoderModel.from_pretrained(vit_model_name).encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading ViT checkpoint from: {vit_checkpoint_path}")
checkpoint = torch.load(vit_checkpoint_path, map_location=device)
vit_state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items() if "encoder" in k}
vit_model.load_state_dict(vit_state_dict, strict=False)
vit_model.eval()
vit_model.to(device)

image_dir = "/mnt/dataset/image_captioning_dataset/FoodImages"
splits_path = "FilteredDataSplit.npy"
partitions = np.load(splits_path, allow_pickle=True).item()

def extract_vit_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
    return outputs.last_hidden_state.cpu().numpy()  # Shape: (1, 197, 768)

for split in ["train", "eval", "test"]:
    if not os.path.exists(f"{split}_features.npy"):
        features = []
        captions = []
        for caption, image_path in partitions[split]:
            full_path = os.path.join(image_dir, image_path)
            feature = extract_vit_features(full_path)
            features.append(feature)
            captions.append(caption)
        np.save(f"{split}_features.npy", np.concatenate(features, axis=0))
        np.save(f"{split}_captions.npy", np.array(captions, dtype=object))
    print(f"{split} features shape: {np.load(f'{split}_features.npy').shape}")

# Step 2: Prepare Dataset (Unsloth-Style, CPU Tensors)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-bnb-4bit",
    max_seq_length=256,
    load_in_4bit=True,
)
tokenizer.pad_token = tokenizer.eos_token

# Define a simple prompt template
PROMPT = "### Instruction: Generate a caption for the image: ### Response: "

class VisionTextDataset(Dataset):
    def __init__(self, features_file, captions_file, tokenizer, projection, max_length=256):
        self.features = np.load(features_file)  # Shape: (num_samples, 197, 768)
        self.captions = np.load(captions_file, allow_pickle=True)
        self.tokenizer = tokenizer
        self.projection = projection
        self.max_length = max_length
        self.feature_seq_len = self.features.shape[1]  # e.g., 197

        # Tokenize the prompt once (on CPU)
        self.prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].squeeze(0)[:-1]  # Remove <eos>
        self.prompt_len = self.prompt_ids.size(0)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # Vision features to embeddings (on CPU initially)
        vision_features = torch.tensor(self.features[idx], dtype=torch.float32)
        projected_features = self.projection(vision_features.to("cuda:0")).cpu()  # Project on GPU, then back to CPU
        
        # Caption (on CPU)
        caption = self.captions[idx]
        caption_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length - self.feature_seq_len - self.prompt_len,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        # Combine into a single sequence: [prompt_ids, dummy_vision_tokens, caption_ids]
        dummy_vision_tokens = torch.full((self.feature_seq_len,), self.tokenizer.pad_token_id, dtype=torch.long)
        input_ids = torch.cat([self.prompt_ids, dummy_vision_tokens, caption_ids]).contiguous()

        # Attention mask: 1s for prompt and caption, 0s for vision placeholder
        attention_mask = torch.cat([
            torch.ones(self.prompt_len, dtype=torch.long),
            torch.zeros(self.feature_seq_len, dtype=torch.long),
            torch.ones(caption_ids.size(0), dtype=torch.long)
        ]).contiguous()

        # Labels: -100 for prompt and vision, caption_ids for caption
        labels = torch.cat([
            torch.full((self.prompt_len,), -100, dtype=torch.long),
            torch.full((self.feature_seq_len,), -100, dtype=torch.long),
            caption_ids
        ]).contiguous()

        # Replace input_ids with projected features for vision part (on CPU)
        text_embeds = model.get_input_embeddings()(input_ids.to("cuda:0")).cpu()
        inputs_embeds = text_embeds.clone()
        inputs_embeds[self.prompt_len:self.prompt_len + self.feature_seq_len] = projected_features

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Step 3: Fine-Tune with Unsloth and WandB (With Checkpointing)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Create projection layer on cuda:0
projection = nn.Linear(768, model.config.hidden_size).to("cuda:0")

train_dataset = VisionTextDataset("train_features.npy", "train_captions.npy", tokenizer, projection)
eval_dataset = VisionTextDataset("eval_features.npy", "eval_captions.npy", tokenizer, projection)
test_dataset = VisionTextDataset("test_features.npy", "test_captions.npy", tokenizer, projection)

training_args = TrainingArguments(
    output_dir="./unsloth_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",           # Save checkpoints every few steps
    save_steps=50,                  # Save every 50 steps
    load_best_model_at_end=True,    # Load the best model at the end
    metric_for_best_model="eval_loss",  # Use evaluation loss to pick the best model
    greater_is_better=False,        # Lower eval_loss is better
    gradient_accumulation_steps=2,
    fp16=True,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=lambda data: {
        "inputs_embeds": torch.stack([d["inputs_embeds"] for d in data]),
        "attention_mask": torch.stack([d["attention_mask"] for d in data]),
        "labels": torch.stack([d["labels"] for d in data]),
    },
)

print("Starting training...")
trainer.train()

# Load metrics
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

# Step 4: Inference and Log Predictions with WandB (Using Best Model) - Modified
model.eval()
test_features = np.load("test_features.npy")
test_captions = np.load("test_captions.npy", allow_pickle=True)

predictions = []
ground_truths = []
with torch.no_grad():
    for i in range(min(10, len(test_features))):
        features = torch.tensor(test_features[i:i+1], dtype=torch.float32).to("cuda:0")  # (1, 197, 768)
        projected_features = projection(features)  # (1, 197, 2048)
        prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to("cuda:0")[:, :-1]  # Remove <eos>
        
        # Dummy input_ids for generation
        dummy_vision_tokens = torch.full((1, 197), tokenizer.pad_token_id, dtype=torch.long).to("cuda:0")
        input_ids = torch.cat([prompt_ids, dummy_vision_tokens], dim=1)
        text_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds = text_embeds.clone()
        inputs_embeds[:, prompt_ids.size(1):] = projected_features

        import pdb; pdb.set_trace()
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=50,
            num_beams=4,
            pad_token_id=tokenizer.pad_token_id
        )
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(PROMPT, "").strip()
        predictions.append(caption)
        ground_truths.append(test_captions[i])
        print(f"Ground Truth: {test_captions[i]}")
        print(f"Generated: {caption}\n")

# Compute metrics
bleu_score = bleu.compute(predictions=predictions, references=ground_truths)
meteor_score = meteor.compute(predictions=predictions, references=ground_truths)
rouge_score = rouge.compute(predictions=predictions, references=ground_truths)

# Log predictions and metrics to WandB
wandb.log({
    "test_predictions": wandb.Table(
        columns=["Ground Truth", "Generated"],
        data=[[gt, pred] for gt, pred in zip(ground_truths, predictions)]
    ),
    "bleu_score": bleu_score["bleu"],
    "bleu_precisions": bleu_score["precisions"],  # BLEU-1, BLEU-2, BLEU-3, BLEU-4
    "meteor_score": meteor_score["meteor"],
    "rouge_scores": rouge_score  # Includes ROUGE-1, ROUGE-2, ROUGE-L
})

# Print metrics
print("Evaluation Metrics:")
print(f"BLEU Score: {bleu_score['bleu']:.4f}")
print(f"BLEU Precisions: {[round(p, 4) for p in bleu_score['precisions']]}")
print(f"METEOR Score: {meteor_score['meteor']:.4f}")
print(f"ROUGE Scores: {rouge_score}")

wandb.finish()
print("Training, inference, and evaluation complete!")