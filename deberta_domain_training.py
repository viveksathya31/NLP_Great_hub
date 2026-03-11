import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# ==========================================
# CONFIGURATION & DEVICE CHECK
# ==========================================
MODEL_NAME = "microsoft/deberta-v3-base"
DATA_FILE = "cleaned_chunks.txt"
OUTPUT_DIR = "./deberta_textbook_model"

MAX_LENGTH = 512

# Check hardware backend
if torch.cuda.is_available():
    print("Device: CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    print("Device: MPS (Apple Silicon)")
else:
    print("Device: CPU (Warning: Training will be very slow)")

# ==========================================
# LOAD DATA
# ==========================================
print("\nLoading textbook chunks...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip() != ""]

dataset = Dataset.from_dict({"text": lines})
print("Total training samples:", len(dataset))

# ==========================================
# LOAD TOKENIZER
# ==========================================
print("\nLoading tokenizer...")
# use_fast=False is often recommended for DeBERTa-v3 to avoid alignment bugs
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# ==========================================
# TOKENIZATION (Optimized)
# ==========================================
def tokenize_function(example):
    # Removed padding="max_length" to allow dynamic padding via the collator
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# ==========================================
# LOAD MODEL & COLLATOR
# ==========================================
print("\nLoading DeBERTa model...")
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15 # 15% of tokens will be masked
)

# ==========================================
# TRAINING ARGUMENTS
# ==========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, # Effective batch size = 16
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    # fp16=True, # Uncomment if using an NVIDIA GPU. MPS handles mixed precision differently.
)

# ==========================================
# TRAINER
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# ==========================================
# TRAIN & SAVE
# ==========================================
print("\nStarting domain training...")
trainer.train()

print("\nSaving trained model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete!")