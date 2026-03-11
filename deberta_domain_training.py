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
# CONFIGURATION
# ==========================================

MODEL_NAME = "microsoft/deberta-v3-base"
DATA_FILE = "cleaned_chunks.txt"
OUTPUT_DIR = "./deberta_textbook_model"

MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 2


# ==========================================
# LOAD DATA
# ==========================================

print("Loading textbook chunks...")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip() != ""]

dataset = Dataset.from_dict({"text": lines})

print("Total training samples:", len(dataset))


# ==========================================
# LOAD TOKENIZER
# ==========================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)


# ==========================================
# TOKENIZATION
# ==========================================

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


print("Tokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading DeBERTa model...")

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)


# ==========================================
# DATA COLLATOR (Masked LM)
# ==========================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)


# ==========================================
# TRAINING ARGUMENTS
# ==========================================

# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     overwrite_output_dir=True,
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     save_steps=1000,
#     save_total_limit=2,
#     logging_steps=100,
#     learning_rate=5e-5,
#     weight_decay=0.01,
#     fp16=False
# )

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
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
# TRAIN MODEL
# ==========================================

print("\nStarting domain training...")

trainer.train()


# ==========================================
# SAVE MODEL
# ==========================================

print("\nSaving trained model...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining complete!")