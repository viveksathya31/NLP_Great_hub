from transformers import AutoTokenizer, AutoModelForMaskedLM

model_path = "./deberta_textbook_model"

print("Loading trained model...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)

print("Model loaded successfully!")

import torch

text = "Machine learning is a type of artificial intelligence"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.logits.shape)