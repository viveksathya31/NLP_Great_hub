import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_PATH = "./deberta_textbook_model"

print("Loading trained model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)

model.eval()

print("Model loaded successfully!")

text = "Machine learning is a branch of [MASK] intelligence."

inputs = tokenizer(text, return_tensors="pt")

mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

mask_logits = logits[0, mask_index, :]

top_tokens = torch.topk(mask_logits, 10, dim=1).indices[0].tolist()

print("\nTop predictions:")

for token_id in top_tokens:
    word = tokenizer.convert_ids_to_tokens(token_id)
    print(word)