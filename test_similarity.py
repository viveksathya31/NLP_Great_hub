import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# MODEL_PATH = "./deberta_textbook_model"
MODEL_PATH = "microsoft/deberta-v3-base"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)

model.eval()

print("Model loaded successfully!")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding


# reference answer
reference = "Machine learning is a method where computers learn patterns from data."

# student answer
student = "Machine learning allows computers to learn from data."

ref_emb = get_embedding(reference)
stu_emb = get_embedding(student)

score = cosine_similarity(ref_emb.numpy(), stu_emb.numpy())[0][0]

print("\nSimilarity score:", score)