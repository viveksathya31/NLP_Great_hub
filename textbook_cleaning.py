import re

INPUT_FILE = "merged_textbook.pdf"
OUTPUT_FILE = "cleaned_chunks.txt"

CHUNK_SIZE = 500 

print("Loading textbook...")

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

print("Textbook loaded successfully.")
print("Total characters:", len(text))


print("Cleaning text...")

# remove non-ascii characters
text = re.sub(r"[^\x00-\x7F]+", " ", text)

# remove page numbers
text = re.sub(r"\n\d+\n", "\n", text)

# remove urls
text = re.sub(r"http\S+", "", text)

# remove references like [1]
text = re.sub(r"\[\d+\]", "", text)

# remove multiple spaces
text = re.sub(r"\s+", " ", text)

print("Cleaning complete.")


print("Splitting text into chunks...")

words = text.split()

chunks = []

for i in range(0, len(words), CHUNK_SIZE):
    chunk = " ".join(words[i:i + CHUNK_SIZE])
    chunks.append(chunk)

print("Total chunks created:", len(chunks))


print("Saving chunks...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n\n")

print("Chunks saved to:", OUTPUT_FILE)


print("\n========== SUMMARY ==========")
print("Total words:", len(words))
print("Chunks created:", len(chunks))
print("=============================")