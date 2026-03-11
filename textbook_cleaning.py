import re
import fitz  

INPUT_FILE = "merged_textbook.pdf"
OUTPUT_FILE = "cleaned_chunks.txt"
CHUNK_SIZE = 500 

print("Loading textbook...")

# Extract text from PDF using PyMuPDF
text = ""
try:
    with fitz.open(INPUT_FILE) as doc:
        for page in doc:
            text += page.get_text()
    print("Textbook loaded successfully.")
    print("Total characters:", len(text))
except Exception as e:
    print(f"Error loading PDF: {e}")
    exit()

print("Cleaning text...")

# Remove non-ascii characters
text = re.sub(r"[^\x00-\x7F]+", " ", text)

# Remove standalone page numbers (improved regex)
text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

# Remove urls
text = re.sub(r"http\S+", "", text)

# Remove references like [1]
text = re.sub(r"\[\d+\]", "", text)

# Remove multiple spaces and clean up newlines
text = re.sub(r"\s+", " ", text).strip()

print("Cleaning complete.")

print("Splitting text into chunks...")

words = text.split()
chunks = []

# Your current approach: strict word-count chunking
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