import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"  # Add this line
os.environ["MKL_NUM_THREADS"] = "1"  # Add this line

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def extract_text_from_lines(input_dir, output_file):
    print("Loading TrOCR model (this might take a minute the first time)...")
    
    # 1. Set up hardware acceleration (MPS for Mac, CUDA for NVIDIA)
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Load the processor and the model
    # We use the 'handwritten' variant specifically tuned for human writing
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

    # 3. Get all the cropped line images and sort them alphabetically 
    # (Because we named them line_000.jpg, line_001.jpg, they will stay in reading order)
    if not os.path.exists(input_dir):
        print(f"Error: Could not find directory {input_dir}")
        return

    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print("No images found to process. Run image_processor.py first.")
        return

    print(f"Found {len(image_files)} lines of text. Starting OCR...")
    
    extracted_text = []

    # 4. Loop through each line, process it, and generate text
    for i, file_name in enumerate(image_files):
        image_path = os.path.join(input_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        
        # Prepare the image for the model
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        
        # Generate the text tokens
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            
        # Decode the tokens back into human-readable text
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Line {i+1}/{len(image_files)}: {text}")
        extracted_text.append(text)

    # 5. Save the raw output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Join the lines with a space (we will handle sentence segmentation later)
    full_text = " ".join(extracted_text)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print(f"\nSuccess! Raw OCR text saved to: {output_file}")


# --- Execute the script ---
if __name__ == "__main__":
    # Define paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input: The folder where your cropped lines are saved
    INPUT_FOLDER = os.path.abspath(os.path.join(base_dir, "../../data/phase1_interim"))
    
    # Output: A raw text file in your processed folder
    OUTPUT_TEXT_FILE = os.path.abspath(os.path.join(base_dir, "../../data/phase1_processed/raw_ocr_output.txt"))

    extract_text_from_lines(INPUT_FOLDER, OUTPUT_TEXT_FILE)