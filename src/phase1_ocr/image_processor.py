import cv2
import numpy as np
import os

def process_handwritten_page(image_path, output_dir):
    print(f"Loading image: {image_path}")
    
    # 1. Load image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Check the path.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Apply a threshold to invert the image (background black, text white)
    # This helps OpenCV's contour detection which looks for white objects
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # 3. Dilation: We stretch the white pixels horizontally. 
    # This merges individual words into a single solid block representing a line of text.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 5)) 
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # 4. Find the contours (the boundaries of these white blocks)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Sort contours from top to bottom (so we read the page in the correct order)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    # Sort based on the 'y' coordinate
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])

    print(f"Detected {len(bounding_boxes)} lines of text. Cropping...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 6. Crop and save each line
    line_count = 0
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Filter out tiny boxes that are likely noise/specks of dust
        if w > 50 and h > 15: 
            # Add a small margin (padding) around the text so TrOCR can read it clearly
            margin = 5
            y1 = max(0, y - margin)
            y2 = min(img.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(img.shape[1], x + w + margin)

            # Crop the original image using Numpy slicing
            roi = img[y1:y2, x1:x2]
            
            output_path = os.path.join(output_dir, f"line_{line_count:03d}.jpg")
            cv2.imwrite(output_path, roi)
            line_count += 1

    print(f"Successfully saved {line_count} cropped lines to {output_dir}")

# --- Execute the script ---
if __name__ == "__main__":
    INPUT_IMAGE = "../../data/phase1_raw/answer_sheets/test_page.jpg"
    OUTPUT_FOLDER = "../../data/phase1_interim"
    
    # Get the absolute path based on where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(base_dir, INPUT_IMAGE))
    output_path = os.path.abspath(os.path.join(base_dir, OUTPUT_FOLDER))

    process_handwritten_page(input_path, output_path)