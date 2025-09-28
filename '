import os
import json
import re
import sys
from pathlib import Path
import cv2  # Import OpenCV for image preprocessing
import numpy as np
from paddleocr import PaddleOCR

# --- Configuration ---
# It's good practice to have configurations at the top
# This makes it easier to change parameters later

# Input folder from command-line arguments
if len(sys.argv) > 1:
    input_folder = sys.argv[1]
else:
    print("Usage: python your_script_name.py <input_folder>")
    sys.exit(1)

# OCR Model Configuration
OCR_CONFIG = {
    "text_detection_model_name": "PP-OCRv5_server_det",
    "text_recognition_model_name": "PP-OCRv5_server_rec",
    "device": "cpu",
    "enable_mkldnn": True,
    "cpu_threads": os.cpu_count() or 8,  # Use available cores
    "mkldnn_cache_capacity": 10
}

# --- Image Preprocessing Function ---
def preprocess_image(image_path):
    """
    Loads an image and applies preprocessing steps to improve OCR accuracy.
    - Converts to grayscale
    - Applies a bilateral filter to reduce noise while keeping edges sharp
    - Uses adaptive thresholding to handle varying lighting conditions
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image at {image_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a bilateral filter to reduce noise while preserving edges
        # This is often better than a simple Gaussian blur for screen images
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Use adaptive thresholding to create a binary image (black and white)
        # This helps with variations in screen brightness and contrast
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

# --- Main Processing Logic ---
def main():
    print("Initializing PaddleOCR...")
    try:
        ocr = PaddleOCR(**OCR_CONFIG)
        print("PaddleOCR initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize PaddleOCR: {e}")
        sys.exit(1)

    all_screen_data = []

    # Ensure the input folder exists
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    # Process files in a sorted order for consistency
    files_to_process = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
    if not files_to_process:
        print(f"No .png files found in '{input_folder}'.")
        return

    print(f"Found {len(files_to_process)} PNG files to process.")

    for filename in files_to_process:
        # Improved regex for more robust matching
        match = re.search(r"output_frame_(\d+)_screen_([1-3])\.png", filename)
        
        if match:
            frame_number = match.group(1)
            screen_number = match.group(2)
            image_path = os.path.join(input_folder, filename)
            
            print(f"\nProcessing: {filename} (Frame: {frame_number}, Screen: {screen_number})")

            # --- Preprocess the image before OCR ---
            preprocessed_image = preprocess_image(image_path)

            if preprocessed_image is not None:
                try:
                    # Run OCR on the preprocessed image
                    result = ocr.ocr(preprocessed_image, cls=True)
                    
                    if result and result[0]:
                        # Extract text and confidence scores
                        texts = [line[1][0] for line in result[0]]
                        scores = [line[1][1] for line in result[0]]

                        this_temp = 0.0
                        this_conf = 0.0

                        # Find the best temperature reading based on regex and confidence
                        for i, text in enumerate(texts):
                            # Regex to find numbers like 32.5, 123.0, etc.
                            temp_match = re.match(r'^\d{1,3}\.\d$', text)
                            if temp_match and scores[i] > this_conf:
                                try:
                                    this_temp = float(temp_match.group(0))
                                    this_conf = scores[i]
                                except ValueError:
                                    continue # In case of conversion error

                        screen_data = {
                            "file": filename,
                            "frame": frame_number,
                            "screen": screen_number,
                            "all_text": texts,
                            "temperature": this_temp,
                            "temperature_confidence": round(this_conf, 4)
                        }

                        print(f"  - Detected Temperature: {screen_data['temperature']} with confidence {screen_data['temperature_confidence']}")
                        all_screen_data.append(screen_data)
                    else:
                        print(f"  - No text detected in {filename}")

                except Exception as e:
                    print(f"  - An error occurred during OCR for {filename}: {e}")

    # --- Save results to JSON ---
    output_filename = "output.json"
    print(f"\nSaving results to {output_filename}...")
    try:
        with open(output_filename, "w") as outfile:
            json.dump(all_screen_data, outfile, indent=4)
        print("Successfully saved results.")
    except IOError as e:
        print(f"Error writing to {output_filename}: {e}")

if __name__ == "__main__":
    main()
