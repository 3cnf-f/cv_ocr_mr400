import pytesseract
import cv2
import numpy as np
import logging

logging = logging.getLogger(__name__)
pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def extract_temperature(image, threshold=150, other_threshold=255):
    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get image text more clear
    _, thresh = cv2.threshold(gray, threshold, other_threshold, cv2.THRESH_BINARY)

    # Use Tesseract to do OCR on the image
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    print(text)

    # Search for temperature pattern (e.g., "32.0" or "31.8")
    for line in text.split('\n'):
        if '.' in line and any(c.isdigit() for c in line):
            temp = ''.join(filter(lambda x: x.isdigit() or x == '.', line))
            if temp and '.' in temp and len(temp.split('.')[1]) == 1:  # Ensure it’s a decimal like 32.0
                return f"{temp}°C"

    return "Temperature not found"

# Example usage with experimentation
if __name__ == "__main__":
    image_path = "test_screen.png"  # Replace with your actual image path
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
    else:
        # Experiment with different threshold values
        for thresh in range(100, 201, 10):  # Example range: 100 to 200 in steps of 10
            for other_thresh in [255]:  # Typically fixed at 255 for binary, but you can add ranges
                temp = extract_temperature(image, threshold=thresh, other_threshold=other_thresh)
                print(f"Threshold: {thresh}, Other: {other_thresh} -> {temp}")
