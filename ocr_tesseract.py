import pytesseract
import cv2
import numpy as np
import logging

logging=logging.getLogger(__name__)
pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def extract_temperature(image):
    # Load the image

    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get image text more clear
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Use Tesseract to do OCR on the image
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    logging.info(text)

    # # Search for temperature pattern (e.g., "32.0" or "31.8")
    # for line in text.split('\n'):
    #     if '.' in line and any(c.isdigit() for c in line):
    #         temp = ''.join(filter(lambda x: x.isdigit() or x == '.', line))
    #         if len(temp.split('.')[1]) == 1:  # Ensure it’s a decimal like 32.0
    #             return f"{temp}°C"
    #
    # return "Temperature not found"

