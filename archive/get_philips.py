# Import Tesseract (needs pytesseract and tesseract-ocr installed)
import pytesseract as tess 
import cv2
tess.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Common path on Linux; check with `which tesseract`
image_path = 'output.png'
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Image not found or unable to load at path: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"Image loaded successfully. Shape: {image.shape} (height, width, channels)")
# Preprocess for OCR: Invert grayscale for white text on dark bezel (makes black text on white)
inverted_gray = cv2.bitwise_not(gray)

# Optional: Threshold to binary for better contrast (Otsu for auto)
_, binary = cv2.threshold(inverted_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save binary for debug
cv2.imwrite('binary_for_ocr.png', binary)

# Run OCR on binary with config for better detection (psm 3 for auto page segmentation, oem 3 for default engine)
config = '--psm 3 --oem 3'
text_data = tess.image_to_data(binary, output_type=tess.Output.DICT, config=config)

# Print full data for debug (to see if PHILIPS shows)
print("OCR data:", text_data)

