import pytesseract
import cv2
import logging

logging = logging.getLogger(__name__)
pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def extract_temperature(image, threshold=150, maxval=255, thresh_type=0, psm=6, whitelist='0123456789.', crop_roi=None, morph_op=None, kernel_size=3, invert=False, resize_factor=2.0):
    # Optional crop to focus on temperature area (bottom-left, e.g., [x, y, w, h])
    if crop_roi:
        x, y, w, h = crop_roi
        cropped = image[y:y+h, x:x+w]
    else:
        cropped = image

    # Resize for better OCR (upscale)
    if resize_factor > 1.0:
        cropped = cv2.resize(cropped, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Optional Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply threshold
    _, thresh = cv2.threshold(gray, threshold, maxval, thresh_type)

    # Optional morphological operation to separate stuck digits
    if morph_op:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if morph_op == 'erode':
            thresh = cv2.erode(thresh, kernel, iterations=1)
        elif morph_op == 'dilate':
            thresh = cv2.dilate(thresh, kernel, iterations=1)
        elif morph_op == 'open':
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        elif morph_op == 'close':
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Optional invert (for white text on black background to black on white)
    if invert:
        thresh = cv2.bitwise_not(thresh)

    # Build Tesseract config
    config = f'--oem 3 --psm {psm}'
    if whitelist:
        config += f' -c tessedit_char_whitelist={whitelist}'

    # OCR
    cv2.imwrite(f'th{thresh_type}_{psm}_{crop_roi}_{morph_op}_{kernel_size}_{invert}_{resize_factor}.png', thresh)
    text = pytesseract.image_to_string(thresh, config=config)
    logging.info(text)

    # Search for temperature pattern (e.g., "32.0" or "31.8")
    import re
    match = re.search(r'(\d+\.\d)', text)
    if match:
        temp = match.group(1)
        if len(temp.split('.')[1]) == 1:  # Single decimal place
            return f"{temp}°C"

    return "Temperature not found"

# Example usage with experimentation
if __name__ == "__main__":
    image_path = "test_screen.png"  # Replace with your actual image path
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
    else:
        h, w = image.shape[:2]
        # Example crop: bottom 200px height, left 400px width (adjust based on your images)
        example_crop = (0, h - 200, 400, 200)

        # Experiment with parameters
        thresh_types = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]
        psms = [6, 7, 8, 10, 11]  # Added 11: single line
        morph_ops = [None, 'erode', 'open']
        inverts = [True, False]
        kernel_sizes = [1, 2, 3]
        resize_factors = [1.5, 2.0, 3.0]

        for thresh_type in thresh_types:
            for psm_val in psms:
                for morph in morph_ops:
                    for inv in inverts:
                        for ksize in kernel_sizes:
                            for rf in resize_factors:
                                temp = extract_temperature(
                                    image, 
                                    thresh_type=thresh_type, 
                                    psm=psm_val, 
                                    crop_roi=example_crop, 
                                    morph_op=morph, 
                                    kernel_size=ksize, 
                                    invert=inv, 
                                    resize_factor=rf
                                )
                                print(f"Type: {thresh_type}, PSM: {psm_val}, Morph: {morph}, Kernel: {ksize}, Invert: {inv}, Resize: {rf} -> {temp}")
