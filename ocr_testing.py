import pytesseract
import cv2
import logging
import os

logging = logging.getLogger(__name__)
pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def extract_temperature(image, threshold=150, maxval=255, thresh_type=0, psm=6, whitelist='0123456789.', crop_roi=None):
    # Optional crop to focus on temperature area (bottom-left, e.g., [h-150:h, 0:300])
    if crop_roi:
        x, y, w, h = crop_roi
        image = image[y:y+h, x:x+w]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not os.path.exists('gray.png'):
        cv2.imwrite('gray.png', gray)

    # Optional Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if not os.path.exists('gray_gauss.png'):
        cv2.imwrite('gray_gauss.png', gray)

    # Apply threshold
    _, thresh = cv2.threshold(gray, threshold, maxval, thresh_type)

    # Optional invert if needed (after thresholding)
    # thresh = cv2.bitwise_not(thresh)
    ifcrop=True if crop_roi else False
    
    if not os.path.exists(f'thres{threshold}-{maxval}-{thresh_type}-{ifcrop}.png'):
        cv2.imwrite(f'thres{threshold}-{maxval}-{thresh_type}-{ifcrop}.png', thresh)

    # Build Tesseract config
    config = f'--oem 3 --psm {psm}'
    if whitelist:
        config += f' -c tessedit_char_whitelist={whitelist}'

    # OCR
    text = pytesseract.image_to_string(thresh, config=config)
    logging.info(text)

    # Search for temperature pattern (e.g., "32.0" or "31.8")
    for line in text.split('\n'):
        if '.' in line and any(c.isdigit() for c in line):
            temp = ''.join(filter(lambda x: x.isdigit() or x == '.', line.strip()))
            if temp and '.' in temp and len(temp.split('.')[1]) == 1:
                return f"{temp}°C",text

    return "Temperature not found",text

# Example usage with experimentation
if __name__ == "__main__":
    image_path = "test_screen.png"  # Replace with your actual image path
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
    else:
        h, w = image.shape[:2]
        # Example crop: bottom 150px height, left 300px width (adjust based on your images)
        example_crop = (0, h - 150, 300, 150)

        # Experiment with parameters
        thresh_types = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU + cv2.THRESH_BINARY, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV]
        psms = [6, 7, 8, 10]  # 6: block, 7: line, 8: word, 10: sparse
        for thresh_type in thresh_types:
            for psm_val in psms:
                # With crop
                temp_crop_a,temp_crop_b = extract_temperature(image, thresh_type=thresh_type, psm=psm_val, crop_roi=example_crop)
                print(f"Type: {thresh_type}, PSM: {psm_val}, Crop: Yes -> result: {temp_crop_a} -> raw: {temp_crop_b}")
                
                # Without crop
                temp_no_crop_a,temp_no_crop_b = extract_temperature(image, thresh_type=thresh_type, psm=psm_val)
                print(f"Type: {thresh_type}, PSM: {psm_val}, Crop: No -> result: {temp_no_crop_a} -> raw: {temp_no_crop_b}")
