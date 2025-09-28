# Import necessary libraries
import pytesseract
from PIL import Image
import cv2
import numpy as np

# === SETUP INSTRUCTIONS ===
# 1. Install Tesseract OCR engine:
#    - Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
#    - Mac: brew install tesseract
#    - Linux: sudo apt-get install tesseract-ocr

# 2. Install Python packages:
#    pip install pytesseract pillow opencv-python

# 3. Configure Tesseract path (Windows only)
# Uncomment and modify the line below if on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# === BASIC TEXT DETECTION ===
def detect_text_simple(image_path):
    """
    Simple text detection from an image file
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Extracted text as string
    """
    # Open the image using PIL
    image = Image.open(image_path)
    
    # Extract text using pytesseract
    text = pytesseract.image_to_string(image)
    
    return text

# === ADVANCED TEXT DETECTION WITH PREPROCESSING ===
def detect_text_advanced(image_path, preprocess=True):
    """
    Advanced text detection with image preprocessing for better accuracy
    
    Args:
        image_path: Path to the image file
        preprocess: Whether to apply preprocessing
    
    Returns:
        Extracted text and additional data
    """
    # Load image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert to grayscale (improves OCR accuracy)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if preprocess:
        # Apply thresholding to get better contrast
        # This converts image to pure black and white
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Optional: Remove noise using morphological operations
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    else:
        processed = gray
    
    # Configure OCR options
    custom_config = r'--oem 3 --psm 6'
    # OEM (OCR Engine Mode):
    #   0 = Original Tesseract only
    #   1 = Neural nets LSTM only
    #   2 = Tesseract + LSTM
    #   3 = Default (automatic)
    
    # PSM (Page Segmentation Mode):
    #   0 = Orientation and script detection only
    #   1 = Automatic page segmentation with OSD
    #   3 = Fully automatic (no OSD)
    #   6 = Uniform block of text (good for most cases)
    #   7 = Single text line
    #   8 = Single word
    #   11 = Sparse text
    #   12 = Sparse text with OSD
    
    # Extract text
    text = pytesseract.image_to_string(processed, config=custom_config)
    
    # Get detailed data including bounding boxes
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
    
    return text, data

# === GET BOUNDING BOXES FOR DETECTED TEXT ===
def get_text_boxes(image_path, confidence_threshold=60):
    """
    Get bounding boxes for detected text regions
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score (0-100)
    
    Returns:
        List of dictionaries with text and coordinates
    """
    # Load image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Get detailed OCR data
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    # Extract text boxes with confidence scores
    text_boxes = []
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        # Filter by confidence score
        if int(data['conf'][i]) > confidence_threshold:
            # Get coordinates
            (x, y, w, h) = (data['left'][i], data['top'][i], 
                           data['width'][i], data['height'][i])
            
            # Get text
            text = data['text'][i].strip()
            
            # Only include non-empty text
            if text:
                text_boxes.append({
                    'text': text,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'confidence': data['conf'][i]
                })
    
    return text_boxes

# === DETECT TEXT IN SPECIFIC LANGUAGES ===
def detect_text_multilingual(image_path, languages='eng+fra+deu'):
    """
    Detect text in multiple languages
    
    Args:
        image_path: Path to the image file
        languages: Language codes separated by '+' 
                  (e.g., 'eng' for English, 'fra' for French, 'deu' for German)
    
    Returns:
        Extracted text
    """
    # Load image
    image = Image.open(image_path)
    
    # Specify languages in the config
    custom_config = f'-l {languages}'
    
    # Extract text
    text = pytesseract.image_to_string(image, config=custom_config)
    
    return text

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # Example 1: Simple text extraction
    image_file = "output.png"
    
    # Basic extraction
    simple_text = detect_text_simple(image_file)
    print("Simple extraction:", simple_text)
    
    # Advanced extraction with preprocessing
    advanced_text, ocr_data = detect_text_advanced(image_file)
    print("Advanced extraction:", advanced_text)
    
    # Get bounding boxes
    boxes = get_text_boxes(image_file)
    for box in boxes:
        print(f"Found '{box['text']}' at ({box['x']}, {box['y']}) "
              f"with confidence {box['confidence']}%")
    
    # Multilingual detection
    # multilingual_text = detect_text_multilingual(image_file, 'eng+spa')
    # print("Multilingual:", multilingual_text)

# === ADDITIONAL TIPS ===
"""
1. Image Quality Tips:
   - Higher resolution images give better results
   - Ensure good contrast between text and background
   - Straight, non-skewed text works best
   - Clean, noise-free images improve accuracy

2. Performance Optimization:
   - Crop to region of interest before OCR
   - Use appropriate PSM mode for your use case
   - Consider using parallel processing for multiple images

3. Common Issues and Solutions:
   - Poor accuracy: Try preprocessing (threshold, denoise, deskew)
   - Missing text: Adjust PSM mode or confidence threshold
   - Slow performance: Reduce image size or use specific regions
   - Special characters: Ensure proper language packs are installed

4. Advanced Features:
   - pytesseract.image_to_pdf_or_hocr(): Generate searchable PDFs
   - pytesseract.image_to_osd(): Get orientation and script detection
   - Custom trained models for specific fonts/languages
"""
