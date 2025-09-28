import pytesseract
import cv2
from PIL import Image, ImageEnhance

# Path to Tesseract if needed (uncomment and adjust if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image with OpenCV
image_path = 'output.png'  # Replace with your actual file path
img_cv = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise (adjust kernel size if needed)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Apply adaptive thresholding for better handling of varying lighting/glare
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Convert to PIL for further enhancement
img_pil = Image.fromarray(thresh)

# Increase contrast
enhancer = ImageEnhance.Contrast(img_pil)
img_enhanced = enhancer.enhance(2.5)  # Higher factor for dark text on dark bg

# Optional: Sharpen the image
sharpener = ImageEnhance.Sharpness(img_enhanced)
img_sharpened = sharpener.enhance(1.5)

# OCR config: Use LSTM engine and treat as sparse text (PSM 11 for individual words/characters)
custom_config = r'--oem 3 --psm 11'

# Extract text
extracted_text = pytesseract.image_to_string(img_sharpened, config=custom_config)

# Print the result
print(extracted_text)
