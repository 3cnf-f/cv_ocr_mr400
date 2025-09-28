from paddleocr import PaddleOCR
import re

# Initialize PaddleOCR with lightweight models and angle classification
ocr = PaddleOCR(
    use_angle_cls=True,                    # Enable angled text detection
    lang='en',
    det_model_dir='ch_PP-OCRv3_det_infer', # Light detection model (~2.3MB)
    rec_model_dir='ch_PP-OCRv3_rec_infer' # Light recognition model (~5.9MB)
)

# Run OCR inference on a sample image
result = ocr.ocr("test_screen.png", cls=True)

# Extract and output text
all_text = []
temperatures = []

print("=== OCR Results ===")
for idx in range(len(result)):
    res = result[idx]
    if res:  # Check if results exist
        for line in res:
            text = line[1][0]
            confidence = line[1][1]
            
            print(f"Text: {text} (Confidence: {confidence:.2f})")
            all_text.append(text)
            
            # Look for temperature patterns
            temp_pattern = r'\d{2}\.\d°?C?'
            matches = re.findall(temp_pattern, text)
            if matches:
                temperatures.extend(matches)

# Output summary
print("\n=== Text Summary ===")
print("All detected text:", " | ".join(all_text))

if temperatures:
    print(f"Found temperatures: {temperatures}")
else:
    print("No temperature readings found")

# Save to text file
with open("ocr_output.txt", "w") as f:
    f.write("OCR Results:\n")
    f.write("\n".join(all_text))
    f.write(f"\n\nTemperatures found: {temperatures}")

print("\nResults saved to 'ocr_output.txt'")
