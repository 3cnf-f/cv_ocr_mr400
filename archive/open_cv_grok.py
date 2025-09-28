# Import necessary libraries
# OpenCV for image processing and contour detection
import cv2
# NumPy for numerical operations
import numpy as np
# OS for file path handling
import os

# Define constants and configurable parameters
# Input image file name as specified
INPUT_IMAGE = "output.png"
# Output folder for annotated images and log
OUTPUT_FOLDER = "screen_detection_outputs"
# Log file name
LOG_FILE = os.path.join(OUTPUT_FOLDER, "detection_log.txt")
# Expected screen area in pixels (adjusted based on image analysis; roughly 180000 for average screen)
EXPECTED_SCREEN_AREA = 140000  # Adjust this if needed after checking logs
# Tolerance for area filtering (50% to cover variations from skew)
AREA_TOLERANCE = 0.50
# Canny edge detection thresholds (lowered to catch more edges)
CANNY_LOW = 30
CANNY_HIGH = 90
# Percentage of screen height to crop from bottom for the value region
BOTTOM_CROP_PERCENT = 0.15  # 15% from bottom
# Minimum number of sides for polygon approximation (relaxed for skewed shapes)
MIN_SIDES = 4
# Dilation kernel size to close small gaps in edges
DILATION_KERNEL = np.ones((5,5), np.uint8)

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load the input image
# Read in color for annotations, but we'll convert to gray for processing
image = cv2.imread(INPUT_IMAGE)
if image is None:
    raise ValueError(f"Could not load image: {INPUT_IMAGE}")

# Log initialization
with open(LOG_FILE, "w") as log:
    log.write(f"Processing image: {INPUT_IMAGE}\n")
    log.write(f"Image dimensions: {image.shape[1]}x{image.shape[0]}\n")
    log.write(f"Expected screen area: {EXPECTED_SCREEN_AREA} (±{AREA_TOLERANCE*100}%)\n")
    log.write(f"Canny thresholds: low={CANNY_LOW}, high={CANNY_HIGH}\n\n")

# Step 1: Preprocess the image
# Convert to grayscale for edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur to reduce noise (larger kernel for better smoothing)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Step 2: Edge detection using Canny
edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

# Dilate edges to close gaps and connect contours
dilated = cv2.dilate(edges, DILATION_KERNEL, iterations=1)

# Save dilated edges image for debugging
cv2.imwrite(os.path.join(OUTPUT_FOLDER, "dilated_edges.png"), dilated)

# Step 3: Find contours on dilated edges
# Use RETR_EXTERNAL for outer contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Log total contours found
with open(LOG_FILE, "a") as log:
    log.write(f"Total contours found: {len(contours)}\n\n")

# Copy of original image for annotations
annotated = image.copy()

# List to store valid screen contours
valid_screens = []
# Counter for screen numbering
screen_count = 0

# Step 4: Filter contours by shape and size
# Log potential candidates
with open(LOG_FILE, "a") as log:
    log.write("Potential quad-like contours:\n")

potential_contours = []
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.03 * cv2.arcLength(contour, True)  # Slightly higher epsilon for approximation
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    
    # Check if roughly quadrilateral or more (for distortions)
    if len(approx) >= MIN_SIDES:
        # Use actual contour area
        area = cv2.contourArea(contour)
        
        # Log potential
        with open(LOG_FILE, "a") as log:
            log.write(f"  Contour with {len(approx)} sides, area={area}\n")
        
        potential_contours.append((area, contour))
        
        # Filter by area
        min_area = EXPECTED_SCREEN_AREA * (1 - AREA_TOLERANCE)
        max_area = EXPECTED_SCREEN_AREA * (1 + AREA_TOLERANCE)
        if min_area <= area <= max_area:
            screen_count += 1
            valid_screens.append(contour)
            
            # Get bounding rect for drawing and cropping
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw green rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Label
            cv2.putText(annotated, f"Screen {screen_count}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Log valid
            with open(LOG_FILE, "a") as log:
                log.write(f"Valid Screen {screen_count}:\n")
                log.write(f"  Bounding: x={x}, y={y}, w={w}, h={h}\n")
                log.write(f"  Area: {area}\n\n")

# Fallback: If no valid screens, take top 3 largest potential quad-like contours
if screen_count == 0 and potential_contours:
    # Sort by area descending
    sorted_potentials = sorted(potential_contours, key=lambda x: x[0], reverse=True)
    for i in range(min(3, len(sorted_potentials))):
        area, contour = sorted_potentials[i]
        screen_count += 1
        valid_screens.append(contour)
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw yellow rectangle for fallback
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(annotated, f"Fallback Screen {screen_count}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Log fallback
        with open(LOG_FILE, "a") as log:
            log.write(f"Fallback Screen {screen_count} (area={area}):\n")
            log.write(f"  Bounding: x={x}, y={y}, w={w}, h={h}\n\n")

# Save annotated screens
cv2.imwrite(os.path.join(OUTPUT_FOLDER, "annotated_screens.png"), annotated)

# Step 5: Crop bottom center for each valid screen
for idx, contour in enumerate(valid_screens, start=1):
    # Get bounding rect for cropping
    x, y, w, h = cv2.boundingRect(contour)
    
    # Bottom crop: start from (1 - percent) of height
    crop_y_start = y + int(h * (1 - BOTTOM_CROP_PERCENT))
    crop_height = int(h * BOTTOM_CROP_PERCENT)
    # Center: middle 50% width
    crop_x_start = x + int(w * 0.25)
    crop_width = int(w * 0.50)
    
    # Ensure crop doesn't exceed image bounds
    crop_y_end = min(crop_y_start + crop_height, image.shape[0])
    crop_x_end = min(crop_x_start + crop_width, image.shape[1])
    crop_height = crop_y_end - crop_y_start
    crop_width = crop_x_end - crop_x_start
    
    # Crop
    crop = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    
    # Save crop
    crop_filename = os.path.join(OUTPUT_FOLDER, f"screen_{idx}_value_crop.png")
    cv2.imwrite(crop_filename, crop)
    
    # Annotate red box
    cv2.rectangle(annotated, (crop_x_start, crop_y_start), 
                  (crop_x_end, crop_y_end), (0, 0, 255), 2)
    cv2.putText(annotated, f"Value Crop {idx}", (crop_x_start, crop_y_start - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Log crop
    with open(LOG_FILE, "a") as log:
        log.write(f"Value Crop {idx}:\n")
        log.write(f"  Bounding: x={crop_x_start}, y={crop_y_start}, w={crop_width}, h={crop_height}\n")
        log.write(f"  Saved: {crop_filename}\n\n")

# Save final annotated with crops
cv2.imwrite(os.path.join(OUTPUT_FOLDER, "annotated_with_crops.png"), annotated)

# Final log
with open(LOG_FILE, "a") as log:
    log.write(f"Detected {screen_count} screens (including fallbacks if any).\n")
    log.write("Processing complete.\n")

# Console message
print("Processing done. Check the output folder for images and log.")
print(f"Log file: {LOG_FILE}")

# Optional: Add Tesseract integration here if ready
# Install pytesseract and tesseract-ocr separately
# from pytesseract import image_to_string
# Then in crop loop: text = image_to_string(crop, config='--psm 7')  # PSM 7 for single line
# Log the text
