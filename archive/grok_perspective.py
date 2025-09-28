import cv2
import numpy as np
import logging

# Setup logging
logging.basicConfig(filename='grok_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the image
image_path = 'output.png'  # Replace with your image file path
img = cv2.imread(image_path)
if img is None:
    logging.error("Image not found. Check the file path.")
    raise ValueError("Image not found. Check the file path.")

# Get image dimensions
img_height, img_width = img.shape[:2]
logging.info(f"Loaded image: {image_path}, dimensions: {img_width}x{img_height}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('debug_gray.jpg', gray)
logging.info("Saved grayscale image: debug_gray.jpg")

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite('debug_blurred.jpg', blurred)
logging.info("Saved blurred image: debug_blurred.jpg")

# Edge detection using Canny (adjust thresholds if needed for your images)
edges = cv2.Canny(blurred, 50, 150)
cv2.imwrite('debug_edges.jpg', edges)
logging.info("Saved edges image: debug_edges.jpg")

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
logging.info(f"Found {len(contours)} contours")

# Create debug image for all contours
img_all_contours = img.copy()
cv2.drawContours(img_all_contours, contours, -1, (0, 0, 255), 2)  # Red for all contours
cv2.imwrite('debug_all_contours.jpg', img_all_contours)
logging.info("Saved all contours image: debug_all_contours.jpg")

# Create debug image for approximated polygons
img_approx_polys = img.copy()

# Filter contours to find screen candidates
detected_screens = []
quad_candidates = 0
for i, contour in enumerate(contours):
    # Approximate the contour to a polygon
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), closed=True)
    
    # Draw all approximated polygons (blue)
    cv2.drawContours(img_approx_polys, [approx], -1, (255, 0, 0), 2)
    
    # We want quadrilaterals (4 sides)
    if len(approx) == 4:
        quad_candidates += 1
        # Get the bounding rectangle (axis-aligned for horizontal/vertical measurement)
        x, y, w, h = cv2.boundingRect(approx)
        
        # Calculate percentages
        width_percent = w / img_width
        height_percent = h / img_height
        aspect_ratio = w / h if h != 0 else 0
        
        logging.info(f"Quad {i}: bounding box {w}x{h} ({width_percent:.2%} width, {height_percent:.2%} height), aspect {aspect_ratio:.2f}")
        
        # Filter by size: 15-40% of image width/height
        if 0.15 <= width_percent <= 0.40 and 0.15 <= height_percent <= 0.40:
            # Filter by aspect ratio: 1.5-1.7
            if 1.5 <= aspect_ratio <= 1.7:
                # Optional: Filter by area to avoid tiny noise
                area = cv2.contourArea(contour)
                logging.info(f"Quad {i}: area {area}")
                if area > 1000:
                    detected_screens.append(approx)
                    # Draw the bounding box (green) on the final image
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    logging.info(f"Quad {i}: Detected as screen")

# Save approx polys debug image
cv2.imwrite('debug_approx_polys.jpg', img_approx_polys)
logging.info(f"Saved approximated polygons image: debug_approx_polys.jpg (found {quad_candidates} quads)")

# Output results
print(f"Detected {len(detected_screens)} screens matching criteria.")
logging.info(f"Detected {len(detected_screens)} screens matching criteria.")

# Save and display the final result
cv2.imwrite('output_grok.jpg', img)
logging.info("Saved final output: output.jpg")
cv2.imshow('Detected Screens', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
