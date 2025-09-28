import cv2
import numpy as np

# Load the image (replace 'your_image_path.jpg' with the actual path to the uploaded image)
# This reads the image file into a numpy array for processing.
image_path = 'output.png'
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Image not found or unable to load at path: {image_path}")

# Print basic image info for debugging
print(f"Image loaded successfully. Shape: {image.shape} (height, width, channels)")

# Convert to grayscale
# Grayscale simplifies edge detection by reducing to one channel.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Converted to grayscale.")

# Apply Gaussian blur to reduce noise
# Blur helps smooth out minor variations that could create false edges.
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
print("Applied Gaussian blur.")

# Use Canny edge detection
# Lowered thresholds for more edges: low=30 detects weaker edges, high=100 caps stronger ones.
edges = cv2.Canny(blurred, 30, 100)
print("Applied Canny edge detection with adjusted thresholds.")

# Optionally save edges for visual inspection
cv2.imwrite('edges.jpg', edges)
print("Saved edge-detected image as 'edges.jpg' for debugging.")

# Find contours
# Contours are outlines of shapes detected from edges. Using RETR_EXTERNAL for outer contours.
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} contours in total.")

# Filter contours to find potential screen rectangles
screens = []
candidate_contours = []  # To store all quad candidates before final filtering
for i, contour in enumerate(contours):
    # Approximate the contour to a polygon
    # Increased epsilon slightly for better handling of slight distortions/tilts.
    epsilon = 0.03 * cv2.arcLength(contour, True)  # Adjusted from 0.02 for more flexibility
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    
    # Calculate area for filtering small noise contours
    area = cv2.contourArea(contour)
    print(f"Contour {i}: Approx sides = {len(approx)}, Area = {area}")
    
    # Check if it's roughly a quadrilateral (4 sides) and has sufficient area
    # Relaxed to len(approx) == 4 (strict quad) but lowered area for smaller/distant screens.
    if len(approx) == 4 and area > 2000:  # Lowered area threshold for more leniency
        # Get bounding rectangle (axis-aligned for simplicity)
        x, y, w, h = cv2.boundingRect(approx)
        
        # Calculate aspect ratio (width/height)
        aspect_ratio = w / float(h)
        print(f"Contour {i}: Bounding rect (x,y,w,h) = ({x},{y},{w},{h}), Aspect ratio = {aspect_ratio:.2f}")
        
        # Store as candidate (will draw in red)
        candidate_contours.append((x, y, w, h))
        
        # Filter by aspect ratio (screens are roughly landscape rectangular)
        # Broadened range to handle variations in tilt/perspective.
        if 1.0 < aspect_ratio < 3.0:  # Even broader for tilted screens
            screens.append((x, y, w, h))
            print(f"Contour {i}: Accepted as screen.")
        else:
            print(f"Contour {i}: Rejected due to aspect ratio.")
    else:
        print(f"Contour {i}: Rejected (not quad or too small).")

# Sort screens by x-coordinate to order them left to right
screens.sort(key=lambda rect: rect[0])
print(f"Total screens detected: {len(screens)}")
print(f"Total quad candidates (pre-filter): {len(candidate_contours)}")

# Draw bounding boxes on the image for visualization
# Use a copy to avoid modifying the original image.
output = image.copy()

# Draw red rectangles for all quad candidates
for (x, y, w, h) in candidate_contours:
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for candidates

# Draw green rectangles for accepted screens (overriding red if accepted)
for i, (x, y, w, h) in enumerate(screens):
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for accepted
    cv2.putText(output, f"Screen {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the result in a window
# This opens a window showing the image with drawings.
cv2.imshow('Detected Screens', output)
cv2.waitKey(0)  # Wait for any key press to close
cv2.destroyAllWindows()
print("Displayed output window.")

# Optionally save the output
cv2.imwrite('detected_screens.jpg', output)
print("Saved annotated image as 'detected_screens.jpg'.")

# Print the bounding boxes
print("Detected screen locations (x, y, width, height):")
for rect in screens:
    print(rect)

# If no screens detected, suggest next steps
if len(screens) == 0:
    print("No screens detected. Check 'edges.jpg' for edge quality. Consider adjusting Canny thresholds or epsilon if contours are incomplete.")
