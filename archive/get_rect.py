import cv2
import numpy as np

# Hardcoded values as per user (removed argv)
block_size = 61  # Odd number for adaptive threshold
constant = 8     # Constant for adaptive threshold

# Load the image
image_path = 'output.png'
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Image not found or unable to load at path: {image_path}")

print(f"Image loaded successfully. Shape: {image.shape} (height, width, channels)")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Converted to grayscale.")

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
print("Applied Gaussian blur.")

# Use adaptive threshold with hardcoded values
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant)

# Optional: dilate to connect gaps from glare (kernel size 5x5, once)
kernel = np.ones((5,5), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=1)
# Find contours on thresh
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours left-to-right
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# After sorting contours...

# Hardcode epsilon_mult for now (0.02 balances detail vs simplification; test 0.03 if too many points)
epsilon_mult = 0.02

# List for detected screens
detected_screens = []

# Make copy for final drawing
img_detected = image.copy()

# Loop and filter with relaxed thresholds and near-miss debug
for i, cnt in enumerate(contours):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_mult * peri, closed=True)
    
    x, y, w, h = cv2.boundingRect(approx)
    area = cv2.contourArea(cnt)
    aspect = w / h if h != 0 else 0
    if area > 50000:  # Still log big ones for debug
        crop_gray = gray[y:y+h, x:x+w]
        mean_gray = cv2.mean(crop_gray)[0]
        print(f"Near-miss #{i+1}: points={len(approx)}, area={area:.0f}, aspect={aspect:.2f}, mean_gray={mean_gray:.1f}, pos=({x},{y}) size={w}x{h}")
    
    # Relaxed filter: more points ok for perspective, wider aspect, higher mean for lit screens, bigger min area
    if 3 <= len(approx) <= 12 and area > 100000 and 1.0 < aspect < 3.0 and mean_gray < 100:
        crop_color = image[y:y+h, x:x+w]
        hsv = cv2.cvtColor(crop_color, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))
        yellow_count = cv2.countNonZero(yellow_mask)
        if yellow_count > 50 or mean_gray < 80:  # Fallback to dark if no yellow (e.g., middle screen)
            detected_screens.append((x, y, w, h))
            cv2.rectangle(img_detected, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"Detected screen #{len(detected_screens)}: points={len(approx)}, area={area:.0f}, aspect={aspect:.2f}, mean_gray={mean_gray:.1f}, yellow_pixels={yellow_count}")

# Limit to top 3
detected_screens = detected_screens[:3]

# Save final
cv2.imwrite('detected_screens.png', img_detected)
print(f"Saved 'detected_screens.png' with {len(detected_screens)} screens.")

# Crop and save each
for num, (x, y, w, h) in enumerate(detected_screens, start=1):
    screen_crop = image[y:y+h, x:x+w]
    cv2.imwrite(f'screen_{num}.png', screen_crop)
    print(f"Saved screen_{num}.png")
