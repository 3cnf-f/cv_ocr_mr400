import cv2
import sys
import numpy as np

argv = sys.argv
# python3 get_rect.py input_file block(odd number) constant
# This reads the image file into a numpy array for processing.
image_path = argv[1]
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


## use adaptive threshhold
# After Gaussian blur on gray...
# default ...BINARY_INV, block 31, constant 2)
# after experimenting with different parameters (grok guessed values using knowledge from prior attempts) we have set 
# block to 61 and constant to 8
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61 ,8 )  # Block size 31 (odd), constant 2; tweak if noisy

# Optional: dilate to connect gaps from glare (kernel size 5x5, once)
kernel = np.ones((5,5), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=1)

# Now find contours on thresh (not edges)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Then sort and loop/filter as before...
# In the filter: up the min area to ~10000 if small junk slips in; lower mean threshold to <80 for grey screens

# Use Canny edge detection
# Lowered thresholds for more edges: low=30 detects weaker edges, high=100 caps stronger ones.

contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
img_copy = image.copy()

# List to hold our detected screen bounding boxes (x, y, width, height)
detected_screens = []

# Loop through each sorted contour and filter for screen candidates
for i, cnt in enumerate(contours):
    # Approximate the contour to a simpler polygon (helps check if it's roughly rectangular)
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), closed=True)
    
    # Get the bounding rectangle around the approximated contour
    x, y, w, h = cv2.boundingRect(approx)
    
    # Filter 1: Must be roughly rectangular (4 sides) and big enough (adjust min_area based on your image; ~5000-10000 for screens)
    # Filter 2: Aspect ratio like a screen (wider than tall, say 1-3 ratio)
    if len(approx) == 4 and w * h > 10000 and 1 < w / h < 3:  # Tweak these if screens are smaller/bigger or taller
        # Crop the grayscale version of this region to check darkness
        crop_gray = gray[y:y+h, x:x+w]
        
        # Filter 3: Mean intensity should be low (dark screen; <80 for grey/black, tweak if your screens are lighter)
        if cv2.mean(crop_gray)[0] < 80:
            # Optional Filter 4: Check for yellow text in the color crop (confirms it's a temp-displaying screen)
            crop_color = image[y:y+h, x:x+w]  # Use original color image
            hsv = cv2.cvtColor(crop_color, cv2.COLOR_BGR2HSV)
            # Yellow range in HSV (hue 20-40, high sat/val; adjust if your yellow is greenish/reddish)
            yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))
            if cv2.countNonZero(yellow_mask) > 50:  # At least some yellow pixels (tweak if no text on middle screen)
                # It's a match! Add to list and draw green rectangle on copy for debug
                detected_screens.append((x, y, w, h))
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print(f"Detected screen #{i+1} at ({x}, {y}) with size {w}x{h}")

# Now save the debug image with only filtered screens outlined
#cv2.imwrite('detected_screens_B' + argv[1] + '_C' + argv[2] + '.png', img_copy)
cv2.imwrite('detected_screens' + '.png', img_copy)
print(f"Saved detected screens image as 'detected_screens.png'.")

# Bonus: Crop and save each detected screen as separate images (for later OCR)
for num, (x, y, w, h) in enumerate(detected_screens, start=1):
    screen_crop = image[y:y+h, x:x+w]
    cv2.imwrite(f'screen_{num}.png', screen_crop)
    print(f"Saved cropped screen {num} as 'screen_{num}.png'.")
