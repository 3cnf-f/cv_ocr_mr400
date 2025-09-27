import cv2
import numpy as np
import sys

def get_philips_logo_confidence(image_input, debug=False, basename="debug_logo"):
    """
    Analyzes an image to find the "PHILIPS" logo (white text on a dark background)
    and returns a confidence score based on visual heuristics.

    Args:
        image_input: Can be a file path (str) or an OpenCV image object (np.ndarray).
        debug (bool): If True, saves intermediate processing steps as images.
        basename (str): The base name for debug image files.

    Returns:
        float: A confidence score between 0.0 and 1.0.
    """
    
    # --- 1. Input Handling ---
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            print(f"Error: Could not load image from path: {image_input}")
            return 0.0
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        print("Error: Invalid input type. Must be a file path or numpy array.")
        return 0.0

    # --- 2. Isolate Region of Interest (ROI) ---
    # The logo is always in the top part of the bezel.
    img_height, img_width = image.shape[:2]
    # Search the top 20% of the image
    roi_height = int(img_height * 0.20)
    roi = image[0:roi_height, :]

    if debug:
        cv2.imwrite(f'{basename}_01_roi.png', roi)

    # --- 3. Isolate Bright Pixels (the text) ---
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get only very bright pixels. This value may need tuning.
    # A value of 190 is good for high-contrast white text.
    _, thresholded = cv2.threshold(gray_roi, 190, 255, cv2.THRESH_BINARY)
    
    if debug:
        cv2.imwrite(f'{basename}_02_thresholded.png', thresholded)

    # --- 4. Clean up with Morphological Operations ---
    # A closing operation helps to connect broken parts of the letters.
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    if debug:
        cv2.imwrite(f'{basename}_03_morphed.png', morphed)

    # --- 5. Find and Filter "Letter-Like" Contours ---
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    letter_contours = []
    roi_debug_img = roi.copy()
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Filter based on typical properties of the logo's letters
        # - Must be taller than it is wide (aspect ratio > 1)
        # - Height should be within a reasonable range (e.g., 10-50 pixels)
        aspect_ratio = h / float(w) if w > 0 else 0
        if 1.2 < aspect_ratio < 4.0 and 10 < h < 50:
            letter_contours.append(c)
            if debug:
                 cv2.rectangle(roi_debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

    if debug:
        cv2.imwrite(f'{basename}_04_letter_contours.png', roi_debug_img)

    # --- 6. Analyze the Group and Calculate Confidence ---
    # If we didn't find enough potential letters, confidence is zero.
    if len(letter_contours) < 4: # "PHILIPS" has 7 letters, so <4 is a fail.
        return 0.0
        
    # Get the bounding box of the entire group of letters
    all_points = np.concatenate(letter_contours)
    group_x, group_y, group_w, group_h = cv2.boundingRect(all_points)

    # SCORE COMPONENT 1: Number of letters found
    # Score is proportional to the number of letters, maxing out at 7.
    num_letters_score = min(1.0, len(letter_contours) / 7.0)

    # SCORE COMPONENT 2: Aspect ratio of the whole word
    # The word "PHILIPS" is very wide. Its w/h ratio should be high.
    group_aspect_ratio = group_w / float(group_h)
    aspect_score = 0.0
    if 4.0 < group_aspect_ratio < 8.0:
        aspect_score = 1.0 # Perfect match
    elif 3.0 < group_aspect_ratio < 9.0:
        aspect_score = 0.5 # Plausible match
        
    # SCORE COMPONENT 3: Horizontal alignment of letters
    # The y-coordinates of the letters should be very similar.
    y_coords = [cv2.boundingRect(c)[1] for c in letter_contours]
    y_std_dev = np.std(y_coords)
    # Normalize by letter height. A low std dev is good.
    alignment_score = max(0.0, 1.0 - (y_std_dev / group_h))

    # FINAL WEIGHTED SCORE
    # Give the most weight to finding the right number of letters.
    final_confidence = (
        (num_letters_score * 0.5) +
        (aspect_score * 0.25) +
        (alignment_score * 0.25)
    )

    if debug:
        print(f"\n--- LOGO DETECTION DEBUG ---")
        print(f"Found {len(letter_contours)} letter-like contours.")
        print(f"Group BBox: w={group_w}, h={group_h}, aspect={group_aspect_ratio:.2f}")
        print(f"Y-Alignment Std Dev: {y_std_dev:.2f}")
        print(f"Scores -> NumLetters: {num_letters_score:.2f}, Aspect: {aspect_score:.2f}, Alignment: {alignment_score:.2f}")
        print(f"Final Confidence: {final_confidence:.2f}")
        
    return final_confidence


# --- Example Usage ---
if __name__ == "__main__":
    # Check if a file path is provided
    if len(sys.argv) < 2:
        print("Usage: python3 logo_detector.py <path_to_image.png>")
        sys.exit(1)
        
    image_path = sys.argv[1]

    # Run the function with debugging enabled
    confidence = get_philips_logo_confidence(image_path, debug=True, basename=image_path.replace('.png', ''))
    
    print(f"\n✅ The confidence of finding the 'PHILIPS' logo in '{image_path}' is: {confidence:.2%}")
