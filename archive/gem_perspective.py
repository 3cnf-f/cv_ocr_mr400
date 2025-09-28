import cv2
import numpy as np
import logging

# Set up logging to a new file to avoid confusion
logging.basicConfig(filename='gem_log_revised.log', level=logging.INFO, 
                    filemode='w', # 'w' overwrites the log file each time
                    format='%(asctime)s - %(message)s')

def find_screens_revised(image_path):
    """
    Finds and outlines screens in an image using bounding box properties,
    which is more robust to rounded corners.

    Args:
        image_path (str): The path to the input image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image from {image_path}")
        print(f"Error: Could not load image from {image_path}")
        return

    img_height, img_width, _ = image.shape
    logging.info(f"Image loaded: {image_path}, Dimensions: {img_width}x{img_height}")

    # --- Preprocessing ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('debug_1_grayscale.png', gray)
    logging.info("Saved grayscale image to debug_1_grayscale.png")

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imwrite('debug_2_blurred.png', blurred)
    logging.info("Saved blurred image to debug_2_blurred.png")

    edges = cv2.Canny(blurred, 30, 100) # Adjusted thresholds for potentially cleaner edges
    cv2.imwrite('debug_3_edges.png', edges)
    logging.info("Saved edge-detected image to debug_3_edges.png")
    
    # --- Contour Detection ---
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.info(f"Found {len(contours)} initial contours.")

    output_image = image.copy()
    screen_contours = []

    # --- Filtering Logic ---
    for i, c in enumerate(contours):
        # Get the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # Basic filter: ignore very small contours to reduce noise/log spam
        if w < 270 or h < 150:
            continue

        # 1. Check size constraints
        width_check = (0.15 * img_width) < w < (0.40 * img_width)
        height_check = (0.15 * img_height) < h < (0.43 * img_height)

        # 2. Check aspect ratio
        # Avoid division by zero if height is 0
        aspect_ratio = w / float(h) if h > 0 else 0
        aspect_ratio_check = 0.8 < aspect_ratio < 2.5
        
        # 3. Check solidity (how "full" the shape is)
        contour_area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / float(hull_area) if hull_area > 0 else 0
        # Screens should be very solid shapes. We look for a high solidity value.
        solidity_check = solidity > 0.60

        log_msg = (f"Contour #{i}: x={x}, y={y}, w={w}, h={h}, "
                   f"width_check={width_check}, height_check={height_check}, "
                   f"aspect_ratio={aspect_ratio:.2f} (check={aspect_ratio_check}) "
                   )

        # A contour is a screen if it passes all checks
        if width_check and height_check and aspect_ratio_check :
            screen_contours.append(c)
            logging.info(f"{log_msg} -> ACCEPTED")
            # Draw the bounding box on the output image
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            logging.warning(f"{log_msg} -> DISCARDED")


    logging.info(f"Found {len(screen_contours)} potential screens after filtering.")

    # Save the final image
    cv2.imwrite('output_with_outlines_revised.png', output_image)
    logging.info("Saved final image with outlines to output_with_outlines_revised.png")
    print("Processing complete. Check 'output_with_outlines_revised.png' and 'gem_log_revised.log'.")

# --- Main execution ---
if __name__ == "__main__":
    # Ensure you have the image saved as 'output.png' in the same directory
    input_image_file = 'output.png' 
    find_screens_revised(input_image_file)
