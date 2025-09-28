# claude_perspective_optimized.py
import blackness
import sys
import numpy as np
import cv2
import os
import logging
from datetime import datetime
import uuid
from pathlib import Path
# --- Script Setup ---
if len(sys.argv) < 3:
    print("❌ Error: Please provide an image file path as an argument and a outputfolder path.")
    print("Usage: python3 claude_perspective_optimized.py <path_to_image.png> folder")
    sys.exit(1)

input_image = sys.argv[1]
basename = Path(input_image).stem #filename without extension
basename = basename+"_out"
foldername = sys.argv[2]
foldername = foldername + "/"
# Create the output folder
os.makedirs(foldername, exist_ok=True)
debuglevel = 0

# Setup logging
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for cleaner production output
    format='%(asctime)s-%(levelname)s-%(module)s-%(funcName)s- %(lineno)s- %(message)s',
    handlers=[
        logging.FileHandler(foldername+'claude_log_optimized.log', mode='a'),
        logging.StreamHandler()
    ]
)
logging=logging.getLogger(__name__)

def find_screens(image_path):
    """
    Finds the 3 main screens in an image using a single, optimized Canny edge detection setting.
    """
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image from {image_path}")
        return []

    img_height, img_width = image.shape[:2]
    logging.info(f"Image loaded: {image_path}, Dimensions: {img_width}x{img_height}")
    
    # --- Preprocessing ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # --- Optimized Edge Detection ---
    # Using the "higher" setting (40, 120) which was found to be most effective.
    low_thresh, high_thresh = 40, 120
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    if debuglevel>2: cv2.imwrite(foldername+f'{basename}_05_edges.png', edges)

    logging.info(f"Using optimized edge detection: Canny({low_thresh}, {high_thresh})")
    
    # --- Contour Detection and Filtering ---
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.info(f"Found {len(contours)} initial contours.")
    
    all_candidates = []
    candidate_id = 0
    debug_img = image.copy()
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip very small contours to reduce noise
        if w < 300 or h < 200:
            continue
        
        # Calculate metrics
        contour_area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / float(hull_area) if hull_area > 0 else 0
        width_ratio = w / float(img_width)
        height_ratio = h / float(img_height)
        aspect_ratio = w / float(h) if h > 0 else 0
        blackness_percentage = blackness.get_blackness_percentage(image[y:y+h,x:x+w])
        avg_y=(y+y+h)/2
        
        
        # Store candidate info
        candidate = {
            'id': candidate_id,
            'x': x, 'y': y, 'w': w, 'h': h,
            'aspect_ratio': aspect_ratio,
            'contour': contour,
            'blackness_percentage': blackness_percentage,
            'avg_y': avg_y
        }
        
        # Define filtering criteria (you can adjust these if needed)
        width_check = (0.15 * img_width) < w < (0.40 * img_width)
        height_check = (0.15 * img_height) < h < (0.6 * img_height)
        aspect_check = 0.8 < aspect_ratio < 2.5
        blackness_check =  blackness_percentage > 50
        
        # Categorize candidate
        if width_check and height_check and aspect_check and blackness_check:
            candidate['category'] = 'EX'
            color = (0, 255, 0)  # Green
            thickness = 3
        else:
            candidate['category'] = 'PO' # Simplified category
            color = (128, 128, 128) # Gray
            thickness = 1
        

        logging.info(f"Candidate #{candidate_id}: Pos({x},{y}) Size({w}x{h}) Aspect({aspect_ratio:.2f}) Solidity({solidity:.2f}) Blackness({blackness_percentage:.2f}) y_avg({avg_y:.2f}) -> Category: {candidate['category']}")
        
        all_candidates.append(candidate)
        
        # Draw on debug image
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
        cv2.putText(debug_img, f"#{candidate_id} ({candidate['category']}) ({candidate['blackness_percentage']:.2f})", (x+5, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        candidate_id += 1
    
    cv2.imwrite(foldername+f'{basename}_06_candidates.png', debug_img)
    logging.info(f"Saved candidates debug image.")

    # --- Final Selection Logic (with overlap removal) ---
    final_screens = []
    y_ex_center_sum = 0
    ex_num = 0

    for candidate in all_candidates:
        if candidate['category'] == 'EX':
            y_ex_center_sum = y_ex_center_sum + (candidate['avg_y'])
            ex_num = ex_num + 1
    y_ex_center_avg = y_ex_center_sum / ex_num
    logging.info(f"y_ex_center_avg: {y_ex_center_avg}")

    for candidate in all_candidates:
        if candidate['category'] == 'EX':

            logging.info(f"EX_Candidate #{candidate_id}:  avg_y #({candidate['avg_y']:.2f})")

                


    # Sort candidates to prioritize the best ones
    # sorted_candidates = sorted([c for c in all_candidates if c['category'] == 'EX'],
    #                           key=lambda c: (c['blackness_percentage']-49)/50 *c['w']*c['h'], reverse=True) # Prioritize larger screens
    sorted_candidates = sorted([c for c in all_candidates if c['category'] == 'EX'],
                              key=lambda c: (c['avg_y'], reversed==True)) # Prioritize closer to y avg 

    for candidate in sorted_candidates:
        if len(final_screens) >= 3:
            break # Stop once we have 3 screens

        # Check for significant overlap with already selected screens
        is_overlapping = False
        for screen in final_screens:
            # Calculate Intersection over Union (IoU) or a simpler overlap metric
            x1 = max(candidate['x'], screen['x'])
            y1 = max(candidate['y'], screen['y'])
            x2 = min(candidate['x'] + candidate['w'], screen['x'] + screen['w'])
            y2 = min(candidate['y'] + candidate['h'], screen['y'] + screen['h'])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            candidate_area = candidate['w'] * candidate['h']
            
            # If the intersection is more than 30% of the candidate's area, it's an overlap
            if intersection / candidate_area > 0.3:
                is_overlapping = True
                logging.warning(f"Candidate #{candidate['id']} overlaps with a selected screen. Discarding.")
                break
        
        if not is_overlapping:
            final_screens.append(candidate)
    
    # --- Generate Final Output ---
    final_img = image.copy()
    if not final_screens:
        logging.error("No final screens selected after filtering.")
    else:
        logging.info(f"Selected {len(final_screens)} final screens.")
        final_screens = sorted(final_screens, key=lambda s: s['x'])

    for i, screen in enumerate(final_screens):
        x, y, w, h = screen['x'], screen['y'], screen['w'], screen['h']
        cv2.rectangle(final_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(final_img, f"Screen {i+1}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save individual screen ROI (Region of Interest)
        screen_roi = image[y:y+h, x:x+w]
        cv2.imwrite(foldername+f'{basename}_08_final_screen_{i+1}.png', screen_roi)
        
        logging.info(f"=== FINAL SCREEN {i+1} (from Cand. #{screen['id']}) ===")
        logging.info(f"  Position: ({x}, {y}), Size: {w}x{h}")

    cv2.imwrite(foldername+f'{basename}_08_final.png', final_img)
    
    logging.info(f"✅ Final processing complete. Found {len(final_screens)} screens.")
    return [(s['x'], s['y'], s['w'], s['h']) for s in final_screens]

# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"=== Monitor Detection Log - {datetime.now()} ===\n")
    
    screens = find_screens(input_image)
    
    if screens:
        print(f"\n✅ SUCCESS: Detected {len(screens)} screens.")
        for i, (x, y, w, h) in enumerate(screens):
            print(f"  Screen {i+1}: Position ({x}, {y}), Size {w}x{h}")
    else:
        print("\n❌ FAILED: No screens detected that meet the criteria.")
        print("Check claude_log_optimized.log and the debug images (*_05_edges.png, *_06_candidates.png).")
