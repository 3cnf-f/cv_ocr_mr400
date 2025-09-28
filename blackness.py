import cv2
import logging
import numpy as np

logging=logging.getLogger(__name__)

def get_blackness_percentage(roi, method='luminance'):
    """
    Calculate how "black" a rectangular region is in a color image.
    
    Args:
        image: OpenCV image (BGR color)
        x, y: Top-left corner coordinates
        w, h: Width and height of the region
        method: 'luminance', 'rgb_sum', 'hsv_value', or 'euclidean'
    
    Returns:
        float: Blackness percentage (0-100, where 100 is pure black)
    """
    # Extract the region of interest
    
    if method == 'luminance':
        # Convert to grayscale using luminance formula (most accurate for human perception)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        blackness = (255 - avg_brightness) / 255 * 100
        
    elif method == 'rgb_sum':
        # Sum of RGB channels (simple but effective)
        rgb_sum = np.mean(np.sum(roi, axis=2))  # Sum B+G+R for each pixel, then average
        max_possible = 255 * 3  # Maximum possible sum (white pixel)
        blackness = (max_possible - rgb_sum) / max_possible * 100
        
    elif method == 'hsv_value':
        # Use HSV Value channel (brightness component)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]  # Value channel
        avg_value = np.mean(v_channel)
        blackness = (255 - avg_value) / 255 * 100
        
    elif method == 'euclidean':
        # Euclidean distance from black (0,0,0)
        distances = np.sqrt(np.sum(roi.astype(np.float32) ** 2, axis=2))
        avg_distance = np.mean(distances)
        max_distance = np.sqrt(255**2 + 255**2 + 255**2)  # Distance to white
        blackness = (max_distance - avg_distance) / max_distance * 100
        
    return blackness

def get_black_pixel_ratio(roi, threshold=30):
    """
    Calculate the ratio of pixels that are considered "black" based on a threshold.
    
    Args:
        threshold: Maximum brightness value to consider a pixel "black" (0-255)
    
    Returns:
        float: Percentage of pixels that are black (0-100)
    """
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Count pixels below threshold
    black_pixels = np.sum(gray <= threshold)
    total_pixels = gray.size
    
    black_ratio = (black_pixels / total_pixels) * 100
    return black_ratio

def is_predominantly_black(roi, blackness_threshold=70):
    """
    Simple function to determine if an area is predominantly black.
    
    Returns:
        bool: True if the area is considered predominantly black
    """
    blackness = get_blackness_percentage(roi, method='luminance')
    return blackness > blackness_threshold

def get_comprehensive_blackness_analysis(roi):
    """
    Get detailed blackness analysis using multiple methods.
    
    Returns:
        dict: Dictionary with various blackness metrics
    """
    
    analysis = {
        'blackness_luminance': get_blackness_percentage(roi, 'luminance'),
        'blackness_rgb_sum': get_blackness_percentage(roi, 'rgb_sum'),
        'blackness_hsv_value': get_blackness_percentage(roi, 'hsv_value'),
        'blackness_euclidean': get_blackness_percentage(roi, 'euclidean'),
        'black_pixel_ratio_strict': get_black_pixel_ratio(roi, threshold=20),
        'black_pixel_ratio_lenient': get_black_pixel_ratio(roi, threshold=50),
        'is_predominantly_black': is_predominantly_black(roi),
        'region_size': w * h,
        'avg_rgb': [np.mean(roi[:,:,i]) for i in range(3)]  # [avg_B, avg_G, avg_R]
    }
    
    return analysis

# For detecting if a screen is "turned off" (black screen)
def is_screen_off(roi, sensitivity='medium'):
    """
    Detect if a monitor/screen appears to be turned off (showing black).
    
    Args:
        sensitivity: 'strict', 'medium', or 'lenient'
    
    Returns:
        bool: True if screen appears to be off
    """
    blackness = get_blackness_percentage(roi, method='luminance')
    black_pixel_ratio = get_black_pixel_ratio(roi, threshold=30)
    
    if sensitivity == 'strict':
        return blackness > 85 and black_pixel_ratio > 80
    elif sensitivity == 'medium':
        return blackness > 75 and black_pixel_ratio > 70
    elif sensitivity == 'lenient':
        return blackness > 65 and black_pixel_ratio > 60
    
    return False

