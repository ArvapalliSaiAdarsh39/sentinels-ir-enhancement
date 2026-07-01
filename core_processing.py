import cv2
import numpy as np

def enhance_structural_features(image_matrix):
    """
    Employs high-speed spatial filtering (CLAHE and Laplacian) to reveal hidden 
    textures and faint boundaries in raw IR telemetry without data distortion[cite: 46].
    """
    # Step 1: Normalize input matrix data to a standard 8-bit array
    normalized_img = cv2.normalize(image_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Step 2: Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) [cite: 46]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(normalized_img)
    
    # Step 3: Extract edge patterns via Laplacian calculations to sharpen the image [cite: 46]
    laplacian_edges = cv2.Laplacian(equalized_img, cv2.CV_64F)
    sharpened_matrix = np.clip(equalized_img - 0.4 * laplacian_edges, 0, 255).astype(np.uint8)
    
    return sharpened_matrix

def apply_thermal_color_mapping(sharpened_matrix, palette_selection="JET"):
    """
    Utilizes calibrated, non-destructive look-up tables (LUTs) to translate 
    thermal distributions into accurate, standardized color channels[cite: 47].
    """
    # Map a standardized deterministic color layout to the structural matrix [cite: 32, 47]
    if palette_selection == "JET":
        colorized_bgr = cv2.applyColorMap(sharpened_matrix, cv2.COLORMAP_JET)
    elif palette_selection == "VIRIDIS":
        colorized_bgr = cv2.applyColorMap(sharpened_matrix, cv2.COLORMAP_VIRIDIS)
    else:
        colorized_bgr = cv2.applyColorMap(sharpened_matrix, cv2.COLORMAP_HOT)
        
    # Convert native OpenCV BGR format to RGB for accurate web browser display
    return cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)