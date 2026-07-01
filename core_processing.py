import cv2
import numpy as np

def enhance_structural_features(image_matrix, clip_limit=3.0, tile_size=8, sharpen_weight=0.4):
    """
    Employs high-speed spatial filters (CLAHE and Laplacian) with adjustable parameters
    to reveal hidden textures and boundaries dynamically[cite: 46, 76].
    """
    # Normalize input matrix data to a standard 8-bit array
    normalized_img = cv2.normalize(image_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 1. Apply Contrast Limited Adaptive Histogram Equalization with dynamic sliders [cite: 46, 76]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    equalized_img = clahe.apply(normalized_img)
    
    # 2. Extract edge patterns via Laplacian calculations with adjustable intensity weighting [cite: 46, 76]
    laplacian_edges = cv2.Laplacian(equalized_img, cv2.CV_64F)
    sharpened_matrix = np.clip(equalized_img - sharpen_weight * laplacian_edges, 0, 255).astype(np.uint8)
    
    return sharpened_matrix

def apply_thermal_color_mapping(sharpened_matrix, palette_selection="JET"):
    """
    Utilizes non-destructive look-up tables to map specific thermal distributions[cite: 47].
    """
    if palette_selection == "JET":
        colorized_bgr = cv2.applyColorMap(sharpened_matrix, cv2.COLORMAP_JET)
    elif palette_selection == "VIRIDIS":
        colorized_bgr = cv2.applyColorMap(sharpened_matrix, cv2.COLORMAP_VIRIDIS)
    else:
        colorized_bgr = cv2.applyColorMap(sharpened_matrix, cv2.COLORMAP_HOT)
        
    return cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)