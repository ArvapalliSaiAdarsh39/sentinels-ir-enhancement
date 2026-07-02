"""
core_processing.py
=====================================================================
Infrared Image Colourization & Enhancement Workstation
Deterministic OpenCV / NumPy Matrix Processing Engine
=====================================================================

Every transformation in this module is a closed-form, explicitly
parameterized mathematical operation. There are no learned weights, no
black-box models, and no generative components of any kind -- feed the
same matrix and the same parameters into this module twice, and you
will get bit-identical output twice. That determinism is the entire
point of Architectural Philosophy #1.

Processing chain implemented here:

    raw band
      -> percentile contrast stretch      (normalize_to_uint8)
      -> Stage 1: bilateral noise filter  (apply_bilateral_denoise)
      -> Stage 2: CLAHE contrast boost    (apply_clahe_enhancement)
      -> edge accentuation                (apply_laplacian_sharpen)
      -> LUT colourization (x3 palettes)  (generate_multi_palette_products)
      -> optional thermal isolation mask  (generate_thermal_signature_mask)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Palette registry -- the three scientific LUTs the export pipeline bundles
# ---------------------------------------------------------------------------

PALETTE_REGISTRY: Dict[str, int] = {
    "JET": cv2.COLORMAP_JET,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "HOT": cv2.COLORMAP_HOT,
}


@dataclass
class EnhancementParameters:
    """Every tunable knob of the dual-stage enhancement chain, bundled into
    one immutable, explicit configuration object. Passing this around
    (rather than five loose function arguments) keeps the pipeline call
    sites readable and makes every run fully reproducible from a single
    logged object."""

    bilateral_diameter: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    sharpen_strength: float = 0.35
    lower_percentile: float = 2.0
    upper_percentile: float = 98.0


# ---------------------------------------------------------------------------
# Band selection & statistics
# ---------------------------------------------------------------------------


def select_band(raw_array: np.ndarray, band_index: int = 0) -> np.ndarray:
    """Extracts a single 2-D working band from a (bands, height, width)
    matrix, safely clamping the index for malformed/edge-case inputs."""
    if raw_array.ndim == 2:
        return raw_array
    band_index = int(np.clip(band_index, 0, raw_array.shape[0] - 1))
    return raw_array[band_index, :, :]


def compute_band_statistics(band: np.ndarray) -> Dict[str, float]:
    """Core descriptive statistics for a raw (pre-normalization) band,
    ignoring any non-finite values (NaN / inf) that raw sensor products
    sometimes carry as no-data sentinels."""
    working = band.astype(np.float64)
    finite = working[np.isfinite(working)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_to_uint8(
    band: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """
    Performs a percentile-based linear contrast stretch, mapping the raw
    sensor dynamic range (which may be uint16, int16, or float32) into a
    clean 8-bit [0, 255] channel, while suppressing the outlier hot/cold
    pixels that would otherwise wash out a naive min/max stretch.

        I_norm = 255 * clip((I - P_low) / (P_high - P_low), 0, 1)
    """
    working = band.astype(np.float64)
    finite_mask = np.isfinite(working)
    if not np.any(finite_mask):
        return np.zeros(band.shape, dtype=np.uint8)

    lo = np.percentile(working[finite_mask], lower_percentile)
    hi = np.percentile(working[finite_mask], upper_percentile)
    if hi <= lo:
        hi = lo + 1.0

    stretched = (working - lo) / (hi - lo)
    stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)
    stretched = np.clip(stretched, 0.0, 1.0)
    return (stretched * 255.0).astype(np.uint8)


def downsample_for_preview(image: np.ndarray, max_dimension: int = 1400) -> np.ndarray:
    """
    Produces a browser-friendly preview copy of a (possibly very large)
    raster by area-averaging downsampling whenever the longest edge
    exceeds `max_dimension`. This never touches the full-resolution
    arrays used for GeoTIFF export -- it is purely a rendering-
    performance convenience for the on-screen canvas.
    """
    height, width = image.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= max_dimension:
        return image

    scale = max_dimension / float(longest_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Dual-stage enhancement chain
# ---------------------------------------------------------------------------


def apply_bilateral_denoise(
    image_uint8: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Stage 1 -- Edge-Preserving Bilateral Sensor Noise Filter.

    Suppresses stochastic sensor grain while explicitly preserving high-
    gradient structural edges (coastlines, thermal boundaries, urban
    structures) by weighting the smoothing kernel with both a spatial-
    proximity term and a pixel-intensity-similarity term, so a pixel is
    only blended with neighbours that are *both* nearby and photometrically
    similar to it.
    """
    return cv2.bilateralFilter(
        image_uint8,
        d=int(diameter),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )


def apply_clahe_enhancement(
    image_uint8: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Stage 2 -- Contrast-Limited Adaptive Histogram Equalization.

    Redistributes local pixel-intensity histograms within tiled
    neighbourhoods to recover latent structural contrast that a single
    global stretch would miss, clipping each tile's histogram at
    `clip_limit x mean_bin_height` before equalizing so that flat, low-
    variance regions (open water, cloud tops) don't get their sensor
    noise amplified into false structure.
    """
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(int(v) for v in tile_grid_size))
    return clahe.apply(image_uint8)


def apply_laplacian_sharpen(image_uint8: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """
    Optional edge-accentuation pass, applied last in the enhancement chain.

    Convolves the enhanced matrix with a discrete Laplacian kernel to
    isolate second-derivative edge energy, then adds a `strength`-weighted
    fraction of that edge map back onto the source image (an unsharp-mask
    construction) to visually crisp thermal boundaries without amplifying
    broadband sensor noise the way a global sharpen would.

        I_sharp = I + lambda * Laplacian(I),   Laplacian kernel =
            [ 0  -1   0 ]
            [-1   4  -1 ]
            [ 0  -1   0 ]
    """
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    edge_energy = cv2.filter2D(image_uint8.astype(np.float32), -1, laplacian_kernel)
    sharpened = image_uint8.astype(np.float32) + (float(strength) * edge_energy)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def run_dual_stage_pipeline(
    raw_band: np.ndarray, params: EnhancementParameters
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes the full deterministic enhancement chain and returns both the
    pre-enhancement (post-normalization only) monochrome matrix and the
    fully enhanced monochrome matrix, so the caller can render a true
    side-by-side, synchronized before/after comparison from a single pass.
    """
    normalized = normalize_to_uint8(raw_band, params.lower_percentile, params.upper_percentile)
    denoised = apply_bilateral_denoise(
        normalized,
        params.bilateral_diameter,
        params.bilateral_sigma_color,
        params.bilateral_sigma_space,
    )
    contrast_enhanced = apply_clahe_enhancement(denoised, params.clahe_clip_limit, params.clahe_tile_grid)
    sharpened = apply_laplacian_sharpen(contrast_enhanced, params.sharpen_strength)
    return normalized, sharpened


# ---------------------------------------------------------------------------
# Colourization
# ---------------------------------------------------------------------------


def apply_colormap(image_uint8: np.ndarray, palette_name: str) -> np.ndarray:
    """
    Maps a monochrome 8-bit matrix through a fixed, precomputed 256-entry
    scientific Look-Up Table, returning an (H, W, 3) array in RGB channel
    order. OpenCV's `applyColorMap` natively emits BGR, so the channel
    order is explicitly reversed here to keep every downstream consumer
    (Streamlit, GeoTIFF export) in unambiguous RGB.
    """
    palette_key = palette_name.upper()
    if palette_key not in PALETTE_REGISTRY:
        raise ValueError(f"Unknown palette '{palette_name}'. Valid options: {list(PALETTE_REGISTRY)}")

    bgr = cv2.applyColorMap(image_uint8, PALETTE_REGISTRY[palette_key])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def generate_multi_palette_products(image_uint8: np.ndarray) -> Dict[str, np.ndarray]:
    """Produces the full JET / VIRIDIS / HOT bundle from a single enhanced
    matrix in one call -- the backbone of the multi-palette export pipeline."""
    return {name: apply_colormap(image_uint8, name) for name in PALETTE_REGISTRY}


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------


def compute_histogram(image_uint8: np.ndarray, bins: int = 256) -> np.ndarray:
    """Computes the exact 256-bin pixel-intensity distribution of a
    monochrome matrix as a flat float32 array of length `bins`."""
    hist = cv2.calcHist([image_uint8], [0], None, [bins], [0, 256])
    return hist.flatten()


# ---------------------------------------------------------------------------
# Thermal isolation masking
# ---------------------------------------------------------------------------


def generate_thermal_signature_mask(image_uint8: np.ndarray, threshold: int) -> np.ndarray:
    """
    Interactive Thermal Isolation Signature Masking.

    Produces a strict binary mask (0 / 255) isolating only pixels whose
    enhanced intensity meets or exceeds the operator-supplied threshold --
    i.e. the hottest / highest-return thermal signatures in the scene.

        M(x, y) = 255  if I_enhanced(x, y) >= threshold
        M(x, y) = 0    otherwise
    """
    threshold = int(np.clip(threshold, 0, 255))
    _, mask = cv2.threshold(image_uint8, threshold, 255, cv2.THRESH_BINARY)
    return mask


def overlay_thermal_mask(
    rgb_image: np.ndarray,
    mask_uint8: np.ndarray,
    highlight_color: Tuple[int, int, int] = (255, 66, 44),
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Alpha-blends a solid highlight colour over the RGB preview wherever the
    thermal mask is active, giving analysts an immediate, high-contrast
    region-of-interest overlay without altering the underlying colourized
    export product.
    """
    if rgb_image.shape[:2] != mask_uint8.shape[:2]:
        raise ValueError("rgb_image and mask_uint8 must share the same (H, W) shape.")

    overlay = rgb_image.copy()
    highlight_layer = np.empty_like(rgb_image)
    highlight_layer[:, :] = highlight_color

    mask_bool = mask_uint8 > 0
    blended = rgb_image.astype(np.float32) * (1.0 - alpha) + highlight_layer.astype(np.float32) * alpha
    overlay[mask_bool] = blended[mask_bool].astype(np.uint8)
    return overlay
