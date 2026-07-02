"""
geospatial_engine.py
=====================================================================
Infrared Image Colourization & Enhancement Workstation
Geospatial I/O & Metadata Fidelity Engine
=====================================================================

This module owns every interaction with raw GeoTIFF byte-streams. It is
responsible for:

    * Parsing uploaded raster products entirely in memory (rasterio's
      MemoryFile / GDAL "vsimem" backend) so raw pixel bytes never touch
      the physical disk on the read path.
    * Extracting a complete, UI-ready metadata descriptor: dimensions,
      dtype, band count, file size, CRS identity, and the scene
      bounding box reprojected into WGS84 longitude / latitude.
    * Re-serializing processed matrices back into fully geotagged
      GeoTIFF byte-streams with the *exact* source affine transform and
      CRS re-attached, guaranteeing zero coordinate drift.
    * Bundling multiple colourized products into a single in-memory ZIP
      archive for one-click browser download.
    * Defensive, disk-hygiene utilities for the rare edge case where an
      underlying C library insists on writing a scratch file.

Architectural guarantee: every function in this module is a pure,
deterministic transformation of its inputs. Nothing here calls out to
the network, and nothing here depends on model weights of any kind.
"""

from __future__ import annotations

import glob
import io
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# GDAL, by default, likes to write ".aux.xml" sidecar files to disk whenever
# it computes raster statistics or overviews. Since this workstation
# guarantees a strict in-memory sandbox (Architectural Philosophy #3), that
# behaviour is disabled *before* rasterio/GDAL is imported so the setting is
# honoured for the entire lifetime of the process.
# ---------------------------------------------------------------------------
os.environ.setdefault("GDAL_PAM_ENABLED", "NO")

import numpy as np
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

try:
    import pyproj

    _PYPROJ_AVAILABLE = True
except ImportError:  # pragma: no cover - pyproj ships as a rasterio dependency
    _PYPROJ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GeoTiffValidationError(Exception):
    """Raised whenever an uploaded artefact fails structural, format, or
    CRS validation and must be rejected before it ever reaches the
    processing pipeline."""


# ---------------------------------------------------------------------------
# Metadata descriptor
# ---------------------------------------------------------------------------


@dataclass
class RasterMetadata:
    """A complete, display-ready snapshot of a parsed raster's identity.

    This dataclass is deliberately flat (no nested objects) so it can be
    dropped straight into Streamlit UI calls without any translation
    layer, while still being fully typed for downstream logic.
    """

    filename: str
    width: int
    height: int
    band_count: int
    dtype: str
    driver: str
    file_size_bytes: int
    crs_string: str
    crs_epsg: Optional[int]
    min_lon: Optional[float]
    min_lat: Optional[float]
    max_lon: Optional[float]
    max_lat: Optional[float]

    def file_size_human(self) -> str:
        """Formats the raw byte count as a human-readable size string."""
        size = float(self.file_size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"

    def dimension_string(self) -> str:
        """Formats matrix dimensions as 'W x H px | N band(s)'."""
        band_label = "band" if self.band_count == 1 else "bands"
        return f"{self.width} x {self.height} px  |  {self.band_count} {band_label}"

    def has_geographic_bounds(self) -> bool:
        """True only if all four WGS84 bounding-box corners were resolved."""
        return None not in (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


# ---------------------------------------------------------------------------
# Disk-hygiene utilities
# ---------------------------------------------------------------------------


class TempWorkspace:
    """
    Context manager for any incidental on-disk scratch space.

    The primary read/write path of this engine is 100% memory-resident
    (MemoryFile / BytesIO), so this workspace is typically never written
    to. It exists as a defensive guarantee: if any downstream library
    call ever drops a scratch file to disk, it will land inside an
    isolated, single-purpose temp directory that is unconditionally
    removed the moment the `with` block exits -- even on exception.
    """

    def __init__(self, prefix: str = "irw_scratch_"):
        self._prefix = prefix
        self.path: Optional[str] = None

    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix=self._prefix)
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        stale_path, self.path = self.path, None
        if stale_path and os.path.isdir(stale_path):
            shutil.rmtree(stale_path, ignore_errors=True)
        return False  # never suppress exceptions raised inside the block


def sweep_temp_artifacts(directory: str) -> int:
    """
    Defensive cleanup sweep. Walks `directory` and removes every file
    fragment found. Returns the number of files removed. Safe to call on
    a directory that does not exist, is empty, or was already cleaned.
    """
    removed = 0
    if not directory or not os.path.isdir(directory):
        return removed
    for root, _dirs, files in os.walk(directory):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
                removed += 1
            except OSError:
                continue
    return removed


def cleanup_stale_workspaces() -> int:
    """
    Removes any leftover TempWorkspace directories from a prior run that
    may not have been cleaned up (e.g. the process was killed mid-run).
    Safe to call once at application startup as a defence-in-depth sweep.
    """
    removed = 0
    pattern = os.path.join(tempfile.gettempdir(), "irw_scratch_*")
    for stale_dir in glob.glob(pattern):
        if os.path.isdir(stale_dir):
            shutil.rmtree(stale_dir, ignore_errors=True)
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------


def _describe_crs(crs: Optional[CRS]) -> Tuple[str, Optional[int]]:
    """Builds a human-friendly 'EPSG:xxxx / Name' string for a CRS object,
    matching the format an analyst expects (e.g. 'EPSG:4326 / WGS 84')."""
    if crs is None:
        return "Undefined / No CRS Detected", None

    epsg = crs.to_epsg()
    friendly_name: Optional[str] = None

    if _PYPROJ_AVAILABLE:
        try:
            friendly_name = pyproj.CRS.from_user_input(crs.to_wkt()).name
        except Exception:
            friendly_name = None

    if epsg and friendly_name:
        return f"EPSG:{epsg} / {friendly_name}", epsg
    if epsg:
        return f"EPSG:{epsg}", epsg
    if friendly_name:
        return friendly_name, None
    return crs.to_string(), None


def _compute_wgs84_bounds(
    crs: Optional[CRS], bounds
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Reprojects native raster bounds into WGS84 lon/lat degrees, if the
    source CRS is known. Returns (min_lon, min_lat, max_lon, max_lat)."""
    if crs is None:
        return None, None, None, None
    try:
        min_lon, min_lat, max_lon, max_lat = transform_bounds(crs, "EPSG:4326", *bounds)
        return min_lon, min_lat, max_lon, max_lat
    except Exception:
        return None, None, None, None


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------


def load_geotiff(
    file_bytes: bytes, filename: str
) -> Tuple[np.ndarray, RasterMetadata, Affine, Optional[CRS]]:
    """
    Parses a raw GeoTIFF byte-stream entirely in memory.

    Parameters
    ----------
    file_bytes : the raw uploaded bytes (e.g. from a Streamlit UploadedFile)
    filename    : the original filename, used only for diagnostics/exports

    Returns
    -------
    array      : np.ndarray, shape (bands, height, width) -- native dtype
    metadata   : RasterMetadata -- fully populated, UI-ready descriptor
    transform  : rasterio.transform.Affine -- exact source affine transform
    crs        : rasterio.crs.CRS or None  -- exact source CRS object

    Raises
    ------
    GeoTiffValidationError on any structural, format, or I/O failure. This
    is the single exception type callers need to catch to present a clean
    user-facing error banner instead of crashing.
    """
    if not file_bytes:
        raise GeoTiffValidationError("The uploaded file is empty (0 bytes) and cannot be parsed.")

    try:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as dataset:
                if dataset.count < 1:
                    raise GeoTiffValidationError(
                        f"'{filename}' contains zero raster bands and is not a usable product."
                    )

                array = dataset.read()
                band_count, height, width = array.shape

                crs = dataset.crs
                transform = dataset.transform
                crs_string, crs_epsg = _describe_crs(crs)
                min_lon, min_lat, max_lon, max_lat = _compute_wgs84_bounds(crs, dataset.bounds)

                metadata = RasterMetadata(
                    filename=filename,
                    width=width,
                    height=height,
                    band_count=band_count,
                    dtype=str(array.dtype),
                    driver=dataset.driver or "Unknown",
                    file_size_bytes=len(file_bytes),
                    crs_string=crs_string,
                    crs_epsg=crs_epsg,
                    min_lon=min_lon,
                    min_lat=min_lat,
                    max_lon=max_lon,
                    max_lat=max_lat,
                )

                return array, metadata, transform, crs

    except GeoTiffValidationError:
        raise
    except RasterioIOError as exc:
        raise GeoTiffValidationError(
            f"'{filename}' could not be opened as a valid GeoTIFF raster. The file may be "
            f"corrupted, truncated, or in an unsupported format. (Driver error: {exc})"
        ) from exc
    except Exception as exc:  # noqa: BLE001 -- final safety net around 3rd-party C bindings
        raise GeoTiffValidationError(
            f"An unexpected error occurred while parsing '{filename}': {exc}"
        ) from exc


def write_rgb_geotiff(rgb_array: np.ndarray, transform: Affine, crs: Optional[CRS]) -> bytes:
    """
    Serializes an (H, W, 3) uint8 matrix into a fully geotagged, 3-band
    GeoTIFF byte-stream. The supplied `transform` and `crs` are written
    back verbatim onto the new product -- this is the mechanism behind
    the 'zero coordinate drift' guarantee.
    """
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) array, received shape {rgb_array.shape}.")

    height, width, _ = rgb_array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "photometric": "RGB",
        "compress": "LZW",
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            for band_index in range(3):
                dataset.write(
                    np.ascontiguousarray(rgb_array[:, :, band_index]).astype("uint8"),
                    band_index + 1,
                )
        return memfile.read()


def write_single_band_geotiff(band_array: np.ndarray, transform: Affine, crs: Optional[CRS]) -> bytes:
    """Serializes a single-band (H, W) uint8 matrix to geotagged GeoTIFF bytes."""
    height, width = band_array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "LZW",
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(np.ascontiguousarray(band_array).astype("uint8"), 1)
        return memfile.read()


def build_multi_palette_archive(
    palette_products: Dict[str, np.ndarray],
    transform: Affine,
    crs: Optional[CRS],
    source_filename: str,
) -> bytes:
    """
    Accepts `{palette_name: (H, W, 3) rgb_array}` for every simultaneously
    generated LUT product, writes each as an independently geotagged
    GeoTIFF, and bundles the full set plus a manifest into a single
    in-memory ZIP archive stream ready for immediate browser download.

    Every product in the archive carries the identical `transform` / `crs`
    pair as the source raster -- the export pipeline never resamples,
    reprojects, or otherwise perturbs the spatial reference.
    """
    base_name = os.path.splitext(os.path.basename(source_filename))[0] or "ir_product"
    zip_buffer = io.BytesIO()

    crs_label, _ = _describe_crs(crs)
    manifest_lines = [
        "INFRARED IMAGE COLOURIZATION & ENHANCEMENT WORKSTATION",
        "Multi-Palette Bundled Export Manifest",
        "=" * 60,
        f"Source Raster            : {source_filename}",
        f"Coordinate Reference Sys : {crs_label}",
        f"Products in this package : {len(palette_products)}",
        "",
        "Contents:",
    ]

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for palette_name, rgb_array in palette_products.items():
            geotiff_bytes = write_rgb_geotiff(rgb_array, transform, crs)
            out_name = f"{base_name}_{palette_name.lower()}.tif"
            archive.writestr(out_name, geotiff_bytes)
            manifest_lines.append(
                f"  - {out_name}   [{palette_name} LUT | "
                f"{rgb_array.shape[1]}x{rgb_array.shape[0]} px | {len(geotiff_bytes) / 1024:.1f} KB]"
            )
        archive.writestr("MANIFEST.txt", "\n".join(manifest_lines))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
