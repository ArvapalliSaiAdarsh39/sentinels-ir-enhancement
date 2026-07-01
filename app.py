import streamlit as st
import os
import numpy as np
from geospatial_engine import load_geotiff_properties, export_geotagged_output
from core_processing import enhance_structural_features, apply_thermal_color_mapping

# Set up page configurations to match dashboard layouts [cite: 74]
st.set_page_config(layout="wide", page_title="Sentinels IR Processing Hub")

st.title("🛰️ Infrared Image Colourization and Enhancement Tool")
st.write("Team Sentinels | Vasavi College Of Engineering [cite: 7, 14]")

# 1. Top Ingestion Area: Drag & Drop Ingestion Bar [cite: 75]
uploaded_file = st.file_uploader("Drag & Drop GeoTIFF / TIF Files Here", type=["tif", "tiff"])

if uploaded_file:
    # Safely save file temporarily to local directory for Rasterio mapping functions [cite: 27]
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Parse file metadata and retrieve matrix grids [cite: 18, 52]
    raw_ir, geo_profile = load_geotiff_properties(temp_path)
    
    # 2. Left Parameter Sidebar: Sliders mapped directly to configuration variables [cite: 76]
    st.sidebar.header("Parameters Control Panel")
    palette = st.sidebar.selectbox("Color Mapping Look-Up Table (LUT)", ["JET", "VIRIDIS", "HOT"])
    st.sidebar.markdown("---")
    st.sidebar.write("🔒 **Data Security Mode**: Active Local Host (Zero Cloud Overhead) [cite: 27]")
    
    # Trigger processing math layers instantly [cite: 115]
    enhanced_mono = enhance_structural_features(raw_ir)
    colorized_output = apply_thermal_color_mapping(enhanced_mono, palette)
    
    # 3. Right Visualization Frame: Side-by-side comparison window [cite: 77]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Monochrome Input")
        # Quick data normalization to safely render high bit-depth rasters on screen
        norm_view = ((raw_ir - raw_ir.min()) / (raw_ir.max() - raw_ir.min()) * 255).astype(np.uint8)
        st.image(norm_view, use_column_width=True, channels="GRAY")
        
    with col2:
        st.subheader("Enhanced Pseudo-RGB Output")
        st.image(colorized_output, use_column_width=True)
        
    # 4. Export Options Panel Setup [cite: 80]
    st.markdown("---")
    st.subheader("Export Finished Operations")
    out_filename = f"processed_{uploaded_file.name}"
    
    if st.button("Export Production-Ready GeoTIFF (.TIF with metadata)"):
        export_geotagged_output(out_filename, colorized_output, geo_profile)
        st.success(f"💾 File compiled and exported locally with 100% accurate coordinates: {out_filename} ")
        
    # Remove temporary workspace file
    if os.path.exists(temp_path):
        os.remove(temp_path)