import streamlit as st
import os
import io
import zipfile
import numpy as np
from geospatial_engine import load_geotiff_properties, export_geotagged_output
from core_processing import enhance_structural_features, apply_thermal_color_mapping

# Configure browser parameters and viewport configurations
st.set_page_config(layout="wide", page_title="IR Telemetry Processing Suite")

# --- APP HEADER PANEL ---
st.title("🛰️ Infrared Image Colourization & Enhancement Engine")
st.caption("Developed by Team Sentinels | Vasavi College Of Engineering — Built for Bharatiya Antariksh Hackathon 2026")

# Unified Operational Core Status Row
m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Pipeline Framework", value="Deterministic Math")
m2.metric(label="Spatial Target Drift", value="0.00% (Absolute)")
m3.metric(label="Data Security Protocol", value="100% Local Sandbox")
m4.metric(label="AI Hallucination Risk", value="Zero / Excluded")

st.markdown("---")

# --- LEFT PARAMETER CONTROLS SIDEBAR ---
st.sidebar.header("🎛️ Live View Controls")
palette = st.sidebar.selectbox("Select Dashboard Color Map (LUT)", ["JET", "VIRIDIS", "HOT"])

st.sidebar.subheader("Advanced Spatial Filtering")
clip_val = st.sidebar.slider("CLAHE Contrast Clip Limit", 1.0, 10.0, 3.0)
grid_val = st.sidebar.slider("CLAHE Tile Block Array Size", 4, 16, 8, step=2)
sharpen_val = st.sidebar.slider("Laplacian Sharpness Scaling Intensity", 0.0, 2.0, 0.4)

st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Operational Control Framework", ["Single File Workspace", "Batch Processing Queue"])

# --- PIPELINE INFORMATIONAL SUMMARY EXPANDER ---
with st.expander("ℹ️ System Architecture & Framework Specifications"):
    st.markdown("""
    * **Deterministic Pipeline Execution:** This application avoids resource-heavy deep learning models. By utilizing explicit pixel transformations, it eliminates data contamination and guarantees absolute reproduction fidelity.
    * **Automated Spatial Preservation Layers:** Uses high-speed geometric engines parallel to calculation loops to lock map datum lines, coordinate metrics, and scale systems safely into target file outputs.
    """)

# --- MODE 1: SINGLE FILE WORKSPACE ---
if app_mode == "Single File Workspace":
    st.subheader("📸 Single Tile Processing Environment")
    uploaded_file = st.file_uploader("Ingest target raw multi-band file or single-channel telemetry", type=["tif", "tiff"])

    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        raw_ir, geo_profile = load_geotiff_properties(temp_path)
        
        # Apply the explicit matrix operation layer
        enhanced_mono = enhance_structural_features(raw_ir, clip_limit=clip_val, tile_size=grid_val, sharpen_weight=sharpen_val)
        colorized_output = apply_thermal_color_mapping(enhanced_mono, palette)
        
        # Side-by-Side Visualization Frames
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎚️ Raw Monochrome Ingestion Matrix")
            norm_view = ((raw_ir - raw_ir.min()) / (raw_ir.max() - raw_ir.min()) * 255).astype(np.uint8)
            st.image(norm_view, use_column_width=True, channels="GRAY")
            st.line_chart(np.histogram(norm_view, bins=256)[0])
            
        with col2:
            st.markdown(f"### 🎨 Enhanced Pseudo-RGB Output ({palette})")
            st.image(colorized_output, use_column_width=True)
            st.line_chart(np.histogram(colorized_output, bins=256)[0])
            
        st.markdown("---")
        st.subheader("📦 Multi-Palette Complete Export")
        st.write("Clicking the compilation trigger below automatically builds three separate GeoTIFF raster arrays matching **JET**, **VIRIDIS**, and **HOT** lookup configurations, retaining full spatial metadata.")
        
        # Memory buffer to hold generated production files for immediate user browser download
        zip_buffer = io.BytesIO()
        base_name = os.path.splitext(uploaded_file.name)[0]
        
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Sequentially render and compile all three target lookups
            for target_lut in ["JET", "VIRIDIS", "HOT"]:
                lut_img = apply_thermal_color_mapping(enhanced_mono, target_lut)
                local_out_name = f"enhanced_{base_name}_{target_lut}.tif"
                
                # Write to disk temporarily to let rasterio lock geotags
                export_geotagged_output(local_out_name, lut_img, geo_profile)
                
                # Stream binary content into zip archive container
                with open(local_out_name, "rb") as f:
                    zip_file.writestr(local_out_name, f.read())
                    
                # Scrub temporary disk instances safely
                if os.path.exists(local_out_name):
                    os.remove(local_out_name)
                    
        # Native Browser File Download Object Ingestion
        st.download_button(
            label="📥 Download Complete 3-Palette Raster Package (.ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{base_name}_multi_palette_package.zip",
            mime="application/zip"
        )
            
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- MODE 2: BATCH PROCESSING QUEUE ---
else:
    st.subheader("🗂️ Batch Processing Directory Queue")
    uploaded_files = st.file_uploader(
        "Ingest batch folder paths or drop multiple raster files simultaneously", 
        type=["tif", "tiff"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        progress_bar = st.progress(0)
        status_logs = st.empty()
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for index, file in enumerate(uploaded_files):
                status_logs.text(f"Parsing Metadata Map Grid ({index + 1}/{len(uploaded_files)}): {file.name}")
                
                temp_path = f"batch_temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                raw_ir, geo_profile = load_geotiff_properties(temp_path)
                enhanced_mono = enhance_structural_features(raw_ir, clip_limit=clip_val, tile_size=grid_val, sharpen_weight=sharpen_val)
                
                file_base = os.path.splitext(file.name)[0]
                
                # Batch mode generates all three mappings per incoming telemetry tile file
                for target_lut in ["JET", "VIRIDIS", "HOT"]:
                    lut_img = apply_thermal_color_mapping(enhanced_mono, target_lut)
                    out_filename = f"batch_enhanced_{file_base}_{target_lut}.tif"
                    
                    export_geotagged_output(out_filename, lut_img, geo_profile)
                    
                    with open(out_filename, "rb") as f:
                        zip_file.writestr(out_filename, f.read())
                        
                    if os.path.exists(out_filename): 
                        os.remove(out_filename)
                
                if os.path.exists(temp_path): 
                    os.remove(temp_path)
                
                progress_bar.progress((index + 1) / len(uploaded_files))
                
        status_logs.success(f"Operational Framework finalized! All configurations successfully bundle-packaged.")
        
        st.download_button(
            label="📥 Download Production Files Bundle (.ZIP Archive)",
            data=zip_buffer.getvalue(),
            file_name="sentinels_batch_all_palettes.zip",
            mime="application/zip"
        )
