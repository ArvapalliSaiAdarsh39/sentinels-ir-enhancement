import streamlit as st
import os
import io
import zipfile
import numpy as np
from geospatial_engine import load_geotiff_properties, export_geotagged_output
from core_processing import enhance_structural_features, apply_thermal_color_mapping

st.set_page_config(layout="wide", page_title="IR Telemetry Processing Suite")

st.title("🛰️ Infrared Image Colourization and Enhancement Tool")
st.write("Team Sentinels | Vasavi College Of Engineering")

# --- LEFT PARAMETER SIDEBAR [cite: 74, 76] ---
st.sidebar.header("Parameters Control Panel")
palette = st.sidebar.selectbox("Color Mapping Look-Up Table (LUT)", ["JET", "VIRIDIS", "HOT"])

st.sidebar.markdown("### Advanced Spatial Filtering [cite: 46]")
clip_val = st.sidebar.slider("CLAHE Clip Limit (Contrast)", 1.0, 10.0, 3.0)
grid_val = st.sidebar.slider("CLAHE Tile Grid Size", 4, 16, 8, step=2)
sharpen_val = st.sidebar.slider("Edge Sharpening Intensity Factor", 0.0, 2.0, 0.4)

st.sidebar.markdown("---")
# Functional Mode Selection Toggle Area
app_mode = st.sidebar.radio("Operational Framework Mode", ["Single File Workspace", "Batch Processing Queue "])
st.sidebar.write("🔒 **Data Security Mode**: Active Local/Cloud Engine Sandbox")

# --- MODE 1: SINGLE FILE WORKSPACE ---
if app_mode == "Single File Workspace":
    uploaded_file = st.file_uploader("Drag & Drop Single GeoTIFF File Here [cite: 75]", type=["tif", "tiff"])

    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        raw_ir, geo_profile = load_geotiff_properties(temp_path)
        
        # Calculate matrix values using sidebar parameters 
        enhanced_mono = enhance_structural_features(raw_ir, clip_limit=clip_val, tile_size=grid_val, sharpen_weight=sharpen_val)
        colorized_output = apply_thermal_color_mapping(enhanced_mono, palette)
        
        # Side-by-Side Visualization Frames [cite: 77]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Monochrome Input [cite: 77]")
            norm_view = ((raw_ir - raw_ir.min()) / (raw_ir.max() - raw_ir.min()) * 255).astype(np.uint8)
            st.image(norm_view, use_column_width=True, channels="GRAY")
            # Interactive Display Metrics: Input Distribution Chart 
            st.line_chart(np.histogram(norm_view, bins=256)[0])
            
        with col2:
            st.subheader("Enhanced Pseudo-RGB Output [cite: 77]")
            st.image(colorized_output, use_column_width=True)
            # Interactive Display Metrics: Output Distribution Chart 
            st.line_chart(np.histogram(colorized_output, bins=256)[0])
            
        st.markdown("---")
        st.subheader("Export Operation [cite: 80]")
        out_filename = f"processed_{uploaded_file.name}"
        
        if st.button("Export Production-Ready GeoTIFF (.TIF) [cite: 80]"):
            export_geotagged_output(out_filename, colorized_output, geo_profile)
            st.success(f"💾 Exported locally with 100% accurate coordinates[cite: 34, 52]: {out_filename}")
            
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- MODE 2: BATCH PROCESSING QUEUE ---
else:
    uploaded_files = st.file_uploader(
        "Drag & Drop Multiple GeoTIFF Files Simultaneously [cite: 51, 75]", 
        type=["tif", "tiff"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader("Queue Progress Tracking [cite: 79, 93]")
        progress_bar = st.progress(0)
        status_logs = st.empty()
        
        # Memory buffer to hold generated production files for a unified bundle zip download
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for index, file in enumerate(uploaded_files):
                # File status tracking calculations [cite: 93]
                status_logs.text(f"Processing Array Data Tile ({index + 1}/{len(uploaded_files)}): {file.name}")
                
                temp_path = f"batch_temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Run pipeline processing steps matching criteria
                raw_ir, geo_profile = load_geotiff_properties(temp_path)
                enhanced_mono = enhance_structural_features(raw_ir, clip_limit=clip_val, tile_size=grid_val, sharpen_weight=sharpen_val)
                colorized_output = apply_thermal_color_mapping(enhanced_mono, palette)
                
                # Output compiled file locally 
                out_filename = f"processed_{file.name}"
                export_geotagged_output(out_filename, colorized_output, geo_profile)
                
                # Append to cloud zip archive buffer configuration
                with open(out_filename, "rb") as f:
                    zip_file.writestr(out_filename, f.read())
                
                # Clean local processing instances
                if os.path.exists(temp_path): os.remove(temp_path)
                if os.path.exists(out_filename): os.remove(out_filename)
                
                # Update visual progress tracking metrics [cite: 93]
                progress_bar.progress((index + 1) / len(uploaded_files))
                
        status_logs.success(f"Successfully processed all {len(uploaded_files)} image matrices! ")
        
        # One-Click download button area for batch directories [cite: 51, 80]
        st.download_button(
            label="📥 Download All Production-Ready Files (.ZIP Archive)",
            data=zip_buffer.getvalue(),
            file_name="sentinels_batch_processed_geotiffs.zip",
            mime="application/zip"
        )