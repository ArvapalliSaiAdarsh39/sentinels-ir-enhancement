"""
app.py
=====================================================================
Infrared Image Colourization & Enhancement Workstation
Primary Streamlit Command Dashboard — ISRO Deep Space Console Theme
=====================================================================

Orchestrates the full operator workflow:

    1. STAGE   -- operator drops a raw GeoTIFF into the ingestion bay;
                  the file is parsed and an "ISRO Telemetry Staging Box"
                  is rendered. No heavy computation happens here.
    2. TUNE    -- operator optionally adjusts the dual-stage filter
                  parameters and selects the working band in the
                  sidebar "Mission Control" panel.
    3. IGNITE  -- operator explicitly clicks "Initialize Computational
                  Ingestion Engine". Only at this point does the
                  bilateral / CLAHE / sharpen / multi-palette pipeline
                  execute.
    4. ANALYZE -- side-by-side comparison, synchronized histograms,
                  real-time thermal isolation masking, and a full
                  metadata / bounding-box audit desk render from the
                  cached pipeline output. Nudging the thermal threshold
                  slider or swapping the preview palette never re-runs
                  the heavy pipeline -- both operate on already-computed
                  arrays cached in session_state.
    5. EXPORT  -- operator compiles a single ZIP package containing
                  JET / VIRIDIS / HOT geotagged GeoTIFF products.

Visual design note: the cosmic background, card system, and orbiting
satellite signature element are implemented as pure CSS (no external
@import / CDN font or image calls) so the interface stays inside the
Strict Data Sandbox guarantee (Architectural Philosophy #3) even while
rendering a fully animated console shell.
"""

from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import core_processing
import geospatial_engine

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IR Colourization & Enhancement Workstation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling — ISRO Deep Space Console Theme
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    """Injects the full console theme: colour tokens, animated starfield /
    nebula background, card system, and interactive hover transitions.
    Everything here is pure CSS -- no network calls of any kind."""
    st.markdown(
        """
        <style>
        :root {
            --irw-bg: #0B0F19;
            --irw-bg-elevated: #121A2C;
            --irw-bg-elevated-2: #17233A;
            --irw-text: #FFFFFF;
            --irw-text-muted: #92A0B8;
            --irw-accent: #FF6B35;
            --irw-accent-soft: rgba(255, 107, 53, 0.14);
            --irw-accent-2: #4FD8E8;
            --irw-accent-2-soft: rgba(79, 216, 232, 0.12);
            --irw-border: rgba(255, 255, 255, 0.09);
            --irw-font-body: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --irw-font-mono: "JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
        }

        html { scroll-behavior: smooth; }

        /* ---- Cosmic background: layered starfield + nebula wash, driven
           entirely by an animated `background-position` on .stApp itself.
           No extra DOM nodes, no z-index stacking games -- just a CSS
           background, which always paints behind normal page content. ---- */
        .stApp {
            background-color: var(--irw-bg);
            background-image:
                radial-gradient(1.6px 1.6px at 40px 60px, rgba(255,255,255,0.85), transparent 100%),
                radial-gradient(1.2px 1.2px at 140px 20px, rgba(255,255,255,0.65), transparent 100%),
                radial-gradient(1.4px 1.4px at 220px 140px, rgba(255,255,255,0.75), transparent 100%),
                radial-gradient(1.1px 1.1px at 40px 220px, rgba(255,255,255,0.55), transparent 100%),
                radial-gradient(1.3px 1.3px at 260px 260px, rgba(255,255,255,0.70), transparent 100%),
                radial-gradient(1.2px 1.2px at 180px 100px, rgba(255,255,255,0.60), transparent 100%),
                radial-gradient(ellipse 60% 45% at 16% 12%, rgba(255,107,53,0.10), transparent 60%),
                radial-gradient(ellipse 55% 42% at 84% 78%, rgba(79,216,232,0.09), transparent 60%);
            background-repeat: repeat, repeat, repeat, repeat, repeat, repeat, no-repeat, no-repeat;
            background-size:
                300px 300px, 300px 300px, 300px 300px,
                300px 300px, 300px 300px, 300px 300px,
                140% 140%, 140% 140%;
            background-position:
                0px 0px, 0px 0px, 0px 0px, 0px 0px, 0px 0px, 0px 0px, 0% 0%, 100% 100%;
            animation: irw-starfield-drift 160s linear infinite;
            color: var(--irw-text);
            font-family: var(--irw-font-body);
        }

        @keyframes irw-starfield-drift {
            from {
                background-position:
                    0px 0px, 0px 0px, 0px 0px, 0px 0px, 0px 0px, 0px 0px, 0% 0%, 100% 100%;
            }
            to {
                background-position:
                    300px 300px, -300px 300px, 300px -300px,
                    -300px -300px, 150px 300px, -300px 150px, 0% 0%, 100% 100%;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .stApp { animation: none; }
            .irw-orbit-path { animation: none !important; }
        }

        section[data-testid="stSidebar"] {
            background: var(--irw-bg-elevated);
            border-right: 1px solid var(--irw-border);
        }
        section[data-testid="stSidebar"] * { color: var(--irw-text); }

        h1, h2, h3 { color: var(--irw-text); letter-spacing: 0.2px; }

        .irw-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: 0.2px;
            background: linear-gradient(90deg, #FFFFFF 0%, var(--irw-accent) 130%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.15rem;
        }
        .irw-subtitle {
            color: var(--irw-text-muted);
            font-size: 0.96rem;
            margin-bottom: 1.4rem;
        }

        .irw-card {
            background: rgba(255,255,255,0.035);
            border: 1px solid var(--irw-border);
            border-radius: 14px;
            padding: 14px 18px;
            height: 100%;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .irw-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255,107,53,0.45);
            box-shadow: 0 10px 26px rgba(255,107,53,0.12);
        }
        .irw-card-label {
            color: var(--irw-text-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 4px;
        }
        .irw-card-value {
            font-family: var(--irw-font-mono);
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--irw-text);
        }
        .irw-card-accent { color: var(--irw-accent); }
        .irw-card-cool   { color: var(--irw-accent-2); }

        .irw-staging-box {
            background: var(--irw-accent-soft);
            border: 1px solid rgba(255,107,53,0.35);
            border-radius: 14px;
            padding: 18px 22px;
            margin: 0.6rem 0 1.1rem 0;
            transition: box-shadow 0.25s ease;
        }
        .irw-staging-box:hover { box-shadow: 0 10px 30px rgba(255,107,53,0.10); }
        .irw-staging-title {
            color: var(--irw-accent);
            font-weight: 800;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-bottom: 10px;
        }
        .irw-staging-box table { width: 100%; border-collapse: collapse; }
        .irw-staging-box table td { padding: 5px 0; font-size: 0.93rem; }
        .irw-staging-box table td:first-child { color: var(--irw-text-muted); width: 34%; }
        .irw-staging-box table td:last-child  { color: var(--irw-text); font-family: var(--irw-font-mono); }

        .irw-section-heading {
            font-size: 1.08rem;
            font-weight: 800;
            color: var(--irw-text);
            border-left: 3px solid var(--irw-accent);
            padding-left: 11px;
            margin: 1.7rem 0 0.65rem 0;
        }

        .stButton > button {
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
            border-radius: 10px !important;
        }
        .stButton > button:hover { transform: translateY(-1px); filter: brightness(1.06); }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--irw-accent) 0%, #FF3D6B 100%) !important;
            border: none !important;
            font-weight: 700 !important;
            letter-spacing: 0.02em;
            box-shadow: 0 6px 20px rgba(255,107,53,0.25);
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 10px 28px rgba(255,107,53,0.40);
        }

        [data-testid="stMetricValue"] { color: var(--irw-text); }

        div[data-testid="stExpander"] {
            border: 1px solid var(--irw-border) !important;
            border-radius: 12px !important;
            transition: border-color 0.2s ease;
        }
        div[data-testid="stExpander"]:hover { border-color: rgba(79,216,232,0.35) !important; }

        /* ---- Orbiting satellite — the one signature motion element ---- */
        .irw-orbit-stage {
            position: fixed;
            top: 10px;
            right: 20px;
            width: 128px;
            height: 128px;
            pointer-events: none;
            z-index: 999;
            opacity: 0.92;
        }
        .irw-orbit-path {
            position: relative;
            width: 100%;
            height: 100%;
            animation: irw-orbit-spin 17s linear infinite;
        }
        .irw-orbit-path::before {
            content: "";
            position: absolute;
            inset: 20px;
            border: 1px dashed rgba(79,216,232,0.28);
            border-radius: 50%;
        }
        .irw-satellite-icon {
            position: absolute;
            top: 6px;
            left: 50%;
            transform: translateX(-50%);
            filter: drop-shadow(0 0 6px rgba(255,107,53,0.6));
        }
        @keyframes irw-orbit-spin {
            from { transform: rotate(0deg); }
            to   { transform: rotate(360deg); }
        }
        @media (max-width: 900px) {
            .irw-orbit-stage { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_cosmic_accents() -> None:
    """Renders the single small fixed orbiting-satellite signature element.
    Kept as one lightweight, pointer-events-disabled node so it never
    interferes with any interactive widget underneath it."""
    st.markdown(
        """
        <div class="irw-orbit-stage">
          <div class="irw-orbit-path">
            <div class="irw-satellite-icon">
              <svg viewBox="0 0 64 40" width="34" height="22" xmlns="http://www.w3.org/2000/svg">
                <rect x="3" y="14" width="15" height="12" fill="#4FD8E8" opacity="0.9"/>
                <rect x="46" y="14" width="15" height="12" fill="#4FD8E8" opacity="0.9"/>
                <rect x="25" y="11" width="14" height="18" rx="2.5" fill="#FF6B35"/>
                <line x1="32" y1="11" x2="32" y2="3" stroke="#FFFFFF" stroke-width="1.5"/>
                <circle cx="32" cy="2.5" r="2" fill="#FFFFFF"/>
              </svg>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    defaults = {
        "staged": None,             # dict: raw_array, metadata, transform, crs
        "staged_signature": None,   # (filename, size) fingerprint of last parsed upload
        "processed_results": None,  # dict: normalized, enhanced, palette_products, histograms
        "export_archive": None,     # bytes: last compiled ZIP package
        "export_filename": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Header + top metrics status panel
# ---------------------------------------------------------------------------


def render_header() -> None:
    st.markdown(
        '<div class="irw-title">🛰️ Infrared Image Colourization &amp; Enhancement Workstation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="irw-subtitle">Deterministic multi-band GeoTIFF processing for '
        'infrared / thermal satellite raster products — Bharatiya Antariksh Hackathon 2026.</div>',
        unsafe_allow_html=True,
    )

    metrics = [
        ("Pipeline Architecture", "Deterministic Matrix Ops", "accent"),
        ("Coordinate Preservation", "100% Locked", "cool"),
        ("Data Privacy Sandbox", "Active / Offline", "accent"),
        ("AI Hallucination Vector", "0% — No Generative Models", "cool"),
    ]
    columns = st.columns(4)
    for column, (label, value, tone) in zip(columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class="irw-card">
                    <div class="irw-card-label">{label}</div>
                    <div class="irw-card-value irw-card-{tone}">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.write("")


# ---------------------------------------------------------------------------
# Upload + staging
# ---------------------------------------------------------------------------


def handle_upload() -> None:
    st.markdown('<div class="irw-section-heading">📥 Raster Ingestion Bay</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a raw multi-band GeoTIFF (.tif / .tiff) satellite product",
        type=["tif", "tiff"],
        help="Files are parsed for diagnostics only at this stage. No enhancement math runs yet.",
    )

    if uploaded_file is None:
        return

    file_bytes = uploaded_file.getvalue()
    signature = (uploaded_file.name, len(file_bytes))

    if signature == st.session_state.staged_signature:
        return  # this exact file is already staged — avoid re-parsing on every rerun

    st.session_state.staged_signature = signature
    st.session_state.processed_results = None
    st.session_state.export_archive = None

    try:
        raw_array, metadata, transform, crs = geospatial_engine.load_geotiff(file_bytes, uploaded_file.name)
        st.session_state.staged = {
            "raw_array": raw_array,
            "metadata": metadata,
            "transform": transform,
            "crs": crs,
        }
    except geospatial_engine.GeoTiffValidationError as exc:
        st.session_state.staged = None
        st.error(f"🚫 Ingestion Rejected — {exc}")


def render_staging_box(metadata: geospatial_engine.RasterMetadata) -> None:
    st.markdown(
        f"""
        <div class="irw-staging-box">
            <div class="irw-staging-title">✅ ISRO Telemetry Staging Box — Awaiting Operator Trigger</div>
            <table>
                <tr><td>Filename</td><td>{metadata.filename}</td></tr>
                <tr><td>Matrix Dimensions</td><td>{metadata.dimension_string()}</td></tr>
                <tr><td>File Size</td><td>{metadata.file_size_human()}</td></tr>
                <tr><td>Detected CRS</td><td>{metadata.crs_string}</td></tr>
                <tr><td>Source Dtype / Driver</td><td>{metadata.dtype} / {metadata.driver}</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar — Mission Control
# ---------------------------------------------------------------------------


def render_sidebar_controls(band_count: int):
    with st.sidebar:
        st.markdown("### 🎛️ Mission Control")
        st.caption("Dual-stage filter tuning — applied on next ignition.")

        band_index = 0
        if band_count > 1:
            band_index = st.selectbox(
                "Working Band (IR / Thermal Channel)",
                options=list(range(band_count)),
                format_func=lambda i: f"Band {i + 1}",
                index=0,
            )

        with st.expander("Stage 1 — Bilateral Noise Filter", expanded=False):
            bilateral_diameter = st.slider("Pixel Neighbourhood Diameter", 3, 15, 9, step=2)
            sigma_color = st.slider("Sigma — Colour Similarity", 10, 150, 75)
            sigma_space = st.slider("Sigma — Spatial Proximity", 10, 150, 75)

        with st.expander("Stage 2 — CLAHE Contrast Enhancement", expanded=False):
            clip_limit = st.slider("Clip Limit", 1.0, 6.0, 2.5, step=0.1)
            tile_size = st.slider("Tile Grid Size", 4, 16, 8, step=2)

        with st.expander("Edge Accentuation", expanded=False):
            sharpen_strength = st.slider("Laplacian Sharpen Strength", 0.0, 1.0, 0.35, step=0.05)

        with st.expander("Contrast Stretch Range", expanded=False):
            percentile_range = st.slider(
                "Percentile Clip — Low % / High %", 0.0, 10.0, (2.0, 2.0), step=0.5
            )

        params = core_processing.EnhancementParameters(
            bilateral_diameter=int(bilateral_diameter),
            bilateral_sigma_color=float(sigma_color),
            bilateral_sigma_space=float(sigma_space),
            clahe_clip_limit=float(clip_limit),
            clahe_tile_grid=(int(tile_size), int(tile_size)),
            sharpen_strength=float(sharpen_strength),
            lower_percentile=float(percentile_range[0]),
            upper_percentile=100.0 - float(percentile_range[1]),
        )
        return params, int(band_index)


# ---------------------------------------------------------------------------
# Ignition — the one place the heavy pipeline actually runs
# ---------------------------------------------------------------------------


def run_ignition(params: "core_processing.EnhancementParameters", band_index: int) -> None:
    staged = st.session_state.staged
    try:
        band = core_processing.select_band(staged["raw_array"], band_index)
        band_stats = core_processing.compute_band_statistics(band)

        normalized, enhanced = core_processing.run_dual_stage_pipeline(band, params)
        palette_products = core_processing.generate_multi_palette_products(enhanced)

        st.session_state.processed_results = {
            "band_index": band_index,
            "band_stats": band_stats,
            "normalized": normalized,
            "enhanced": enhanced,
            "palette_products": palette_products,
            "raw_histogram": core_processing.compute_histogram(normalized),
            "enhanced_histogram": core_processing.compute_histogram(enhanced),
        }
        st.session_state.export_archive = None
    except Exception as exc:  # noqa: BLE001 — surfaced as a clean banner, never a crash
        st.session_state.processed_results = None
        st.error(f"🚫 Computational Ingestion Engine Failure — {exc}")


# ---------------------------------------------------------------------------
# Results — visualization
# ---------------------------------------------------------------------------


def render_visualization_row(results: dict) -> None:
    st.markdown(
        '<div class="irw-section-heading">🖼️ Raw vs. Pseudo-RGB Comparative Viewframes</div>',
        unsafe_allow_html=True,
    )

    palette_names = list(results["palette_products"].keys())
    left, right = st.columns(2)

    with left:
        st.caption("Raw Monochrome (contrast-stretched, pre-enhancement)")
        preview = core_processing.downsample_for_preview(results["normalized"])
        st.image(preview, use_container_width=True, clamp=True)

    with right:
        selected_palette = st.selectbox(
            "Pseudo-RGB Preview Palette", palette_names, index=0, key="preview_palette"
        )
        st.caption(f"Enhanced & Colourized — {selected_palette} LUT")
        preview_rgb = core_processing.downsample_for_preview(results["palette_products"][selected_palette])
        st.image(preview_rgb, use_container_width=True, clamp=True)


def render_histogram_row(results: dict) -> None:
    st.markdown(
        '<div class="irw-section-heading">📊 Synchronized Pixel Intensity Histograms</div>',
        unsafe_allow_html=True,
    )

    bins = np.arange(256)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=bins,
            y=results["raw_histogram"],
            mode="lines",
            name="Raw (Pre-Enhancement)",
            line=dict(color="#4FD8E8", width=1.6),
            fill="tozeroy",
            fillcolor="rgba(79,216,232,0.10)",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=bins,
            y=results["enhanced_histogram"],
            mode="lines",
            name="Enhanced (Post Dual-Stage)",
            line=dict(color="#FF6B35", width=1.6),
            fill="tozeroy",
            fillcolor="rgba(255,107,53,0.12)",
        )
    )
    figure.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C7D2E0"),
        xaxis=dict(title="Pixel Intensity (0-255)", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Pixel Count", gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(figure, use_container_width=True, theme=None)


def render_thermal_isolation_panel(results: dict) -> None:
    st.markdown(
        '<div class="irw-section-heading">🔥 Interactive Thermal Isolation Signature Masking</div>',
        unsafe_allow_html=True,
    )

    threshold = st.slider(
        "High-Pass Segmentation Threshold (isolates the hottest signatures)",
        min_value=0,
        max_value=255,
        value=200,
        key="thermal_threshold",
    )

    mask = core_processing.generate_thermal_signature_mask(results["enhanced"], threshold)
    base_palette = st.session_state.get("preview_palette") or list(results["palette_products"].keys())[0]
    base_rgb = results["palette_products"][base_palette]
    overlay = core_processing.overlay_thermal_mask(base_rgb, mask)
    isolated_fraction = float(np.mean(mask > 0)) * 100.0

    left, right = st.columns([3, 1])
    with left:
        st.image(core_processing.downsample_for_preview(overlay), use_container_width=True, clamp=True)
    with right:
        st.markdown(
            f"""
            <div class="irw-card">
                <div class="irw-card-label">Isolated Area</div>
                <div class="irw-card-value irw-card-accent">{isolated_fraction:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Pixels at or above the threshold, highlighted on the overlay at left. Moving the slider recomputes the mask instantly — no re-ignition required.")


def render_metadata_audit_desk(metadata: geospatial_engine.RasterMetadata, band_stats: dict) -> None:
    st.markdown(
        '<div class="irw-section-heading">🌐 Metadata Audit &amp; Bounding Box Tracking Desk</div>',
        unsafe_allow_html=True,
    )

    if metadata.has_geographic_bounds():
        bbox_cells = [
            ("Min Longitude", f"{metadata.min_lon:.5f}\u00b0"),
            ("Max Longitude", f"{metadata.max_lon:.5f}\u00b0"),
            ("Min Latitude", f"{metadata.min_lat:.5f}\u00b0"),
            ("Max Latitude", f"{metadata.max_lat:.5f}\u00b0"),
        ]
        columns = st.columns(4)
        for column, (label, value) in zip(columns, bbox_cells):
            with column:
                st.markdown(
                    f"""<div class="irw-card"><div class="irw-card-label">{label}</div>
                    <div class="irw-card-value irw-card-cool">{value}</div></div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Bounding box unavailable — the source raster carries no embedded CRS.")

    info_columns = st.columns(3)
    with info_columns[0]:
        st.markdown(
            f"""<div class="irw-card"><div class="irw-card-label">Coordinate Reference System</div>
            <div class="irw-card-value">{metadata.crs_string}</div></div>""",
            unsafe_allow_html=True,
        )
    with info_columns[1]:
        st.markdown(
            f"""<div class="irw-card"><div class="irw-card-label">Raw Band Range (min / max)</div>
            <div class="irw-card-value">{band_stats['min']:.1f} / {band_stats['max']:.1f}</div></div>""",
            unsafe_allow_html=True,
        )
    with info_columns[2]:
        st.markdown(
            f"""<div class="irw-card"><div class="irw-card-label">Raw Band Mean &plusmn; Std</div>
            <div class="irw-card-value">{band_stats['mean']:.1f} &plusmn; {band_stats['std']:.1f}</div></div>""",
            unsafe_allow_html=True,
        )


def render_technical_documentation() -> None:
    with st.expander("📚 Technical Documentation — Underlying Mathematics", expanded=False):
        st.markdown("**1. Percentile Contrast Stretch**")
        st.latex(
            r"I_{norm}(x,y) = 255 \times \mathrm{clip}"
            r"\left(\frac{I(x,y) - P_{low}}{P_{high} - P_{low}},\ 0,\ 1\right)"
        )
        st.caption(
            "P_low / P_high are the operator-tunable low/high percentile clip points, "
            "computed over finite pixel values only."
        )

        st.markdown("**2. Bilateral Sensor Noise Filter (Stage 1)**")
        st.latex(
            r"I_{f}(x) = \frac{1}{W_p}\sum_{x_i \in S} I(x_i)\, "
            r"f_r(\lVert I(x_i)-I(x)\rVert)\, g_s(\lVert x_i-x\rVert)"
        )
        st.caption(
            "f_r weights by intensity similarity, g_s weights by spatial proximity — jointly "
            "preserving edges while suppressing stochastic sensor grain."
        )

        st.markdown("**3. CLAHE — Contrast-Limited Adaptive Histogram Equalization (Stage 2)**")
        st.markdown(
            "Each tile's local histogram is clipped at `clip_limit x mean_bin_height`; the "
            "clipped mass is redistributed uniformly across all bins before equalization, "
            "which prevents the noise amplification a naive global histogram equalization "
            "would introduce."
        )

        st.markdown("**4. Laplacian Edge Accentuation**")
        st.latex(
            r"I_{sharp} = I + \lambda \nabla^2 I, \qquad "
            r"\nabla^2 = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}"
        )
        st.caption("Lambda is the operator-tunable sharpen strength; the kernel isolates second-derivative edge energy.")

        st.markdown("**5. Look-Up Table Colourization**")
        st.markdown(
            "Each 8-bit intensity value in `[0, 255]` is remapped through a fixed, precomputed "
            "256-entry RGB table (JET / VIRIDIS / HOT) — a pure table lookup with no learned "
            "or generative component."
        )

        st.markdown("**6. Thermal Isolation Threshold**")
        st.latex(
            r"M(x,y) = \begin{cases} 255 & I_{enhanced}(x,y) \geq \tau \\ 0 & \text{otherwise} \end{cases}"
        )
        st.caption("Tau is the operator-controlled slider threshold; M is the binary mask alpha-blended over the preview.")


def render_export_panel(results: dict, staged: dict) -> None:
    st.markdown(
        '<div class="irw-section-heading">📦 Multi-Palette Bundled Export Pipeline</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Compiles JET, VIRIDIS, and HOT geotagged GeoTIFF products into a single ZIP — "
        "CRS and affine transform preserved exactly on every product."
    )

    compile_clicked = st.button(
        "🗜️ Compile Multi-Palette GeoTIFF Package", type="primary", use_container_width=True
    )

    if compile_clicked:
        try:
            with geospatial_engine.TempWorkspace() as scratch_dir:
                archive_bytes = geospatial_engine.build_multi_palette_archive(
                    results["palette_products"],
                    staged["transform"],
                    staged["crs"],
                    staged["metadata"].filename,
                )
                geospatial_engine.sweep_temp_artifacts(scratch_dir)

            base_name = os.path.splitext(os.path.basename(staged["metadata"].filename))[0] or "ir_product"
            st.session_state.export_archive = archive_bytes
            st.session_state.export_filename = f"{base_name}_multipalette_export.zip"
        except Exception as exc:  # noqa: BLE001
            st.session_state.export_archive = None
            st.error(f"🚫 Export Compilation Failed — {exc}")

    if st.session_state.export_archive:
        st.download_button(
            label="⬇️ Download Multi-Palette Package (.zip)",
            data=st.session_state.export_archive,
            file_name=st.session_state.export_filename,
            mime="application/zip",
            use_container_width=True,
        )
        st.success(
            f"Package ready — {len(st.session_state.export_archive) / 1024:.1f} KB, "
            f"{len(results['palette_products'])} geotagged products (JET + VIRIDIS + HOT)."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    geospatial_engine.cleanup_stale_workspaces()
    init_session_state()
    inject_custom_css()
    render_cosmic_accents()
    render_header()
    handle_upload()

    staged = st.session_state.staged
    if staged is None:
        st.info("⬆️ Awaiting a raw GeoTIFF upload to begin staging.Proceed When")
        return

    render_staging_box(staged["metadata"])
    params, band_index = render_sidebar_controls(staged["metadata"].band_count)

    ignite_clicked = st.button(
        "🚀 Initialize Computational Ingestion Engine",
        type="primary",
        use_container_width=True,
    )
    if ignite_clicked:
        with st.spinner("Executing deterministic dual-stage enhancement pipeline..."):
            run_ignition(params, band_index)

    results = st.session_state.processed_results
    if results is None:
        return

    render_visualization_row(results)
    render_histogram_row(results)
    render_thermal_isolation_panel(results)
    render_metadata_audit_desk(staged["metadata"], results["band_stats"])
    render_technical_documentation()
    render_export_panel(results, staged)


if __name__ == "__main__":
    main()
