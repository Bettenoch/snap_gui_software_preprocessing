# %% [markdown]
# # 🛰️ Sentinel-1 SNAP Preprocessing Quality Assessment
# 
# **Comprehensive quality analysis of SNAP GPT-preprocessed Sentinel-1 GRD imagery**
# 
# This notebook performs:
# - ✅ **Metadata validation** (CRS, bands, resolution, nodata handling)
# - ✅ **Statistical quality metrics** (SNR, dynamic range, speckle assessment)
# - ✅ **Spatial quality checks** (edge artifacts, stripe detection)
# - ✅ **Radiometric calibration validation** (expected backscatter ranges)
# - ✅ **Side-by-side comparison with GEE preprocessing**
# - ✅ **Automated quality scoring and reporting**
# 
# ---
# 
# ## 📋 Prerequisites
# 
# ```bash
# pip install rasterio numpy matplotlib seaborn pandas scipy earthengine-api geemap shapely
# ```
# 
# **Setup Earth Engine:**
# ```bash
# earthengine authenticate
# ```

# %% [markdown]
# ## 1️⃣ Configuration & Setup

# %%
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds, calculate_default_transform
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats, ndimage
from shapely.geometry import shape, mapping
import json

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %% [markdown]
# ### 📂 Locate SNAP-preprocessed GeoTIFF

# %%
def find_geotiff(search_dir: str = '.') -> Path:
    """Find the first GeoTIFF in the specified directory."""
    search_path = Path(search_dir)
    tif_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    
    for pattern in tif_patterns:
        tif_files = list(search_path.glob(pattern))
        if tif_files:
            return tif_files[0]
    
    raise FileNotFoundError(
        f"No GeoTIFF found in '{search_dir}'. "
        "Please ensure your SNAP-preprocessed Sentinel-1 image is in this directory."
    )

# Locate the SNAP-preprocessed image
SNAP_GEOTIFF = find_geotiff()
print(f"📁 Found SNAP GeoTIFF: {SNAP_GEOTIFF}")

# %% [markdown]
# ## 2️⃣ Metadata Inspection & Validation

# %%
def inspect_metadata(tif_path: Path) -> Dict:
    """Extract and validate GeoTIFF metadata."""
    with rasterio.open(tif_path) as src:
        metadata = {
            'path': str(tif_path),
            'crs': str(src.crs),
            'epsg': src.crs.to_epsg() if src.crs else None,
            'bands': src.count,
            'width': src.width,
            'height': src.height,
            'dtypes': src.dtypes,
            'nodata': src.nodatavals,
            'bounds': src.bounds,
            'transform': src.transform,
            'resolution': (src.transform[0], abs(src.transform[4])),
            'units': src.units,
            'descriptions': src.descriptions
        }
        
        # Check if projected
        metadata['is_projected'] = src.crs is not None and not src.crs.is_geographic
        
        # Estimate size in MB
        metadata['size_mb'] = (src.width * src.height * src.count * 
                               np.dtype(src.dtypes[0]).itemsize) / (1024 * 1024)
    
    return metadata

# Inspect SNAP image
snap_meta = inspect_metadata(SNAP_GEOTIFF)

print("\n" + "="*60)
print("📊 SNAP PREPROCESSING METADATA")
print("="*60)
for key, value in snap_meta.items():
    if key not in ['transform', 'bounds']:
        print(f"{key:20s}: {value}")
print("="*60)

# %% [markdown]
# ### 🎯 Define Region of Interest (ROI)
# 
# You can either:
# - **Option A:** Use a GeoJSON geometry file
# - **Option B:** Clip a centered patch of specified size
# - **Option C:** Use the full image

# %%
# Configuration
USE_GEOJSON = False  # Set to True if you have a GeoJSON ROI
GEOJSON_PATH = "roi.geojson"  # Path to your GeoJSON file
CLIP_SIZE = 512  # Patch size if not using GeoJSON (pixels)
USE_FULL_IMAGE = False  # Set to True to analyze entire image

def load_roi_geometry(geojson_path: str):
    """Load ROI from GeoJSON file."""
    with open(geojson_path, 'r') as f:
        geojson = json.load(f)
    
    if geojson['type'] == 'FeatureCollection':
        return shape(geojson['features'][0]['geometry'])
    else:
        return shape(geojson)

def get_clip_window(src, clip_size: int) -> Tuple[Window, any]:
    """Calculate centered clip window."""
    window_size = min(clip_size, src.width, src.height)
    center_col = src.width // 2
    center_row = src.height // 2
    col_off = int(center_col - window_size // 2)
    row_off = int(center_row - window_size // 2)
    
    win = Window(col_off, row_off, window_size, window_size)
    transform = src.window_transform(win)
    
    return win, transform

# Load and clip data
with rasterio.open(SNAP_GEOTIFF) as src:
    if USE_GEOJSON and os.path.exists(GEOJSON_PATH):
        print(f"📍 Using ROI from: {GEOJSON_PATH}")
        roi_geom = load_roi_geometry(GEOJSON_PATH)
        clipped_data, clipped_transform = mask(src, [roi_geom], crop=True, filled=False)
        clip_bounds = roi_geom.bounds
        clip_method = "geojson"
        
    elif USE_FULL_IMAGE:
        print("🗺️ Using full image")
        clipped_data = src.read(masked=True)
        clipped_transform = src.transform
        clip_bounds = src.bounds
        clip_method = "full"
        
    else:
        print(f"✂️ Clipping centered {CLIP_SIZE}x{CLIP_SIZE} patch")
        win, clipped_transform = get_clip_window(src, CLIP_SIZE)
        clipped_data = src.read(window=win, masked=True)
        clip_bounds = src.window_bounds(win)
        clip_method = "centered"
    
    # Convert to float32 for processing
    clipped_data = clipped_data.astype('float32')
    profile = src.profile.copy()

# Update profile for clipped data
profile.update({
    'height': clipped_data.shape[1],
    'width': clipped_data.shape[2],
    'transform': clipped_transform
})

print(f"\n✅ Clipped data shape: {clipped_data.shape}")
print(f"   Method: {clip_method}")
print(f"   Bounds: {clip_bounds}")

# %% [markdown]
# ## 3️⃣ Statistical Quality Assessment

# %%
def compute_band_statistics(data: np.ma.MaskedArray, band_idx: int) -> Dict:
    """Compute comprehensive statistics for a single band."""
    band = data[band_idx]
    mask = np.ma.getmaskarray(band)
    valid_count = np.count_nonzero(~mask)
    valid_pct = 100 * valid_count / band.size
    
    if valid_count == 0:
        return {
            'band': band_idx + 1,
            'valid_pixels': 0,
            'valid_pct': 0,
            'status': '❌ NO VALID DATA'
        }
    
    vals = band.compressed()
    
    # Basic statistics
    stats_dict = {
        'band': band_idx + 1,
        'valid_pixels': valid_count,
        'valid_pct': round(valid_pct, 2),
        'min': round(float(vals.min()), 4),
        'max': round(float(vals.max()), 4),
        'mean': round(float(vals.mean()), 4),
        'median': round(float(np.median(vals)), 4),
        'std': round(float(vals.std()), 4),
        'cv': round(float(vals.std() / vals.mean() if vals.mean() != 0 else 0), 4),
    }
    
    # Data quality indicators
    stats_dict['zeros_count'] = int(np.sum(vals == 0))
    stats_dict['zeros_pct'] = round(100 * stats_dict['zeros_count'] / vals.size, 2)
    stats_dict['negatives_pct'] = round(100 * np.sum(vals < 0) / vals.size, 2)
    stats_dict['dynamic_range_db'] = round(10 * np.log10(vals.max() / vals.min()) if vals.min() > 0 else np.nan, 2)
    
    # Convert to dB if in linear scale
    if vals.max() < 100:  # Likely linear power
        db_vals = 10 * np.log10(vals + 1e-10)
        stats_dict['mean_db'] = round(float(db_vals.mean()), 2)
        stats_dict['median_db'] = round(float(np.median(db_vals)), 2)
        stats_dict['expected_range'] = 'VV: -25 to 5 dB, VH: -30 to -5 dB'
        
        # Check if values are in expected range
        if -30 <= stats_dict['mean_db'] <= 5:
            stats_dict['range_check'] = '✅ PASS'
        else:
            stats_dict['range_check'] = '⚠️ OUT OF EXPECTED RANGE'
    
    # Skewness and Kurtosis
    stats_dict['skewness'] = round(float(stats.skew(vals)), 4)
    stats_dict['kurtosis'] = round(float(stats.kurtosis(vals)), 4)
    
    return stats_dict

# Compute statistics for all bands
print("\n" + "="*80)
print("📈 BAND STATISTICS & QUALITY METRICS")
print("="*80)

band_stats = []
for b in range(clipped_data.shape[0]):
    stats_dict = compute_band_statistics(clipped_data, b)
    band_stats.append(stats_dict)

df_stats = pd.DataFrame(band_stats)
print(df_stats.to_string(index=False))
print("="*80)

# %% [markdown]
# ### 📊 Quality Score Calculation

# %%
def calculate_quality_score(stats: Dict) -> Dict:
    """Calculate quality score based on multiple criteria."""
    score = 100
    issues = []
    
    # Valid data percentage
    if stats['valid_pct'] < 95:
        deduction = (95 - stats['valid_pct']) * 0.5
        score -= deduction
        issues.append(f"Low valid data: {stats['valid_pct']:.1f}%")
    
    # Zero values
    if stats['zeros_pct'] > 5:
        score -= min(20, stats['zeros_pct'])
        issues.append(f"High zeros: {stats['zeros_pct']:.1f}%")
    
    # Negative values (should be minimal)
    if stats['negatives_pct'] > 0.1:
        score -= min(15, stats['negatives_pct'] * 10)
        issues.append(f"Negative values: {stats['negatives_pct']:.1f}%")
    
    # Dynamic range check
    if 'dynamic_range_db' in stats and not np.isnan(stats['dynamic_range_db']):
        if stats['dynamic_range_db'] < 20:
            score -= 10
            issues.append(f"Low dynamic range: {stats['dynamic_range_db']:.1f} dB")
    
    # Expected range check
    if stats.get('range_check') == '⚠️ OUT OF EXPECTED RANGE':
        score -= 15
        issues.append(f"Mean {stats.get('mean_db', 'N/A')} dB outside expected range")
    
    score = max(0, score)
    
    if score >= 90:
        grade = "A - Excellent"
    elif score >= 80:
        grade = "B - Good"
    elif score >= 70:
        grade = "C - Acceptable"
    elif score >= 60:
        grade = "D - Poor"
    else:
        grade = "F - Fail"
    
    return {
        'score': round(score, 1),
        'grade': grade,
        'issues': issues
    }

# Calculate quality scores
print("\n" + "="*80)
print("🎯 QUALITY SCORING")
print("="*80)

for idx, stats in enumerate(band_stats):
    quality = calculate_quality_score(stats)
    print(f"\nBand {stats['band']}:")
    print(f"  Score: {quality['score']}/100 ({quality['grade']})")
    if quality['issues']:
        print(f"  Issues:")
        for issue in quality['issues']:
            print(f"    - {issue}")
    else:
        print(f"  ✅ No issues detected")

print("="*80)

# %% [markdown]
# ## 4️⃣ Visualization: Bands & Histograms

# %%
def plot_band_analysis(data: np.ma.MaskedArray, band_idx: int, stats: Dict):
    """Create comprehensive visualization for a single band."""
    band = data[band_idx].filled(np.nan)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Band {band_idx + 1} Analysis', fontsize=16, fontweight='bold')
    
    # 1. Linear scale image
    ax = axes[0, 0]
    im1 = ax.imshow(band, cmap='gray', vmin=np.nanpercentile(band, 2), 
                    vmax=np.nanpercentile(band, 98))
    ax.set_title('Linear Scale (2-98 percentile stretch)')
    ax.axis('off')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    
    # 2. dB scale image (if applicable)
    ax = axes[0, 1]
    if stats['max'] < 100:  # Linear power values
        band_db = 10 * np.log10(band + 1e-10)
        im2 = ax.imshow(band_db, cmap='gray', vmin=-25, vmax=0)
        ax.set_title('dB Scale (-25 to 0 dB)')
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='dB')
    else:
        im2 = ax.imshow(band, cmap='viridis')
        ax.set_title('Alternative Colormap')
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    
    # 3. Histogram (linear)
    ax = axes[1, 0]
    valid_data = band[~np.isnan(band)].ravel()
    ax.hist(valid_data, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.4f}")
    ax.axvline(stats['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.4f}")
    ax.set_xlabel('Backscatter (linear)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram - Linear Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Histogram (dB if applicable)
    ax = axes[1, 1]
    if stats['max'] < 100:
        db_data = 10 * np.log10(valid_data + 1e-10)
        ax.hist(db_data, bins=100, color='forestgreen', alpha=0.7, edgecolor='black')
        ax.axvline(stats.get('mean_db', 0), color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {stats.get('mean_db', 0):.2f} dB")
        ax.set_xlabel('Backscatter (dB)')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram - dB Scale')
        ax.legend()
    else:
        # Box plot for statistics
        ax.boxplot(valid_data, vert=True)
        ax.set_ylabel('Value')
        ax.set_title('Box Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Plot all bands
for b in range(clipped_data.shape[0]):
    plot_band_analysis(clipped_data, b, band_stats[b])

# %% [markdown]
# ## 5️⃣ Spatial Quality Assessment

# %%
def assess_spatial_quality(data: np.ma.MaskedArray, band_idx: int) -> Dict:
    """Detect spatial artifacts like stripes, edge effects, and noise patterns."""
    band = data[band_idx].filled(np.nan)
    
    results = {}
    
    # Edge artifact detection (check border pixels)
    border_size = 10
    top_edge = band[:border_size, :]
    bottom_edge = band[-border_size:, :]
    left_edge = band[:, :border_size]
    right_edge = band[:, -border_size:]
    
    edges = [top_edge, bottom_edge, left_edge, right_edge]
    edge_means = [np.nanmean(e) for e in edges]
    center_mean = np.nanmean(band[border_size:-border_size, border_size:-border_size])
    
    results['edge_anomaly'] = any(abs(em - center_mean) / center_mean > 0.3 for em in edge_means if not np.isnan(em))
    
    # Stripe detection (column-wise variation)
    col_means = np.nanmean(band, axis=0)
    col_variation = np.nanstd(col_means) / np.nanmean(col_means) if np.nanmean(col_means) != 0 else 0
    results['stripe_indicator'] = col_variation
    results['potential_stripes'] = col_variation > 0.15
    
    # Texture smoothness (Laplacian variance)
    laplacian = ndimage.laplace(np.nan_to_num(band))
    results['laplacian_variance'] = float(np.var(laplacian))
    results['texture_quality'] = 'smooth' if results['laplacian_variance'] < 1000 else 'noisy'
    
    return results

# Assess spatial quality
print("\n" + "="*80)
print("🗺️ SPATIAL QUALITY ASSESSMENT")
print("="*80)

for b in range(clipped_data.shape[0]):
    spatial_qa = assess_spatial_quality(clipped_data, b)
    print(f"\nBand {b + 1}:")
    print(f"  Edge artifacts: {'⚠️ DETECTED' if spatial_qa['edge_anomaly'] else '✅ None'}")
    print(f"  Stripe detection: {'⚠️ POTENTIAL' if spatial_qa['potential_stripes'] else '✅ None'} "
          f"(CV: {spatial_qa['stripe_indicator']:.4f})")
    print(f"  Texture quality: {spatial_qa['texture_quality'].upper()} "
          f"(Laplacian var: {spatial_qa['laplacian_variance']:.2f})")

print("="*80)

# %% [markdown]
# ## 6️⃣ Google Earth Engine Comparison
# 
# This section fetches a Sentinel-1 GRD image from GEE matching your SNAP image's:
# - Geographic extent
# - Acquisition date (if available in metadata)
# - Polarization bands

# %%
# Check if Earth Engine is needed
COMPARE_WITH_GEE = True  # Set to False to skip GEE comparison

if COMPARE_WITH_GEE:
    try:
        import ee
        import geemap
        
        # Initialize Earth Engine
        try:
            ee.Initialize()
            print("✅ Earth Engine initialized successfully")
        except Exception as e:
            print(f"⚠️ Earth Engine initialization failed: {e}")
            print("Run: earthengine authenticate")
            COMPARE_WITH_GEE = False
    except ImportError:
        print("⚠️ Earth Engine not installed. Install with: pip install earthengine-api geemap")
        COMPARE_WITH_GEE = False

# %% [markdown]
# ### 🌍 Fetch Sentinel-1 from GEE

# %%
if COMPARE_WITH_GEE:
    # Get bounds in WGS84
    with rasterio.open(SNAP_GEOTIFF) as src:
        if USE_GEOJSON and os.path.exists(GEOJSON_PATH):
            roi_geom = load_roi_geometry(GEOJSON_PATH)
            bounds_4326 = roi_geom.bounds
        else:
            bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *clip_bounds)
    
    # Create GEE geometry
    geom = ee.Geometry.Rectangle(list(bounds_4326))
    
    # Date range (adjust as needed - using 2023 as default)
    START_DATE = '2023-01-01'
    END_DATE = '2023-12-31'
    
    print(f"\n🌍 Fetching Sentinel-1 from GEE...")
    print(f"   Date range: {START_DATE} to {END_DATE}")
    print(f"   Bounds: {bounds_4326}")
    
    # Build S1 collection
    s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                     .filterBounds(geom)
                     .filterDate(START_DATE, END_DATE)
                     .filter(ee.Filter.eq('instrumentMode', 'IW'))
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    
    # Get collection size
    collection_size = s1_collection.size().getInfo()
    print(f"   Found {collection_size} Sentinel-1 images")
    
    if collection_size == 0:
        print("⚠️ No Sentinel-1 images found for specified parameters")
        COMPARE_WITH_GEE = False
    else:
        # Use median composite (or first image)
        USE_MEDIAN = True  # Set to False to use first image only
        
        if USE_MEDIAN:
            s1_image = s1_collection.select(['VV', 'VH']).median()
            print("   Using: Median composite")
        else:
            s1_image = s1_collection.select(['VV', 'VH']).first()
            print("   Using: First image")
        
        # Download to local file
        GEE_OUTPUT = "gee_s1_comparison.tif"
        
        print(f"\n📥 Downloading GEE image to: {GEE_OUTPUT}")
        print("   This may take a few minutes...")
        
        try:
            # Use geemap to download
            geemap.ee_export_image(
                s1_image,
                filename=GEE_OUTPUT,
                scale=10,
                region=geom,
                file_per_band=False
            )
            print(f"✅ Downloaded: {GEE_OUTPUT}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\nAlternative: Export to Google Drive")
            print("Run this in a separate cell:")
            print(f"""
task = ee.batch.Export.image.toDrive(
    image=s1_image,
    description='gee_s1_comparison',
    region=geom.getInfo()['coordinates'],
    scale=10,
    fileFormat='GeoTIFF'
)
task.start()
print("Task started - check Google Drive")
            """)
            COMPARE_WITH_GEE = False

# %% [markdown]
# ### 📊 Compare SNAP vs GEE

# %%
if COMPARE_WITH_GEE and os.path.exists("gee_s1_comparison.tif"):
    print("\n" + "="*80)
    print("📊 SNAP vs GEE COMPARISON")
    print("="*80)
    
    # Load GEE data
    with rasterio.open("gee_s1_comparison.tif") as gee_src:
        gee_data = gee_src.read(masked=True).astype('float32')
        gee_meta = inspect_metadata(Path("gee_s1_comparison.tif"))
    
    print("\nGEE Image Info:")
    print(f"  Bands: {gee_meta['bands']}")
    print(f"  Shape: {gee_data.shape}")
    print(f"  CRS: {gee_meta['crs']}")
    
    # Compare band by band
    num_bands = min(clipped_data.shape[0], gee_data.shape[0])
    
    for b in range(num_bands):
        snap_band = clipped_data[b].filled(np.nan)
        gee_band = gee_data[b].filled(np.nan)
        
        # Ensure same shape (resample if needed)
        if snap_band.shape != gee_band.shape:
            print(f"\n⚠️ Band {b+1}: Shape mismatch - SNAP {snap_band.shape} vs GEE {gee_band.shape}")
            from scipy.ndimage import zoom
            zoom_factors = (snap_band.shape[0] / gee_band.shape[0], 
                           snap_band.shape[1] / gee_band.shape[1])
            gee_band = zoom(gee_band, zoom_factors, order=1)
            print(f"   Resampled GEE to match SNAP shape")
        
        # Calculate difference
        diff = snap_band - gee_band
        
        # Statistics
        print(f"\nBand {b+1} Comparison:")
        print(f"  SNAP  - Mean: {np.nanmean(snap_band):.4f}, Std: {np.nanstd(snap_band):.4f}")
        print(f"  GEE   - Mean: {np.nanmean(gee_band):.4f}, Std: {np.nanstd(gee_band):.4f}")
        print(f"  Diff  - Mean: {np.nanmean(diff):.4f}, Std: {np.nanstd(diff):.4f}")
        print(f"  RMSE: {np.sqrt(np.nanmean(diff**2)):.4f}")
        print(f"  Correlation: {np.corrcoef(snap_band[~np.isnan(snap_band)].ravel(), gee_band[~np.isnan(gee_band)].ravel())[0,1]:.4f}")
        
        # Visualize comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Band {b+1}: SNAP vs GEE Comparison', fontsize=16, fontweight='bold')
        
        # SNAP
        vmin, vmax = np.nanpercentile(snap_band, [2, 98])
        im1 = axes[0, 0].imshow(snap_band, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('SNAP Preprocessed')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # GEE
        im2 = axes[0, 1].imshow(gee_band, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('GEE Preprocessed')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        # Difference
        diff_lim = np.nanpercentile(np.abs(diff), 95)
        im3 = axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-diff_lim, vmax=diff_lim)
        axes[0, 2].set_title('Difference (SNAP - GEE)')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
        
        # Histograms
        axes[1, 0].hist(snap_band[~np.isnan(snap_band)].ravel(), bins=50, 
                        alpha=0.7, label='SNAP', color='blue')
        axes[1, 0].hist(gee_band[~np.isnan(gee_band)].ravel(), bins=50, 
                        alpha=0.7, label='GEE', color='red')
        axes[1, 0].set_xlabel('Backscatter (linear)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Value Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        snap_flat = snap_band[~np.isnan(snap_band) & ~np.isnan(gee_band)].ravel()
        gee_flat = gee_band[~np.isnan(snap_band) & ~np.isnan(gee_band)].ravel()
        axes[1, 1].scatter(gee_flat, snap_flat, alpha=0.1, s=1)
        axes[1, 1].plot([gee_flat.min(), gee_flat.max()], 
                        [gee_flat.min(), gee_flat.max()], 
                        'r--', linewidth=2, label='1:1 line')
        axes[1, 1].set_xlabel('GEE Backscatter')
        axes[1, 1].set_ylabel('SNAP Backscatter')
        axes[1, 1].set_title('Correlation Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Difference histogram
        axes[1, 2].hist(diff[~np.isnan(diff)].ravel(), bins=50, 
                        color='purple', alpha=0.7)
        axes[1, 2].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 2].set_xlabel('Difference (SNAP - GEE)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Difference Distribution')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print("="*80)

elif COMPARE_WITH_GEE:
    print("\n⚠️ GEE comparison image not found. Please download it first.")

# %% [markdown]
# ## 7️⃣ Summary Report

# %%
def generate_quality_report(band_stats: list, output_file: str = "quality_report.txt"):
    """Generate comprehensive quality assessment report."""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("SENTINEL-1 SNAP PREPROCESSING QUALITY ASSESSMENT REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nImage: {SNAP_GEOTIFF}")
    report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Clipping Method: {clip_method}")
    report_lines.append(f"\nImage Metadata:")
    report_lines.append(f"  CRS: {snap_meta['crs']}")
    report_lines.append(f"  Bands: {snap_meta['bands']}")
    report_lines.append(f"  Resolution: {snap_meta['resolution'][0]}m x {snap_meta['resolution'][1]}m")
    report_lines.append(f"  Dimensions: {snap_meta['width']} x {snap_meta['height']} pixels")
    report_lines.append(f"  Size: {snap_meta['size_mb']:.2f} MB")
    
    report_lines.append(f"\n{'='*80}")
    report_lines.append("QUALITY ASSESSMENT SUMMARY")
    report_lines.append("="*80)
    
    overall_scores = []
    
    for idx, stats in enumerate(band_stats):
        quality = calculate_quality_score(stats)
        overall_scores.append(quality['score'])
        
        report_lines.append(f"\n{'─'*80}")
        report_lines.append(f"Band {stats['band']} Quality Assessment")
        report_lines.append(f"{'─'*80}")
        report_lines.append(f"Quality Score: {quality['score']}/100 ({quality['grade']})")
        report_lines.append(f"\nKey Metrics:")
        report_lines.append(f"  Valid Pixels: {stats['valid_pct']:.2f}%")
        report_lines.append(f"  Mean (linear): {stats['mean']:.4f}")
        if 'mean_db' in stats:
            report_lines.append(f"  Mean (dB): {stats['mean_db']:.2f} dB")
        report_lines.append(f"  Std Dev: {stats['std']:.4f}")
        report_lines.append(f"  Coefficient of Variation: {stats['cv']:.4f}")
        report_lines.append(f"  Dynamic Range: {stats.get('dynamic_range_db', 'N/A')} dB")
        report_lines.append(f"  Zeros: {stats['zeros_pct']:.2f}%")
        report_lines.append(f"  Negatives: {stats['negatives_pct']:.2f}%")
        
        if quality['issues']:
            report_lines.append(f"\nIdentified Issues:")
            for issue in quality['issues']:
                report_lines.append(f"  ⚠️ {issue}")
        else:
            report_lines.append(f"\n✅ No quality issues detected")
    
    # Overall assessment
    avg_score = np.mean(overall_scores)
    report_lines.append(f"\n{'='*80}")
    report_lines.append("OVERALL ASSESSMENT")
    report_lines.append("="*80)
    report_lines.append(f"Average Quality Score: {avg_score:.1f}/100")
    
    if avg_score >= 90:
        assessment = "EXCELLENT - Image is production-ready"
        emoji = "🟢"
    elif avg_score >= 80:
        assessment = "GOOD - Image is suitable for most applications"
        emoji = "🟢"
    elif avg_score >= 70:
        assessment = "ACCEPTABLE - Image may have minor issues"
        emoji = "🟡"
    elif avg_score >= 60:
        assessment = "POOR - Image has significant quality issues"
        emoji = "🟠"
    else:
        assessment = "FAIL - Image quality is insufficient"
        emoji = "🔴"
    
    report_lines.append(f"\n{emoji} {assessment}")
    
    # Recommendations
    report_lines.append(f"\n{'='*80}")
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("="*80)
    
    recommendations = []
    
    for stats in band_stats:
        if stats['zeros_pct'] > 5:
            recommendations.append("• High percentage of zero values detected - verify thermal noise removal settings")
        if stats.get('negatives_pct', 0) > 0.1:
            recommendations.append("• Negative values present - check calibration parameters")
        if stats.get('valid_pct', 100) < 95:
            recommendations.append("• Low valid pixel percentage - review masking and nodata handling")
        if stats.get('range_check') == '⚠️ OUT OF EXPECTED RANGE':
            recommendations.append("• Backscatter values outside expected range - verify calibration type (sigma0/gamma0)")
    
    if not recommendations:
        recommendations.append("✅ No critical issues detected - preprocessing appears successful")
    
    for rec in set(recommendations):  # Remove duplicates
        report_lines.append(rec)
    
    report_lines.append(f"\n{'='*80}")
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # Write to file
    report_text = "\n".join(report_lines)
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n💾 Report saved to: {output_file}")
    
    return report_text

# Generate report
quality_report = generate_quality_report(band_stats)

# %% [markdown]
# ## 8️⃣ Export Clipped Data (Optional)

# %%
EXPORT_CLIP = True  # Set to False to skip export

if EXPORT_CLIP:
    output_path = f"snap_quality_clip_{clip_method}.tif"
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(clipped_data.filled(profile.get('nodata', 0)))
    
    print(f"\n💾 Clipped data exported to: {output_path}")
    print(f"   Shape: {clipped_data.shape}")
    print(f"   Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

# %% [markdown]
# ## 9️⃣ Advanced Analysis (Optional)

# %%
# Speckle noise assessment using Coefficient of Variation
def assess_speckle_noise(data: np.ma.MaskedArray, band_idx: int, window_size: int = 7):
    """Assess speckle noise using local statistics."""
    band = data[band_idx].filled(np.nan)
    
    # Calculate local mean and std using uniform filter
    from scipy.ndimage import uniform_filter
    
    local_mean = uniform_filter(band, size=window_size, mode='constant', cval=np.nan)
    local_sq_mean = uniform_filter(band**2, size=window_size, mode='constant', cval=np.nan)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    
    # Coefficient of variation
    with np.errstate(divide='ignore', invalid='ignore'):
        local_cv = local_std / local_mean
    
    # Speckle index (lower is better)
    speckle_index = np.nanmean(local_cv)
    
    results = {
        'speckle_index': float(speckle_index),
        'quality': 'excellent' if speckle_index < 0.2 else 'good' if speckle_index < 0.4 else 'poor'
    }
    
    return results, local_cv

print("\n" + "="*80)
print("🔬 ADVANCED SPECKLE ASSESSMENT")
print("="*80)

for b in range(clipped_data.shape[0]):
    speckle_results, cv_map = assess_speckle_noise(clipped_data, b)
    print(f"\nBand {b+1}:")
    print(f"  Speckle Index: {speckle_results['speckle_index']:.4f}")
    print(f"  Quality: {speckle_results['quality'].upper()}")
    
    # Visualize speckle map
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Band {b+1}: Speckle Assessment', fontsize=14, fontweight='bold')
    
    # Original band
    band_data = clipped_data[b].filled(np.nan)
    im1 = axes[0].imshow(band_data, cmap='gray', 
                         vmin=np.nanpercentile(band_data, 2),
                         vmax=np.nanpercentile(band_data, 98))
    axes[0].set_title('Original Band')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Coefficient of Variation (speckle indicator)
    im2 = axes[1].imshow(cv_map, cmap='hot', vmin=0, vmax=0.5)
    axes[1].set_title('Coefficient of Variation (Speckle Indicator)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, label='CV')
    
    plt.tight_layout()
    plt.show()

print("="*80)

# %% [markdown]
# ## 🎉 Analysis Complete!
# 
# ### Summary of Outputs:
# 1. **Quality Report**: `quality_report.txt` - Comprehensive assessment summary
# 2. **Clipped Image**: `snap_quality_clip_*.tif` - Extracted ROI for analysis
# 3. **Visualizations**: Band statistics, histograms, spatial quality, and comparisons
# 
# ### Next Steps:
# - Review quality scores and recommendations
# - If issues are found, adjust SNAP GPT preprocessing parameters
# - Compare with GEE preprocessing to validate methodology
# - Use the clipped patch for further analysis or model training
# 
# ### Preprocessing Quality Checklist:
# - ✅ Valid data coverage > 95%
# - ✅ Backscatter values in expected range (-30 to 5 dB)
# - ✅ Low percentage of zeros (< 5%)
# - ✅ No significant edge artifacts
# - ✅ Adequate dynamic range (> 20 dB)
# - ✅ Effective speckle filtering (CV < 0.4)
# - ✅ Consistent results compared to GEE

# %%