# dF/F Analysis Pipeline for Voltage Indicator Data

## Overview
This is a complete analysis pipeline for voltage indicator imaging data that implements robust dF/F (delta F over F) analysis with organized output management, temporal smoothing, and comprehensive video comparison tools.

## Pipeline Components

### Core Analysis (`analyze_organized.py`)
- **Batch Processing**: Processes all AVI files in the DATA folder automatically
- **Memory Efficient**: Spatial downsampling (4x) and frame limiting (1000 frames) for manageable processing
- **Robust Baseline**: Uses 5th percentile method for F0 calculation (Chen et al. 2013)
- **Temporal Smoothing**: 5-frame moving average to reduce noise while preserving signal dynamics
- **Organized Outputs**: Structured directory system with timestamps and metadata

### Video Comparison (`video_converter.py`)
- **Raw Video Conversion**: AVI to MP4 conversion for web compatibility
- **2-Way Comparisons**: Side-by-side Raw | dF/F videos
- **3-Way Comparisons**: Triple view Raw | dF/F | Smoothed videos
- **Frame Synchronization**: Matches temporal coverage between raw and processed videos

### Output Management (`output_manager.py`)
- **Directory Structure**: Organized PROCESSED folder with videos, data, logs, comparisons
- **Metadata Tracking**: Complete parameter logging and analysis summaries
- **File Organization**: Timestamp-based naming for version control

## Key Features

### 1. Production-Ready Pipeline
- **Consistent Parameters**: 4x spatial downsampling, 1000 frames, 5th percentile baseline
- **Quality Control**: Both regular and smoothed dF/F outputs for noise assessment
- **Comprehensive Logging**: Full analysis parameters and results tracking
- **Batch Processing**: Handles multiple videos with identical settings

### 2. Baseline (F0) Calculation - Percentile Method
- **5th Percentile**: More robust than mean of initial frames
- **Photobleaching Resistant**: Accounts for gradual fluorescence decline  
- **Temporal Stability**: Represents true quiescent state across entire recording
- **Validated Approach**: Established methodology from Chen et al. (2013)

## Scientific Background


### 3. Temporal Smoothing
- **5-Frame Moving Average**: Reduces noise while preserving signal dynamics
- **Dual Output**: Both original and smoothed dF/F videos for comparison
- **Quality Assessment**: Smoothed videos help evaluate signal vs noise
- **Artifact Reduction**: Removes high-frequency imaging artifacts

## Current Workflow

### Step 1: Data Preparation
```
Place AVI video files in the DATA/ folder
```

### Step 2: Batch Analysis
```bash
python analyze_organized.py
```
This processes ALL videos in DATA/ with consistent parameters:
- 4x spatial downsampling (reduces memory usage)
- 1000 frames maximum per video (extended temporal coverage)
- 5th percentile baseline calculation
- Temporal smoothing with 5-frame window

### Step 3: Video Comparisons
```bash
python video_converter.py
```
Creates comparison videos for visual quality assessment:
- 2-way: Raw | dF/F side-by-side
- 3-way: Raw | dF/F | Smoothed triple view

### Output Structure
```
PROCESSED/
├── videos/          # dF/F analysis results (.mp4)
├── data/           # Summary TIFF files and metadata
├── logs/           # Analysis logs and parameters
├── comparisons/    # Side-by-side comparison videos
└── raw_mp4/       # Converted raw videos
```


## Scientific Background

### dF/F Calculation
```
dF/F = (F(t) - F0) / F0
```
Where F(t) is fluorescence at time t and F0 is baseline fluorescence.

### Why 5th Percentile Baseline?
- **Photobleaching Resistant**: Accounts for gradual fluorescence decline
- **Outlier Robust**: Ignores bright artifacts and spikes  
- **Temporally Stable**: Represents true quiescent state across recording
- **Validated Method**: Established in Chen et al. (2013) Nature paper

## Key Parameters

### Current Settings (Production)
- **Spatial Downsampling**: 4x (reduces 608×608 to 152×152)
- **Frame Limit**: 1000 frames (16-28 seconds depending on frame rate)
- **Baseline Method**: 5th percentile across all frames
- **Temporal Smoothing**: 5-frame moving average window
- **Video Format**: MP4 output for compatibility

### Adjustable Parameters
To modify processing parameters, edit `analyze_organized.py`:
```python
analysis_params = {
    'spatial_downsample': 4,     # Change to 2 for higher resolution
    'max_frames': 1000,          # Increase for longer videos
    'baseline_percentile': 5     # Adjust baseline calculation
}
```

## Performance & Results

### Typical Processing Times
- **Small videos (608×608)**: ~3 seconds per video
- **Large videos (1080×1440)**: ~10 seconds per video  
- **Memory usage**: 88-371 MB per video during processing

### Output File Sizes (1000 frames)
- **dF/F videos**: 1.2-23.3 MB depending on content and resolution
- **Smoothed videos**: Similar to dF/F, often smaller due to noise reduction
- **TIFF summaries**: 4.4-18.6 MB for quantitative analysis
- **Comparison videos**: 1.7-82 MB for visual assessment

## Signal Quality Examples
From recent analysis of 11 videos:
- **Strong signals**: dF/F ranges up to 16.143 (Rat-G1-13_30_22_2)
- **Typical ranges**: -1.000 to 5-8 for most videos
- **Low noise recordings**: Ranges around ±0.3-0.8 for some preparations

## Primary Reference
**Chen, T. W., Wardill, T. J., Sun, Y., Pulver, S. R., Renninger, S. L., Baohan, A., ... & Kim, D. S. (2013).**  
*Ultrasensitive fluorescent proteins for imaging neuronal activity.*  
**Nature, 499(7458), 295-300.**

This paper established percentile-based F0 calculation as the standard method for calcium and voltage imaging.

## Usage Examples

### Basic Analysis
```python
# Traditional method
result1 = analyze_video("video.avi", 
                       baseline_method='frames', 
                       baseline_frames=50)

# Percentile method (recommended)
result2 = analyze_video("video.avi", 
                       baseline_method='percentile', 
                       baseline_percentile=5)
```

### Memory-Efficient Processing
```python
# For large videos
result = analyze_video("large_video.avi",
                      window_size=500,    # Process 500 frames at a time
                      overlap=50,         # 50 frame overlap
                      baseline_method='percentile')
```

## Parameters

### Window Size Selection
- **Small windows (100-300 frames)**: Lower memory usage, more processing overhead
- **Large windows (500-1000 frames)**: Higher memory usage, fewer processing steps
- **Rule of thumb**: Use largest window that fits comfortably in available RAM

### Overlap Selection
- **Small overlap (10-25 frames)**: Faster processing, potential edge artifacts
- **Large overlap (50-100 frames)**: Smoother transitions, more computation
- **Rule of thumb**: Use 10-20% of window size for overlap

### Percentile Selection
- **5th percentile**: Conservative, good for low SNR data
- **10th percentile**: Moderate, general purpose
- **20th percentile**: Liberal, for high baseline variability

## Applications

### Ideal Use Cases
1. **Voltage Indicator Imaging**: Fast voltage changes, photobleaching concerns
2. **Calcium Imaging**: Activity-dependent fluorescence changes
3. **Long Recordings**: Extended imaging sessions with drift
4. **Large Field of View**: When objective > camera sensor size
5. **Moving Preparations**: Slight movement during recording

### Microscopy Setups
- **Wide-field fluorescence microscopy**
- **Two-photon microscopy**
- **Light-sheet microscopy**
- **Confocal microscopy** (for time series)

## Output Files
- **TIFF Stack**: Complete dF/F time series for further analysis
- **Analysis Plots**: Cluster maps, time traces, and summary statistics
- **Summary Data**: Cluster assignments and statistics

## Performance Considerations
- **Memory**: Windowed processing scales with window size, not video length
- **Speed**: Processing time scales linearly with total frames
- **Storage**: Output TIFF files can be large for long recordings
- **Clustering**: Spatial/temporal subsampling for very large datasets

## Future Enhancements
1. **Motion Correction**: Integration with registration algorithms
2. **ROI Analysis**: Automated region-of-interest detection
3. **Spike Detection**: Event detection in voltage traces
4. **Multi-channel**: Support for multi-wavelength imaging
