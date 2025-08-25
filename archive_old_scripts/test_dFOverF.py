#!/usr/bin/env python3
"""
Test script to demonstrate dF/F analysis with synthetic voltage indicator data.
This creates a synthetic video with known signal characteristics to test the analysis.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from dFOverF import compute_dF_over_F, analyze_video, plot_cluster_analysis
import os

def create_synthetic_voltage_data(width=64, height=64, n_frames=200, fps=30):
    """
    Create synthetic voltage indicator data that mimics real neural recordings.
    
    Parameters:
    - width, height: spatial dimensions of the video
    - n_frames: number of time frames
    - fps: frames per second (for realistic timing)
    
    Returns:
    - video_data: 3D numpy array (time, height, width)
    """
    print(f"Creating synthetic voltage indicator data: {n_frames} frames of {height}x{width} pixels")
    
    # Create time axis
    time = np.arange(n_frames) / fps
    
    # Create spatial grid
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    
    # Initialize video array
    video_data = np.zeros((n_frames, height, width), dtype=np.float32)
    
    # Create baseline fluorescence map (heterogeneous)
    baseline_F = 100 + 50 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    
    # Add different types of signals
    for frame in range(n_frames):
        # Start with baseline
        current_frame = baseline_F.copy()
        
        # Add slow photobleaching (1% decrease over recording)
        bleach_factor = 1 - 0.01 * (frame / n_frames)
        current_frame *= bleach_factor
        
        # Region 1: High frequency oscillations (voltage spikes)
        mask1 = (x < 0.3) & (y < 0.3)
        signal1 = 15 * np.sin(2 * np.pi * 10 * time[frame]) * np.exp(-((x-0.15)**2 + (y-0.15)**2) / 0.02)
        current_frame[mask1] += signal1[mask1]
        
        # Region 2: Slower waves (dendritic activity)
        mask2 = (x > 0.7) & (y > 0.7)
        signal2 = 25 * np.sin(2 * np.pi * 2 * time[frame]) * np.exp(-((x-0.85)**2 + (y-0.85)**2) / 0.05)
        current_frame[mask2] += signal2[mask2]
        
        # Region 3: Transient events (action potentials)
        if frame > 50 and frame < 60:  # Brief event
            mask3 = (x > 0.4) & (x < 0.6) & (y > 0.4) & (y < 0.6)
            signal3 = 40 * np.exp(-((frame-55)**2) / 10) * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)
            current_frame[mask3] += signal3[mask3]
        
        # Add noise
        noise = np.random.normal(0, 2, (height, width))
        current_frame += noise
        
        # Ensure positive values
        current_frame = np.maximum(current_frame, 1)
        
        video_data[frame] = current_frame
    
    return video_data

def save_synthetic_video(video_data, output_path, fps=30):
    """
    Save synthetic data as an AVI video file.
    """
    print(f"Saving synthetic video to: {output_path}")
    
    height, width = video_data.shape[1], video_data.shape[2]
    
    # Convert to 8-bit for video saving
    video_8bit = ((video_data - np.min(video_data)) / 
                  (np.max(video_data) - np.min(video_data)) * 255).astype(np.uint8)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
    
    for frame in video_8bit:
        out.write(frame)
    
    out.release()
    print(f"Video saved successfully: {os.path.getsize(output_path)} bytes")

def test_both_baseline_methods():
    """
    Test both baseline methods on synthetic data.
    """
    # Create synthetic data
    video_data = create_synthetic_voltage_data(width=80, height=80, n_frames=150)
    
    # Save as video file
    synthetic_video_path = "/Users/nick/Projects/klabCode/DATA/synthetic_voltage_data.avi"
    save_synthetic_video(video_data, synthetic_video_path)
    
    # Test both baseline methods
    print("\n" + "="*60)
    print("TESTING BASELINE METHODS")
    print("="*60)
    
    # Method 1: Initial frames baseline
    print("\n1. FRAMES-BASED BASELINE (Traditional Method)")
    print("-" * 45)
    try:
        result1 = analyze_video("synthetic_voltage_data.avi", 
                               baseline_frames=30, 
                               n_clusters=4, 
                               baseline_method='frames')
        if result1:
            dF_frames, cluster_map1, labels1 = result1
            print(f"✓ Success: dF/F range = {np.min(dF_frames):.3f} to {np.max(dF_frames):.3f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Method 2: Percentile baseline
    print("\n2. PERCENTILE-BASED BASELINE (Robust Method)")
    print("-" * 48)
    try:
        result2 = analyze_video("synthetic_voltage_data.avi", 
                               baseline_frames=30, 
                               n_clusters=4, 
                               baseline_method='percentile', 
                               baseline_percentile=5)
        if result2:
            dF_percentile, cluster_map2, labels2 = result2
            print(f"✓ Success: dF/F range = {np.min(dF_percentile):.3f} to {np.max(dF_percentile):.3f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Compare methods if both succeeded
    if 'result1' in locals() and 'result2' in locals() and result1 and result2:
        print("\n3. COMPARISON OF METHODS")
        print("-" * 25)
        
        # Calculate correlation
        correlation = np.corrcoef(dF_frames.flatten(), dF_percentile.flatten())[0,1]
        print(f"Spatial-temporal correlation: {correlation:.3f}")
        
        # Compare signal-to-noise ratio
        snr_frames = np.var(np.mean(dF_frames, axis=(1,2))) / np.var(dF_frames)
        snr_percentile = np.var(np.mean(dF_percentile, axis=(1,2))) / np.var(dF_percentile)
        print(f"SNR (frames method): {snr_frames:.4f}")
        print(f"SNR (percentile method): {snr_percentile:.4f}")
        
        # Photobleaching correction assessment
        mean_trace_frames = np.mean(dF_frames, axis=(1,2))
        mean_trace_percentile = np.mean(dF_percentile, axis=(1,2))
        
        # Linear trend (indicates photobleaching correction)
        from scipy import stats
        slope_frames, _, _, p_frames, _ = stats.linregress(range(len(mean_trace_frames)), mean_trace_frames)
        slope_percentile, _, _, p_percentile, _ = stats.linregress(range(len(mean_trace_percentile)), mean_trace_percentile)
        
        print(f"Temporal drift (frames): slope={slope_frames:.6f}, p={p_frames:.3f}")
        print(f"Temporal drift (percentile): slope={slope_percentile:.6f}, p={p_percentile:.3f}")
        
        if abs(slope_percentile) < abs(slope_frames):
            print("✓ Percentile method better corrects for photobleaching drift")
        else:
            print("? Frames method shows less temporal drift")

if __name__ == "__main__":
    print("Testing dF/F Analysis with Synthetic Voltage Indicator Data")
    print("=" * 65)
    
    test_both_baseline_methods()
    
    print("\n" + "="*65)
    print("SUMMARY: Methods for Computing dF/F Baseline")
    print("="*65)
    print("1. FRAMES METHOD (Traditional):")
    print("   - Uses mean of initial N frames as F0")
    print("   - Good when recording starts at rest")
    print("   - Sensitive to photobleaching and initial conditions")
    print()
    print("2. PERCENTILE METHOD (Robust):")
    print("   - Uses low percentile (e.g., 5th) across all frames")
    print("   - Robust to photobleaching and temporal drift")
    print("   - Better represents true baseline fluorescence")
    print("   - Recommended for voltage imaging (Chen et al. 2013)")
    print()
    print("Choose method based on your experimental conditions!")
