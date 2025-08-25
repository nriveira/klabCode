#!/usr/bin/env python3
"""
Quick test of windowed dF/F analysis with a smaller subset of frames.
This tests the functionality without processing the entire large video.
"""

import sys
import os
sys.path.append('/Users/nick/Projects/klabCode')

from dFOverF import analyze_video
import numpy as np

def test_windowed_analysis():
    """
    Test the windowed dF/F analysis with conservative settings.
    """
    print("Testing Windowed dF/F Analysis")
    print("=" * 40)
    
    video_file = "Rat-G1-vid1.avi"
    
    # Very conservative settings for testing
    test_settings = {
        'baseline_frames': 30,
        'n_clusters': 4,
        'window_size': 200,     # Smaller windows for testing
        'overlap': 20,          # Small overlap for faster processing
        'spatial_downsample': 4 # 4x downsampling for much smaller files
    }
    
    print(f"Test settings: {test_settings}")
    print(f"Spatial downsampling will reduce 1080x1440 to {1080//4}x{1440//4} pixels")
    print(f"File size reduction: {4**2}x smaller (16x fewer pixels)")
    
    # Test 1: Frames method
    print("\nTesting frames-based baseline method...")
    try:
        result1 = analyze_video(
            video_file,
            baseline_method='frames',
            **test_settings
        )
        
        if result1:
            dF_frames, cluster_map, cluster_labels = result1
            print(f"✓ Success! dF/F shape: {dF_frames.shape}")
            print(f"✓ dF/F range: {np.min(dF_frames):.3f} to {np.max(dF_frames):.3f}")
            print(f"✓ Clusters found: {len(np.unique(cluster_labels))}")
        else:
            print("✗ Failed to analyze video")
            
    except Exception as e:
        print(f"✗ Error in frames method: {e}")
    
    # Test 2: Percentile method
    print("\nTesting percentile-based baseline method...")
    try:
        result2 = analyze_video(
            video_file,
            baseline_method='percentile',
            baseline_percentile=5,
            **test_settings
        )
        
        if result2:
            dF_percentile, cluster_map2, cluster_labels2 = result2
            print(f"✓ Success! dF/F shape: {dF_percentile.shape}")
            print(f"✓ dF/F range: {np.min(dF_percentile):.3f} to {np.max(dF_percentile):.3f}")
            print(f"✓ Clusters found: {len(np.unique(cluster_labels2))}")
            
            # Compare methods if both worked
            if 'result1' in locals() and result1:
                correlation = np.corrcoef(dF_frames.flatten(), dF_percentile.flatten())[0,1]
                print(f"✓ Method correlation: {correlation:.3f}")
        else:
            print("✗ Failed to analyze video")
            
    except Exception as e:
        print(f"✗ Error in percentile method: {e}")
    
    print("\n" + "=" * 40)
    print("Test completed!")
    
    # Check for output files
    base_name = video_file.replace('.avi', '')
    tif_file = f"/Users/nick/Projects/klabCode/{base_name}_dFoverF.tif"
    png_file = f"/Users/nick/Projects/klabCode/{base_name}_analysis.png"
    
    if os.path.exists(tif_file):
        size_mb = os.path.getsize(tif_file) / (1024*1024)
        print(f"✓ TIFF output created: {size_mb:.1f} MB")
    else:
        print("? No TIFF output found (may be expected for large files)")
        
    if os.path.exists(png_file):
        print(f"✓ Analysis plot created: {png_file}")
    else:
        print("? No analysis plot found")

if __name__ == "__main__":
    test_windowed_analysis()
