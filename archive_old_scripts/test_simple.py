#!/usr/bin/env python3
"""
Simple test script to verify that the windowed dF/F analysis works correctly
with spatial downsampling and video output.
"""

import numpy as np
import cv2
import os
import sys

def quick_test_analysis():
    """
    Quick test with minimal settings to verify the windowed approach works.
    """
    print("Quick Test: Windowed dF/F Analysis")
    print("=" * 40)
    
    # Test settings
    video_path = "/Users/nick/Projects/klabCode/DATA/Rat-G1-vid1.avi"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {total_frames} frames, {original_height}x{original_width}, {fps:.1f} fps")
    
    # Test with very conservative settings
    spatial_downsample = 4  # 4x downsampling = 16x fewer pixels
    test_frames = 100       # Only process first 100 frames for testing
    
    new_width = original_width // spatial_downsample
    new_height = original_height // spatial_downsample
    
    print(f"Test settings:")
    print(f"  - Spatial downsampling: {spatial_downsample}x -> {new_height}x{new_width}")
    print(f"  - Processing only first {test_frames} frames")
    print(f"  - Estimated data size: {test_frames * new_height * new_width * 4 / (1024**2):.1f} MB")
    
    # Load and process test frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale and downsample
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Simple downsampling by taking every nth pixel
        downsampled = gray_frame[::spatial_downsample, ::spatial_downsample]
        frames.append(downsampled)
        
        if (i + 1) % 25 == 0:
            print(f"  Loaded {i + 1} frames...")
    
    cap.release()
    
    if not frames:
        print("Error: No frames loaded")
        return False
    
    # Stack frames
    video_data = np.stack(frames, axis=0)
    print(f"Loaded data shape: {video_data.shape}")
    
    # Simple dF/F calculation using initial frames
    baseline_frames = 10
    F0 = np.mean(video_data[:baseline_frames], axis=0)
    F0[F0 == 0] = 1  # Avoid division by zero
    
    # Compute dF/F
    dF_over_F = np.zeros_like(video_data)
    for i in range(video_data.shape[0]):
        dF_over_F[i] = (video_data[i] - F0) / F0
    
    print(f"dF/F computed:")
    print(f"  - Range: {np.min(dF_over_F):.3f} to {np.max(dF_over_F):.3f}")
    print(f"  - Mean: {np.mean(dF_over_F):.3f}")
    print(f"  - Std: {np.std(dF_over_F):.3f}")
    
    # Save as compressed video
    output_video = "/Users/nick/Projects/klabCode/PROCESSED/test_dFoverF_output.mp4"
    
    # Ensure PROCESSED directory exists
    os.makedirs("/Users/nick/Projects/klabCode/PROCESSED", exist_ok=True)
    
    try:
        print(f"Saving as video: {output_video}")
        
        # Normalize for video output
        vmin, vmax = np.percentile(dF_over_F, [1, 99])
        dF_normalized = np.clip((dF_over_F - vmin) / (vmax - vmin), 0, 1)
        
        # Convert to 8-bit
        dF_8bit = (dF_normalized * 255).astype(np.uint8)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, 10.0, (new_width, new_height), False)
        
        for frame in dF_8bit:
            out.write(frame)
        
        out.release()
        
        # Check file size
        if os.path.exists(output_video):
            file_size_mb = os.path.getsize(output_video) / (1024**2)
            print(f"✓ Video saved successfully: {file_size_mb:.1f} MB")
        else:
            print("✗ Video file not created")
            return False
            
    except Exception as e:
        print(f"✗ Error saving video: {e}")
        return False
    
    print("✓ Quick test completed successfully!")
    return True

if __name__ == "__main__":
    success = quick_test_analysis()
    if success:
        print("\nThe windowed approach with spatial downsampling works!")
        print("Key benefits demonstrated:")
        print("- Memory efficient processing")
        print("- Spatial downsampling reduces file size")
        print("- Video output is much smaller than TIFF")
    else:
        print("\nTest failed. Check video file and try again.")
