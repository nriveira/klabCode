#!/usr/bin/env python3
"""
Video converter and side-by-side comparison tool.
Converts AVI videos to MP4 and creates side-by-side videos of raw vs processed data.
"""

import numpy as np
import cv2
import os
import sys
from datetime import datetime

def convert_avi_to_mp4(input_path, output_path, fps=None, quality='medium'):
    """
    Convert AVI video to MP4 format.
    
    Parameters:
    - input_path: path to input AVI file
    - output_path: path for output MP4 file
    - fps: frame rate (None = use original)
    - quality: 'low', 'medium', 'high'
    """
    
    print(f"Converting: {os.path.basename(input_path)}")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use original fps if not specified
    if fps is None:
        fps = original_fps
    
    print(f"  Original: {total_frames} frames, {height}x{width}, {original_fps:.1f} fps")
    print(f"  Output: {fps:.1f} fps")
    
    # Set up codec and quality
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create output video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
    
    if not out.isOpened():
        print(f"Error: Could not create output video")
        cap.release()
        return False
    
    # Process frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    # Clean up
    cap.release()
    out.release()
    
    # Check output file
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"  ✓ Converted: {size_mb:.1f} MB")
        return True
    else:
        print(f"  ✗ Conversion failed")
        return False

def create_side_by_side_video(raw_video_path, processed_video_path, output_path, 
                             label_raw="Raw", label_processed="dF/F"):
    """
    Create side-by-side video comparison.
    
    Parameters:
    - raw_video_path: path to raw video (MP4)
    - processed_video_path: path to processed video (MP4)
    - output_path: path for combined output video
    - label_raw: text label for raw video
    - label_processed: text label for processed video
    """
    
    print(f"Creating side-by-side video...")
    print(f"  Raw: {os.path.basename(raw_video_path)}")
    print(f"  Processed: {os.path.basename(processed_video_path)}")
    
    # Open both videos
    cap_raw = cv2.VideoCapture(raw_video_path)
    cap_processed = cv2.VideoCapture(processed_video_path)
    
    if not cap_raw.isOpened() or not cap_processed.isOpened():
        print("Error: Could not open input videos")
        return False
    
    # Get video properties
    fps_raw = cap_raw.get(cv2.CAP_PROP_FPS)
    fps_processed = cap_processed.get(cv2.CAP_PROP_FPS)
    
    width_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    width_processed = int(cap_processed.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_processed = int(cap_processed.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_processed = int(cap_processed.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Raw: {frames_raw} frames, {height_raw}x{width_raw}, {fps_raw:.1f} fps")
    print(f"  Processed: {frames_processed} frames, {height_processed}x{width_processed}, {fps_processed:.1f} fps")
    
    # Determine output dimensions and fps
    # Use the smaller frame count and matching fps
    output_fps = min(fps_raw, fps_processed)
    max_frames = min(frames_raw, frames_processed)
    
    # Make heights match by scaling
    if height_raw != height_processed:
        target_height = min(height_raw, height_processed)
        scale_raw = target_height / height_raw
        scale_processed = target_height / height_processed
        
        width_raw_scaled = int(width_raw * scale_raw)
        width_processed_scaled = int(width_processed * scale_processed)
        height_scaled = target_height
    else:
        width_raw_scaled = width_raw
        width_processed_scaled = width_processed
        height_scaled = height_raw
        scale_raw = 1.0
        scale_processed = 1.0
    
    # Combined width with small gap
    gap = 10
    combined_width = width_raw_scaled + width_processed_scaled + gap
    combined_height = height_scaled + 40  # Extra space for labels
    
    print(f"  Output: {combined_height}x{combined_width}, {output_fps:.1f} fps")
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (combined_width, combined_height), True)
    
    if not out.isOpened():
        print("Error: Could not create output video")
        cap_raw.release()
        cap_processed.release()
        return False
    
    # Process frames
    frame_count = 0
    while frame_count < max_frames:
        # Read frames
        ret_raw, frame_raw = cap_raw.read()
        ret_processed, frame_processed = cap_processed.read()
        
        if not ret_raw or not ret_processed:
            break
        
        # Scale frames if needed
        if scale_raw != 1.0:
            frame_raw = cv2.resize(frame_raw, (width_raw_scaled, height_scaled))
        if scale_processed != 1.0:
            frame_processed = cv2.resize(frame_processed, (width_processed_scaled, height_scaled))
        
        # Convert processed grayscale to color if needed
        if len(frame_processed.shape) == 2:
            frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_GRAY2BGR)
        
        # Create combined frame
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Add labels
        cv2.putText(combined_frame, label_raw, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_frame, label_processed, 
                   (width_raw_scaled + gap + 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Place videos
        combined_frame[40:40+height_scaled, 0:width_raw_scaled] = frame_raw
        combined_frame[40:40+height_scaled, width_raw_scaled+gap:width_raw_scaled+gap+width_processed_scaled] = frame_processed
        
        out.write(combined_frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Combined {frame_count}/{max_frames} frames...")
    
    # Clean up
    cap_raw.release()
    cap_processed.release()
    out.release()
    
    # Check output
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"  ✓ Side-by-side video created: {size_mb:.1f} MB")
        return True
    else:
        print(f"  ✗ Side-by-side creation failed")
        return False

def create_three_way_video(raw_video_path, processed_video_path, smoothed_video_path, output_path,
                          label_raw="Raw", label_processed="dF/F", label_smoothed="dF/F Smooth"):
    """
    Create three-way video comparison: Raw | dF/F | Smoothed dF/F.
    
    Parameters:
    - raw_video_path: path to raw video (MP4)
    - processed_video_path: path to processed video (MP4)
    - smoothed_video_path: path to smoothed processed video (MP4)
    - output_path: path for combined output video
    - label_raw, label_processed, label_smoothed: text labels for videos
    """
    
    print(f"Creating three-way video comparison...")
    print(f"  Raw: {os.path.basename(raw_video_path)}")
    print(f"  Processed: {os.path.basename(processed_video_path)}")
    print(f"  Smoothed: {os.path.basename(smoothed_video_path)}")
    
    # Open all three videos
    cap_raw = cv2.VideoCapture(raw_video_path)
    cap_processed = cv2.VideoCapture(processed_video_path)
    cap_smoothed = cv2.VideoCapture(smoothed_video_path)
    
    if not cap_raw.isOpened() or not cap_processed.isOpened() or not cap_smoothed.isOpened():
        print("Error: Could not open all input videos")
        return False
    
    # Get video properties
    fps_raw = cap_raw.get(cv2.CAP_PROP_FPS)
    fps_processed = cap_processed.get(cv2.CAP_PROP_FPS)
    fps_smoothed = cap_smoothed.get(cv2.CAP_PROP_FPS)
    
    width_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    width_processed = int(cap_processed.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_processed = int(cap_processed.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    width_smoothed = int(cap_smoothed.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_smoothed = int(cap_smoothed.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_processed = int(cap_processed.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_smoothed = int(cap_smoothed.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Raw: {frames_raw} frames, {height_raw}x{width_raw}, {fps_raw:.1f} fps")
    print(f"  Processed: {frames_processed} frames, {height_processed}x{width_processed}, {fps_processed:.1f} fps")
    print(f"  Smoothed: {frames_smoothed} frames, {height_smoothed}x{width_smoothed}, {fps_smoothed:.1f} fps")
    
    # Determine output dimensions and fps
    output_fps = min(fps_raw, fps_processed, fps_smoothed)
    max_frames = min(frames_raw, frames_processed, frames_smoothed)
    
    # Make heights match by scaling
    target_height = min(height_raw, height_processed, height_smoothed)
    scale_raw = target_height / height_raw
    scale_processed = target_height / height_processed
    scale_smoothed = target_height / height_smoothed
    
    width_raw_scaled = int(width_raw * scale_raw)
    width_processed_scaled = int(width_processed * scale_processed)
    width_smoothed_scaled = int(width_smoothed * scale_smoothed)
    height_scaled = target_height
    
    # Combined width with gaps
    gap = 10
    combined_width = width_raw_scaled + width_processed_scaled + width_smoothed_scaled + 2 * gap
    combined_height = height_scaled + 40  # Extra space for labels
    
    print(f"  Output: {combined_height}x{combined_width}, {output_fps:.1f} fps")
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (combined_width, combined_height), True)
    
    if not out.isOpened():
        print("Error: Could not create output video")
        cap_raw.release()
        cap_processed.release()
        cap_smoothed.release()
        return False
    
    # Process frames
    frame_count = 0
    while frame_count < max_frames:
        # Read frames
        ret_raw, frame_raw = cap_raw.read()
        ret_processed, frame_processed = cap_processed.read()
        ret_smoothed, frame_smoothed = cap_smoothed.read()
        
        if not ret_raw or not ret_processed or not ret_smoothed:
            break
        
        # Scale frames if needed
        if scale_raw != 1.0:
            frame_raw = cv2.resize(frame_raw, (width_raw_scaled, height_scaled))
        if scale_processed != 1.0:
            frame_processed = cv2.resize(frame_processed, (width_processed_scaled, height_scaled))
        if scale_smoothed != 1.0:
            frame_smoothed = cv2.resize(frame_smoothed, (width_smoothed_scaled, height_scaled))
        
        # Convert grayscale to color if needed
        if len(frame_processed.shape) == 2:
            frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_GRAY2BGR)
        if len(frame_smoothed.shape) == 2:
            frame_smoothed = cv2.cvtColor(frame_smoothed, cv2.COLOR_GRAY2BGR)
        
        # Create combined frame
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Add labels
        cv2.putText(combined_frame, label_raw, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_frame, label_processed, 
                   (width_raw_scaled + gap + 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_frame, label_smoothed,
                   (width_raw_scaled + width_processed_scaled + 2*gap + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Place videos
        # Raw video
        combined_frame[40:40+height_scaled, 0:width_raw_scaled] = frame_raw
        # Processed video
        combined_frame[40:40+height_scaled, width_raw_scaled+gap:width_raw_scaled+gap+width_processed_scaled] = frame_processed
        # Smoothed video
        combined_frame[40:40+height_scaled, width_raw_scaled+width_processed_scaled+2*gap:width_raw_scaled+width_processed_scaled+2*gap+width_smoothed_scaled] = frame_smoothed
        
        out.write(combined_frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Combined {frame_count}/{max_frames} frames...")
    
    # Clean up
    cap_raw.release()
    cap_processed.release()
    cap_smoothed.release()
    out.release()
    
    # Check output
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"  ✓ Three-way video created: {size_mb:.1f} MB")
        return True
    else:
        print(f"  ✗ Three-way creation failed")
        return False
    """
    Create side-by-side video comparison.
    
    Parameters:
    - raw_video_path: path to raw video (MP4)
    - processed_video_path: path to processed video (MP4)
    - output_path: path for combined output video
    - label_raw: text label for raw video
    - label_processed: text label for processed video
    """
    
    print(f"Creating side-by-side video...")
    print(f"  Raw: {os.path.basename(raw_video_path)}")
    print(f"  Processed: {os.path.basename(processed_video_path)}")
    
    # Open both videos
    cap_raw = cv2.VideoCapture(raw_video_path)
    cap_processed = cv2.VideoCapture(processed_video_path)
    
    if not cap_raw.isOpened() or not cap_processed.isOpened():
        print("Error: Could not open input videos")
        return False
    
    # Get video properties
    fps_raw = cap_raw.get(cv2.CAP_PROP_FPS)
    fps_processed = cap_processed.get(cv2.CAP_PROP_FPS)
    
    width_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    width_processed = int(cap_processed.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_processed = int(cap_processed.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_processed = int(cap_processed.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Raw: {frames_raw} frames, {height_raw}x{width_raw}, {fps_raw:.1f} fps")
    print(f"  Processed: {frames_processed} frames, {height_processed}x{width_processed}, {fps_processed:.1f} fps")
    
    # Determine output dimensions and fps
    # Use the smaller frame count and matching fps
    output_fps = min(fps_raw, fps_processed)
    max_frames = min(frames_raw, frames_processed)
    
    # Make heights match by scaling
    if height_raw != height_processed:
        target_height = min(height_raw, height_processed)
        scale_raw = target_height / height_raw
        scale_processed = target_height / height_processed
        
        width_raw_scaled = int(width_raw * scale_raw)
        width_processed_scaled = int(width_processed * scale_processed)
        height_scaled = target_height
    else:
        width_raw_scaled = width_raw
        width_processed_scaled = width_processed
        height_scaled = height_raw
        scale_raw = 1.0
        scale_processed = 1.0
    
    # Combined width with small gap
    gap = 10
    combined_width = width_raw_scaled + width_processed_scaled + gap
    combined_height = height_scaled + 40  # Extra space for labels
    
    print(f"  Output: {combined_height}x{combined_width}, {output_fps:.1f} fps")
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (combined_width, combined_height), True)
    
    if not out.isOpened():
        print("Error: Could not create output video")
        cap_raw.release()
        cap_processed.release()
        return False
    
    # Process frames
    frame_count = 0
    while frame_count < max_frames:
        # Read frames
        ret_raw, frame_raw = cap_raw.read()
        ret_processed, frame_processed = cap_processed.read()
        
        if not ret_raw or not ret_processed:
            break
        
        # Scale frames if needed
        if scale_raw != 1.0:
            frame_raw = cv2.resize(frame_raw, (width_raw_scaled, height_scaled))
        if scale_processed != 1.0:
            frame_processed = cv2.resize(frame_processed, (width_processed_scaled, height_scaled))
        
        # Convert processed grayscale to color if needed
        if len(frame_processed.shape) == 2:
            frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_GRAY2BGR)
        
        # Create combined frame
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Add labels
        cv2.putText(combined_frame, label_raw, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_frame, label_processed, 
                   (width_raw_scaled + gap + 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Place videos
        combined_frame[40:40+height_scaled, 0:width_raw_scaled] = frame_raw
        combined_frame[40:40+height_scaled, width_raw_scaled+gap:width_raw_scaled+gap+width_processed_scaled] = frame_processed
        
        out.write(combined_frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Combined {frame_count}/{max_frames} frames...")
    
    # Clean up
    cap_raw.release()
    cap_processed.release()
    out.release()
    
    # Check output
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"  ✓ Side-by-side video created: {size_mb:.1f} MB")
        return True
    else:
        print(f"  ✗ Side-by-side creation failed")
        return False

def batch_convert_avis(data_folder, output_folder):
    """
    Convert all AVI files in DATA folder to MP4.
    """
    
    print("Batch Converting AVI to MP4")
    print("=" * 40)
    
    os.makedirs(output_folder, exist_ok=True)
    
    avi_files = [f for f in os.listdir(data_folder) if f.endswith('.avi')]
    
    if not avi_files:
        print("No AVI files found")
        return
    
    converted_files = []
    
    for avi_file in avi_files:
        input_path = os.path.join(data_folder, avi_file)
        output_name = avi_file.replace('.avi', '_raw.mp4')
        output_path = os.path.join(output_folder, output_name)
        
        if convert_avi_to_mp4(input_path, output_path):
            converted_files.append((avi_file, output_path))
        
        print()
    
    print(f"Converted {len(converted_files)} files:")
    for original, converted in converted_files:
        print(f"  {original} -> {os.path.basename(converted)}")
    
    return converted_files

def create_all_comparisons(base_dir):
    """
    Create side-by-side videos for all processed videos.
    """
    
    print("Creating Side-by-Side Comparisons")
    print("=" * 40)
    
    data_folder = os.path.join(base_dir, "DATA")
    processed_folder = os.path.join(base_dir, "PROCESSED", "videos")
    raw_mp4_folder = os.path.join(base_dir, "PROCESSED", "raw_mp4")
    comparison_folder = os.path.join(base_dir, "PROCESSED", "comparisons")
    
    # First convert AVIs to MP4
    print("Step 1: Converting AVI files to MP4...")
    converted_files = batch_convert_avis(data_folder, raw_mp4_folder)
    
    # Create comparisons folder
    os.makedirs(comparison_folder, exist_ok=True)
    
    print("\nStep 2: Creating comparison videos...")
    
    # Find processed videos
    processed_videos = [f for f in os.listdir(processed_folder) 
                       if f.endswith('.mp4') and 'dFoverF' in f and '_smoothed' not in f]
    
    comparisons_created = 0
    three_way_comparisons_created = 0
    
    for processed_video in processed_videos:
        # Extract base name to find matching raw video
        # e.g., "Rat-G3-vid1_dFoverF_ds4x_20250824_180226.mp4" -> "Rat-G3-vid1"
        base_name = processed_video.split('_dFoverF')[0]
        
        # Find matching raw MP4
        raw_mp4_name = f"{base_name}_raw.mp4"
        raw_mp4_path = os.path.join(raw_mp4_folder, raw_mp4_name)
        
        # Find matching smoothed video
        smoothed_video_name = processed_video.replace('.mp4', '_smoothed.mp4')
        smoothed_video_path = os.path.join(processed_folder, smoothed_video_name)
        
        if os.path.exists(raw_mp4_path):
            processed_path = os.path.join(processed_folder, processed_video)
            
            # Create 2-way comparison (raw vs processed)
            comparison_name = f"{base_name}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            comparison_path = os.path.join(comparison_folder, comparison_name)
            
            print(f"\nCreating 2-way comparison for {base_name}...")
            
            if create_side_by_side_video(raw_mp4_path, processed_path, comparison_path):
                comparisons_created += 1
            
            # Create 3-way comparison if smoothed video exists
            if os.path.exists(smoothed_video_path):
                three_way_name = f"{base_name}_three_way_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                three_way_path = os.path.join(comparison_folder, three_way_name)
                
                print(f"Creating 3-way comparison for {base_name}...")
                
                if create_three_way_video(raw_mp4_path, processed_path, smoothed_video_path, three_way_path):
                    three_way_comparisons_created += 1
            else:
                print(f"No smoothed video found for {base_name} (skipping 3-way)")
        else:
            print(f"Warning: No raw MP4 found for {processed_video}")
    
    print(f"\n✓ Created {comparisons_created} two-way comparison videos")
    print(f"✓ Created {three_way_comparisons_created} three-way comparison videos")
    print(f"✓ All videos saved in: {comparison_folder}")

if __name__ == "__main__":
    
    base_dir = "/Users/nick/Projects/klabCode"
    
    print("Video Converter and Comparison Tool")
    print("=" * 50)
    
    # Create all comparisons
    create_all_comparisons(base_dir)
    
    print("\nDone! Check the PROCESSED/comparisons folder for side-by-side videos.")
