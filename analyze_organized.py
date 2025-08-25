#!/usr/bin/env python3
"""
Organized dF/F analysis with proper output management.
Uses spatial downsampling and saves outputs to PROCESSED directory.
"""

import numpy as np
import cv2
import os
import sys
from datetime import datetime

# Import our output manager
from output_manager import setup_output_directories, generate_output_paths, create_analysis_log, save_metadata

def analyze_video_organized(video_filename, spatial_downsample=None, max_frames=None, 
                           baseline_method='percentile', baseline_percentile=5):
    """
    Analyze video with organized output management.
    
    Parameters:
    - video_filename: name of video file in DATA folder
    - spatial_downsample: spatial downsampling factor (None = auto-determine from resolution)
    - max_frames: maximum frames to process (None = all)
    - baseline_method: 'frames' or 'percentile'
    - baseline_percentile: percentile for baseline calculation
    """
    
    print(f"Organized dF/F Analysis")
    print("=" * 50)
    
    # Setup directory structure
    base_dir = "/Users/nick/Projects/klabCode"
    setup_output_directories(base_dir)
    
    # Input video path
    video_path = os.path.join(base_dir, "DATA", video_filename)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    # Get video properties first to determine optimal downsampling
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Auto-determine spatial downsampling based on resolution
    if spatial_downsample is None:
        # Calculate total pixels
        total_pixels = original_width * original_height
        
        if total_pixels <= 400000:  # ~640x625 or smaller (e.g., 608x608 = 369,664)
            spatial_downsample = 1  # Light downsampling for smaller videos
            reasoning = "small resolution"
        elif total_pixels <= 1000000:  # ~1000x1000 or smaller
            spatial_downsample = 2  # Moderate downsampling
            reasoning = "medium resolution"
        elif total_pixels <= 2000000:  # ~1414x1414 or smaller (e.g., 1080x1440 = 1,555,200)
            spatial_downsample = 4  # Standard downsampling
            reasoning = "large resolution"
        else:  # Very large videos
            spatial_downsample = 6  # Aggressive downsampling for very large videos
            reasoning = "very large resolution"
        
        print(f"Auto-selected {spatial_downsample}x downsampling for {reasoning} ({original_height}x{original_width})")
    else:
        print(f"Using manual {spatial_downsample}x downsampling")
    
    # Generate output paths (now that we know the downsampling factor)
    output_paths = generate_output_paths(video_filename, base_dir, "dFoverF", spatial_downsample)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    # Analysis parameters
    analysis_params = {
        'video_filename': video_filename,
        'spatial_downsample': spatial_downsample,
        'max_frames': max_frames,
        'baseline_method': baseline_method,
        'baseline_percentile': baseline_percentile,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"Input: {video_path}")
    print(f"Parameters: {analysis_params}")
    
    # Calculate processing parameters
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
    new_width = original_width // spatial_downsample
    new_height = original_height // spatial_downsample
    
    video_info = {
        'total_frames': total_frames,
        'original_dimensions': [original_height, original_width],
        'processed_dimensions': [new_height, new_width],
        'fps': fps,
        'frames_processed': frames_to_process
    }
    
    print(f"Video: {total_frames} frames, {original_height}x{original_width}, {fps:.1f} fps")
    print(f"Processing: {frames_to_process} frames -> {new_height}x{new_width} pixels")
    
    # Estimate memory usage
    estimated_memory_mb = (frames_to_process * new_height * new_width * 4) / (1024**2)
    print(f"Estimated memory: {estimated_memory_mb:.1f} MB")
    
    # Load and process frames
    print(f"Loading frames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    
    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale and downsample
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        downsampled = gray_frame[::spatial_downsample, ::spatial_downsample]
        frames.append(downsampled)
        
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{frames_to_process} frames...")
    
    cap.release()
    
    if not frames:
        print("Error: No frames loaded")
        return None
    
    # Stack frames and compute dF/F
    video_data = np.stack(frames, axis=0)
    print(f"Data shape: {video_data.shape}")
    
    # Compute baseline F0
    if baseline_method == 'frames':
        baseline_frames = min(50, len(frames) // 4)
        F0 = np.mean(video_data[:baseline_frames], axis=0)
        print(f"F0 from first {baseline_frames} frames")
    else:  # percentile method
        F0 = np.percentile(video_data, baseline_percentile, axis=0)
        print(f"F0 from {baseline_percentile}th percentile")
    
    F0[F0 == 0] = 1  # Avoid division by zero
    
    # Compute dF/F
    print("Computing dF/F...")
    dF_over_F = np.zeros_like(video_data)
    for i in range(video_data.shape[0]):
        dF_over_F[i] = (video_data[i] - F0) / F0
    
    # Results summary
    results_summary = {
        'dF_range_min': float(np.min(dF_over_F)),
        'dF_range_max': float(np.max(dF_over_F)),
        'dF_mean': float(np.mean(dF_over_F)),
        'dF_std': float(np.std(dF_over_F)),
        'baseline_method': baseline_method,
        'F0_min': float(np.min(F0)),
        'F0_max': float(np.max(F0)),
        'F0_mean': float(np.mean(F0))
    }
    
    print(f"dF/F results:")
    print(f"  Range: {results_summary['dF_range_min']:.3f} to {results_summary['dF_range_max']:.3f}")
    print(f"  Mean: {results_summary['dF_mean']:.3f} ± {results_summary['dF_std']:.3f}")
    
    # Save outputs
    output_info = {}
    
    # 1. Save as video (original dF/F)
    try:
        print(f"Saving video: {output_paths['video']}")
        
        # Normalize for video
        vmin, vmax = np.percentile(dF_over_F, [5, 95])  # More conservative percentiles
        dF_normalized = np.clip((dF_over_F - vmin) / (vmax - vmin), 0, 1)
        dF_8bit = (dF_normalized * 255).astype(np.uint8)
        
        # Create video with original frame rate
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_paths['video'], fourcc, fps, (new_width, new_height), False)
        
        for frame in dF_8bit:
            out.write(frame)
        out.release()
        
        video_size_mb = os.path.getsize(output_paths['video']) / (1024**2)
        output_info['video'] = f"{output_paths['video']} ({video_size_mb:.1f} MB)"
        print(f"  ✓ Video saved: {video_size_mb:.1f} MB")
        
    except Exception as e:
        print(f"  ✗ Video save failed: {e}")
    
    # 1b. Save temporally smoothed video
    try:
        # Create smoothed output path
        smoothed_video_path = output_paths['video'].replace('.mp4', '_smoothed.mp4')
        print(f"Saving smoothed video: {smoothed_video_path}")
        
        # Apply temporal smoothing (moving average)
        smoothing_window = 5  # 5-frame moving average
        dF_smoothed = np.copy(dF_over_F)
        
        for i in range(len(dF_over_F)):
            start_idx = max(0, i - smoothing_window // 2)
            end_idx = min(len(dF_over_F), i + smoothing_window // 2 + 1)
            dF_smoothed[i] = np.mean(dF_over_F[start_idx:end_idx], axis=0)
        
        # Normalize smoothed data
        vmin_smooth, vmax_smooth = np.percentile(dF_smoothed, [5, 95])
        dF_smooth_normalized = np.clip((dF_smoothed - vmin_smooth) / (vmax_smooth - vmin_smooth), 0, 1)
        dF_smooth_8bit = (dF_smooth_normalized * 255).astype(np.uint8)
        
        # Create smoothed video
        out_smooth = cv2.VideoWriter(smoothed_video_path, fourcc, fps, (new_width, new_height), False)
        
        for frame in dF_smooth_8bit:
            out_smooth.write(frame)
        out_smooth.release()
        
        smoothed_size_mb = os.path.getsize(smoothed_video_path) / (1024**2)
        output_info['video_smoothed'] = f"{smoothed_video_path} ({smoothed_size_mb:.1f} MB)"
        print(f"  ✓ Smoothed video saved: {smoothed_size_mb:.1f} MB")
        
        # Update results summary with smoothing info
        results_summary['smoothing_window'] = smoothing_window
        results_summary['smoothed_range_min'] = float(np.min(dF_smoothed))
        results_summary['smoothed_range_max'] = float(np.max(dF_smoothed))
        
    except Exception as e:
        print(f"  ✗ Smoothed video save failed: {e}")
    
    # 2. Save summary TIFF (every 10th frame)
    try:
        print(f"Saving summary TIFF: {output_paths['tiff_summary']}")
        
        # Downsample temporally for TIFF
        tiff_data = dF_over_F[::10]  # Every 10th frame
        tiff_scaled = ((tiff_data + 1) * 32767).astype(np.uint16)
        
        # Use skimage if available, otherwise skip
        try:
            from skimage import io
            io.imsave(output_paths['tiff_summary'], tiff_scaled)
            
            tiff_size_mb = os.path.getsize(output_paths['tiff_summary']) / (1024**2)
            output_info['tiff'] = f"{output_paths['tiff_summary']} ({tiff_size_mb:.1f} MB)"
            print(f"  ✓ Summary TIFF saved: {tiff_size_mb:.1f} MB")
        except ImportError:
            print(f"  ⚠ Skipped TIFF (skimage not available)")
            
    except Exception as e:
        print(f"  ✗ TIFF save failed: {e}")
    
    # 3. Create analysis log
    try:
        create_analysis_log(output_paths['log'], video_path, analysis_params, results_summary)
        
        # Add output file info to log
        with open(output_paths['log'], 'a') as f:
            for key, value in output_info.items():
                f.write(f"{key}: {value}\n")
        
        output_info['log'] = output_paths['log']
        
    except Exception as e:
        print(f"  ✗ Log creation failed: {e}")
    
    # 4. Save metadata
    try:
        processing_info = {
            'analysis_params': analysis_params,
            'results_summary': results_summary
        }
        
        save_metadata(output_paths['metadata'], video_info, processing_info, output_info)
        
    except Exception as e:
        print(f"  ✗ Metadata save failed: {e}")
    
    print(f"\nAnalysis complete!")
    print(f"Outputs saved to: /Users/nick/Projects/klabCode/PROCESSED/")
    
    return {
        'dF_over_F': dF_over_F,
        'video_info': video_info,
        'results_summary': results_summary,
        'output_paths': output_paths
    }

if __name__ == "__main__":
    base_dir = "/Users/nick/Projects/klabCode"
    data_folder = os.path.join(base_dir, "DATA")
    
    # Get all AVI files in DATA folder
    avi_files = [f for f in os.listdir(data_folder) 
                 if f.endswith('.avi') and not f.endswith('.part')]
    
    # Sort files for consistent processing order
    avi_files.sort()
    
    if not avi_files:
        print("No AVI files found in DATA folder!")
        sys.exit(1)
    
    # Sort files for consistent processing order
    avi_files.sort()
    
    print("Batch dF/F Analysis - All Videos")
    print("=" * 60)
    print(f"Found {len(avi_files)} videos to process:")
    for i, video in enumerate(avi_files, 1):
        print(f"  {i}. {video}")
    print()
    
    # Consistent processing parameters for all videos
    analysis_params = {
        'spatial_downsample': None,  # Auto-determine based on video resolution
        'max_frames': 1000,  # Process 1000 frames for extended temporal coverage
        'baseline_method': 'percentile',
        'baseline_percentile': 10
    }
    
    print("Analysis Parameters:")
    print(f"  - Spatial downsampling: Auto-determined based on resolution")
    print(f"  - Max frames per video: {analysis_params['max_frames']}")
    print(f"  - Baseline method: {analysis_params['baseline_method']} ({analysis_params['baseline_percentile']}th percentile)")
    print(f"  - Video frame rate: Original (preserved)")
    print(f"  - Video normalization: 5th-95th percentiles")
    print()
    
    # Process each video
    successful_analyses = []
    failed_analyses = []
    
    for i, video_file in enumerate(avi_files, 1):
        print(f"Processing Video {i}/{len(avi_files)}: {video_file}")
        print("-" * 50)
        
        try:
            # Run analysis
            result = analyze_video_organized(
                video_filename=video_file,
                spatial_downsample=analysis_params['spatial_downsample'],
                max_frames=analysis_params['max_frames'],
                baseline_method=analysis_params['baseline_method'],
                baseline_percentile=analysis_params['baseline_percentile']
            )
            
            if result:
                successful_analyses.append({
                    'filename': video_file,
                    'result': result
                })
                print(f"✓ {video_file} - SUCCESS")
            else:
                failed_analyses.append(video_file)
                print(f"✗ {video_file} - FAILED")
                
        except Exception as e:
            failed_analyses.append(video_file)
            print(f"✗ {video_file} - ERROR: {e}")
        
        print()  # Space between videos
    
    # Final summary
    print("=" * 60)
    print("BATCH ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Successful: {len(successful_analyses)}/{len(avi_files)} videos")
    print(f"Failed: {len(failed_analyses)}/{len(avi_files)} videos")
    
    if successful_analyses:
        print("\n✅ Successfully Processed:")
        for analysis in successful_analyses:
            result = analysis['result']
            print(f"  • {analysis['filename']}")
            print(f"    - Dimensions: {result['video_info']['original_dimensions']} → {result['video_info']['processed_dimensions']}")
            print(f"    - Frames: {result['video_info']['frames_processed']}/{result['video_info']['total_frames']}")
            print(f"    - dF/F range: {result['results_summary']['dF_range_min']:.3f} to {result['results_summary']['dF_range_max']:.3f}")
            print(f"    - FPS: {result['video_info']['fps']:.1f}")
    
    if failed_analyses:
        print(f"\n❌ Failed to Process:")
        for video in failed_analyses:
            print(f"  • {video}")
    
    # Show final organized output summary
    print(f"\n📁 Output Location: {base_dir}/PROCESSED/")
    from output_manager import get_analysis_summary
    get_analysis_summary(base_dir)
