# Given a .avi video, compute dF/F for each pixel and save as a .tif stack
# Also plot mean dF/F over time for clumps of pixels
#
# dF/F baseline methods:
# 1. Initial frames: Traditional method using mean of first N frames as F0
#    - Good when recording starts in resting state
#    - Sensitive to drift and outliers in initial frames
# 2. Percentile method: Uses low percentile (e.g., 5th) across all frames as F0
#    - More robust to tempodef analyze_video(video_filename, baseline_frames=30, n_clusters=5, 
                 baseline_method='frames', baseline_percentile=5,
                 window_size=500, overlap=50, spatial_downsample=2,
                 save_video=True, save_tiff=False):
    """
    Convenience function to analyze a single video file using windowed processing.
    
    Parameters:
    - video_filename: name of video file in DATA folder
    - baseline_frames: number of initial frames for baseline (when method='frames')
    - n_clusters: number of pixel clusters for analysis
    - baseline_method: 'frames' or 'percentile'
    - baseline_percentile: percentile for baseline (when method='percentile')
    - window_size: number of frames to process in each window
    - overlap: number of frames to overlap between windows
    - spatial_downsample: spatial downsampling factor (2 = half resolution)
    - save_video: save dF/F as compressed MP4 video (recommended, smaller files)
    - save_tiff: save dF/F as TIFF stack (large files, disable for memory efficiency)
    """otobleaching
#    - Better represents true baseline fluorescence
#    - Commonly used in calcium/voltage imaging (Chen et al. 2013, Nature 499:295-300)
#
# Windowed processing benefits:
# 1. Memory efficiency: handles large videos without memory overflow
# 2. Objective movement tolerance: when field of view > camera sensor, windowed
#    processing maintains spatial registration across the recording
# 3. Scalability: can process videos of any length
# 4. Smooth temporal transitions: overlapping windows prevent frame artifacts

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage import io
import os   
from scipy import ndimage

def compute_dF_over_F(video_path, output_tif_path, output_plot_path, baseline_frames=30, 
                     n_clusters=5, baseline_method='frames', baseline_percentile=5, 
                     window_size=500, overlap=50, spatial_downsample=2):
    """
    Compute dF/F for each pixel in a video using windowed processing and analyze pixel clusters.
    
    This implementation uses windowed processing which provides several key advantages:
    1. Memory efficiency: processes large videos without loading all frames into memory
    2. Objective movement robustness: when the microscope objective field of view is larger
       than the camera sensor, small movements during recording are naturally accommodated
       within each window without requiring explicit motion correction
    3. Temporal stability: overlapping windows ensure smooth transitions between processing chunks
    4. Scalability: can handle videos of any length regardless of available system memory
    5. Spatial efficiency: optional downsampling reduces file sizes while preserving dynamics
    
    Parameters:
    - video_path: path to input .avi file
    - output_tif_path: path to save dF/F stack as .tif
    - output_plot_path: path to save cluster plots
    - baseline_frames: number of initial frames to use for baseline F0 (when method='frames')
    - n_clusters: number of pixel clusters for analysis 
    - baseline_method: method for computing F0 baseline ('frames' or 'percentile')
        - 'frames': use mean of first baseline_frames frames (traditional method)
        - 'percentile': use specified percentile across all frames for each pixel
          (robust to photobleaching, recommended by Chen et al. 2013, Nature 499:295-300)
    - baseline_percentile: percentile to use when baseline_method='percentile' (default: 5th percentile)
    - window_size: number of frames to process in each window (larger = more memory, fewer windows)
    - overlap: number of frames to overlap between windows (larger = smoother transitions)
    - spatial_downsample: factor to downsample pixels (1=no downsampling, 2=quarter pixels, 4=1/16 pixels)
    
    Returns:
    - dF_over_F: 3D numpy array (time, height, width) containing dF/F values
    - cluster_map: 2D array showing pixel cluster assignments
    - cluster_labels: 1D array of cluster labels for active pixels
    """
    print(f"Processing video: {video_path}")
    
    # Get video properties first
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate downsampled dimensions
    width = original_width // spatial_downsample
    height = original_height // spatial_downsample
    
    print(f"Original video: {total_frames} frames, {original_height}x{original_width} pixels, {fps:.1f} fps")
    if spatial_downsample > 1:
        print(f"Spatial downsampling: {spatial_downsample}x -> {height}x{width} pixels")
        reduction_factor = (spatial_downsample ** 2)
        print(f"File size reduction: {reduction_factor}x smaller ({reduction_factor:.1f}x fewer pixels)")
    
    print(f"Processing in windows of {window_size} frames with {overlap} frame overlap")
    
    # Calculate memory estimate
    estimated_memory_gb = (total_frames * height * width * 4) / (1024**3)
    print(f"Estimated memory usage: {estimated_memory_gb:.2f} GB")
    
    # Calculate number of windows needed
    step_size = window_size - overlap
    num_windows = max(1, (total_frames - overlap) // step_size)
    if (total_frames - overlap) % step_size > 0:
        num_windows += 1
    
    print(f"Will process {num_windows} windows to cover all {total_frames} frames")
    
    # For percentile method, we need to sample frames across the entire video first
    F0 = None
    if baseline_method == 'percentile':
        print(f"Computing F0 baseline using {baseline_percentile}th percentile across all frames...")
        print("Sampling frames for percentile calculation...")
        
        # Sample frames evenly across the video for percentile calculation
        sample_frames = []
        sample_indices = np.linspace(0, total_frames-1, min(500, total_frames), dtype=int)
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                # Apply spatial downsampling for consistency
                if spatial_downsample > 1:
                    gray_frame = cv2.resize(gray_frame, (width, height), interpolation=cv2.INTER_AREA)
                sample_frames.append(gray_frame)
        
        if sample_frames:
            sample_data = np.stack(sample_frames, axis=0)
            F0 = np.percentile(sample_data, baseline_percentile, axis=0)
            print(f"F0 baseline computed from {len(sample_frames)} sample frames")
        else:
            raise ValueError("Could not read any frames for percentile calculation")
    
    # Initialize output arrays
    all_dF_over_F = []
    
    # Process video in windows
    for window_idx in range(num_windows):
        start_frame = window_idx * step_size
        end_frame = min(start_frame + window_size, total_frames)
        actual_window_size = end_frame - start_frame
        
        print(f"\nProcessing window {window_idx + 1}/{num_windows}: frames {start_frame}-{end_frame-1} ({actual_window_size} frames)")
        
        # Load frames for this window
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        window_frames = []
        
        for frame_idx in range(actual_window_size):
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # Apply spatial downsampling if requested
            if spatial_downsample > 1:
                gray_frame = cv2.resize(gray_frame, (width, height), interpolation=cv2.INTER_AREA)
            
            window_frames.append(gray_frame)
        
        if not window_frames:
            print(f"Warning: No frames loaded for window {window_idx + 1}")
            continue
            
        # Stack frames for this window
        window_data = np.stack(window_frames, axis=0)
        print(f"Loaded {len(window_frames)} frames for this window")
        
        # Compute F0 for this window if using frames method
        if baseline_method == 'frames':
            if window_idx == 0:  # Only compute F0 from first window
                frames_for_baseline = min(baseline_frames, len(window_frames))
                F0 = np.mean(window_data[:frames_for_baseline], axis=0)
                print(f"Computing F0 baseline using mean of first {frames_for_baseline} frames from first window")
            # For subsequent windows, use the F0 from the first window
        
        # Ensure F0 is computed
        if F0 is None:
            raise ValueError("F0 baseline could not be computed")
        
        # Avoid division by zero
        F0_safe = F0.copy()
        F0_safe[F0_safe == 0] = 1
        
        # Compute dF/F for this window
        print(f"Computing dF/F for window {window_idx + 1}...")
        window_dF_over_F = np.zeros_like(window_data)
        for i in range(window_data.shape[0]):
            window_dF_over_F[i] = (window_data[i] - F0_safe) / F0_safe
        
        # Handle overlap: for overlapping frames, average with previous window
        if window_idx > 0 and overlap > 0:
            overlap_start = max(0, len(all_dF_over_F) - overlap)
            overlap_frames = min(overlap, len(window_dF_over_F))
            
            # Average overlapping frames
            for i in range(overlap_frames):
                if overlap_start + i < len(all_dF_over_F):
                    # Average the overlapping frames
                    all_dF_over_F[overlap_start + i] = (all_dF_over_F[overlap_start + i] + window_dF_over_F[i]) / 2
            
            # Add non-overlapping frames
            all_dF_over_F.extend(window_dF_over_F[overlap_frames:])
        else:
            # First window or no overlap
            all_dF_over_F.extend(window_dF_over_F)
    
    cap.release()
    
    # Convert list to numpy array
    dF_over_F = np.stack(all_dF_over_F, axis=0)
    print(f"\nFinal dF/F stack shape: {dF_over_F.shape}")
    print(f"F0 baseline stats: min={np.min(F0):.2f}, max={np.max(F0):.2f}, mean={np.mean(F0):.2f}")
    
    # Save outputs (TIFF and/or video)
    outputs_saved = []
    
    # Save as compressed video (recommended for large datasets)
    video_path = output_tif_path.replace('.tif', '_dFoverF.mp4')
    try:
        print(f"Saving dF/F as compressed video: {video_path}")
        save_dF_as_video(dF_over_F, video_path, fps=30, colormap='RdBu_r')
        outputs_saved.append(f"Video: {video_path}")
    except Exception as e:
        print(f"Warning: Could not save video ({e})")
    
    # Save dF/F stack as TIFF (optional, can be memory intensive)
    try:
        print(f"Saving dF/F stack as TIFF: {output_tif_path}")
        # For large datasets, offer different saving options
        estimated_size_gb = (dF_over_F.size * 2) / (1024**3)  # 16-bit = 2 bytes per pixel
        
        if estimated_size_gb > 2:  # If > 2GB, save summary instead
            print(f"Large dataset detected ({estimated_size_gb:.1f} GB). Saving summary TIFF...")
            # Save every 10th frame to reduce size
            downsampled = dF_over_F[::10]
            dF_scaled = ((downsampled + 1) * 32767).astype(np.uint16)
            summary_path = output_tif_path.replace('.tif', '_summary.tif')
            io.imsave(summary_path, dF_scaled)
            outputs_saved.append(f"Summary TIFF: {summary_path}")
            print(f"Saved summary TIFF (every 10th frame): {summary_path}")
        else:
            # Save full TIFF for smaller datasets
            dF_scaled = ((dF_over_F + 1) * 32767).astype(np.uint16)
            io.imsave(output_tif_path, dF_scaled)
            outputs_saved.append(f"Full TIFF: {output_tif_path}")
            print(f"Saved full TIFF stack: {output_tif_path}")
            
    except Exception as e:
        print(f"Warning: Could not save TIFF ({e})")
    
    if outputs_saved:
        print(f"Outputs saved:")
        for output in outputs_saved:
            print(f"  - {output}")
    else:
        print("Warning: No outputs were saved successfully")
    
    # Perform pixel clustering based on dF/F time series (memory efficient)
    print("Performing pixel clustering...")
    height, width = dF_over_F.shape[1], dF_over_F.shape[2]
    
    # For very large datasets, subsample for clustering to save memory
    if dF_over_F.shape[0] > 2000 or (height * width) > 500000:
        print("Large dataset detected. Using subsampled data for clustering...")
        
        # Temporal subsampling: use every Nth frame
        temporal_subsample = max(1, dF_over_F.shape[0] // 1000)  # Target ~1000 frames
        time_indices = np.arange(0, dF_over_F.shape[0], temporal_subsample)
        
        # Spatial subsampling: downsample image resolution
        spatial_subsample = 2 if (height * width) > 500000 else 1
        
        if spatial_subsample > 1:
            # Downsample spatially
            new_height = height // spatial_subsample
            new_width = width // spatial_subsample
            dF_subsampled = np.zeros((len(time_indices), new_height, new_width))
            
            for i, t_idx in enumerate(time_indices):
                frame = dF_over_F[t_idx]
                # Simple downsampling by taking every nth pixel
                dF_subsampled[i] = frame[::spatial_subsample, ::spatial_subsample]
            
            print(f"Using subsampled data: {len(time_indices)} frames, {new_height}x{new_width} pixels")
        else:
            dF_subsampled = dF_over_F[time_indices]
            new_height, new_width = height, width
            print(f"Using temporally subsampled data: {len(time_indices)} frames")
    else:
        dF_subsampled = dF_over_F
        new_height, new_width = height, width
        spatial_subsample = 1
    
    # Reshape dF/F data for clustering (pixels x time)
    dF_reshaped = dF_subsampled.reshape(dF_subsampled.shape[0], -1).T
    
    # Remove pixels with very low variance (likely background)
    pixel_variance = np.var(dF_reshaped, axis=1)
    active_pixels = pixel_variance > np.percentile(pixel_variance, 25)
    
    if np.sum(active_pixels) < n_clusters:
        print(f"Warning: Only {np.sum(active_pixels)} active pixels found, reducing clusters to {np.sum(active_pixels)}")
        n_clusters = max(1, np.sum(active_pixels))
    
    # Standardize the time series for clustering
    scaler = StandardScaler()
    dF_active = dF_reshaped[active_pixels]
    dF_scaled_for_clustering = scaler.fit_transform(dF_active)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(dF_scaled_for_clustering)
    
    # Create full cluster map (upsample if needed)
    if spatial_subsample > 1:
        # Create small cluster map first
        full_cluster_labels_small = np.full(new_height * new_width, -1)
        full_cluster_labels_small[active_pixels] = cluster_labels
        cluster_map_small = full_cluster_labels_small.reshape(new_height, new_width)
        
        # Upsample cluster labels back to original resolution using nearest neighbor
        cluster_map = np.zeros((height, width), dtype=int)
        for i in range(height):
            for j in range(width):
                si, sj = i // spatial_subsample, j // spatial_subsample
                if si < new_height and sj < new_width:
                    cluster_map[i, j] = cluster_map_small[si, sj]
                else:
                    cluster_map[i, j] = -1
    else:
        # No spatial subsampling
        full_cluster_labels = np.full(height * width, -1)
        full_cluster_labels[active_pixels] = cluster_labels
        cluster_map = full_cluster_labels.reshape(height, width)
    
    # Plot results (use subsampled data for plotting if available)
    plot_data = dF_subsampled if 'dF_subsampled' in locals() else dF_over_F
    plot_cluster_analysis(plot_data, cluster_map, cluster_labels, dF_active, 
                         output_plot_path, n_clusters)
    
    return dF_over_F, cluster_map, cluster_labels

def save_dF_as_video(dF_over_F, output_video_path, fps=30, colormap='RdBu_r'):
    """
    Save dF/F data as a compressed video file for easier viewing and smaller file size.
    
    Parameters:
    - dF_over_F: 3D numpy array (time, height, width)
    - output_video_path: path to save output video
    - fps: frames per second for output video
    - colormap: matplotlib colormap for visualization ('RdBu_r', 'viridis', 'plasma')
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    print(f"Saving dF/F as video: {output_video_path}")
    
    # Get data range for normalization
    vmin, vmax = np.percentile(dF_over_F, [1, 99])  # Use 1st-99th percentile for robust scaling
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Get video dimensions
    n_frames, height, width = dF_over_F.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 for good compression
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), True)
    
    print(f"Processing {n_frames} frames for video...")
    
    for i in range(n_frames):
        if i % 100 == 0:
            print(f"Video frame {i}/{n_frames}")
        
        # Normalize and apply colormap
        frame_normalized = norm(dF_over_F[i])
        frame_colored = cmap(frame_normalized)
        
        # Convert to 8-bit BGR for OpenCV (matplotlib uses RGBA)
        frame_bgr = (frame_colored[:, :, :3] * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
        
        out.write(frame_bgr)
    
    out.release()
    
    # Get file size
    file_size_mb = os.path.getsize(output_video_path) / (1024*1024)
    print(f"Video saved! Size: {file_size_mb:.1f} MB")
    
    return output_video_path

def plot_cluster_analysis(dF_over_F, cluster_map, cluster_labels, dF_active, 
                         output_plot_path, n_clusters):
    """
    Create comprehensive plots of the cluster analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Cluster map
    ax1 = axes[0, 0]
    im1 = ax1.imshow(cluster_map, cmap='tab10', vmin=-1, vmax=n_clusters-1)
    ax1.set_title('Pixel Clusters')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Height (pixels)')
    plt.colorbar(im1, ax=ax1, label='Cluster ID')
    
    # Plot 2: Mean dF/F baseline image
    ax2 = axes[0, 1]
    mean_dF = np.mean(dF_over_F, axis=0)
    im2 = ax2.imshow(mean_dF, cmap='RdBu_r', vmin=np.percentile(mean_dF, 1), 
                     vmax=np.percentile(mean_dF, 99))
    ax2.set_title('Mean dF/F')
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    plt.colorbar(im2, ax=ax2, label='dF/F')
    
    # Plot 3: Cluster time traces
    ax3 = axes[1, 0]
    time_points = np.arange(dF_over_F.shape[0])
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for cluster_id in range(n_clusters):
        cluster_pixels = cluster_labels == cluster_id
        if np.sum(cluster_pixels) > 0:
            cluster_trace = np.mean(dF_active[cluster_pixels], axis=0)
            ax3.plot(time_points, cluster_trace, color=colors[cluster_id], 
                    label=f'Cluster {cluster_id} (n={np.sum(cluster_pixels)})',
                    linewidth=2)
    
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Mean dF/F')
    ax3.set_title('Cluster Time Traces')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Standard deviation of dF/F
    ax4 = axes[1, 1]
    std_dF = np.std(dF_over_F, axis=0)
    im4 = ax4.imshow(std_dF, cmap='viridis', vmin=0, vmax=np.percentile(std_dF, 95))
    ax4.set_title('dF/F Standard Deviation')
    ax4.set_xlabel('Width (pixels)')
    ax4.set_ylabel('Height (pixels)')
    plt.colorbar(im4, ax=ax4, label='Std dF/F')
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to: {output_plot_path}")

def analyze_video(video_filename, baseline_frames=30, n_clusters=5, 
                 baseline_method='frames', baseline_percentile=5,
                 window_size=500, overlap=50, spatial_downsample=2):
    """
    Convenience function to analyze a single video file using windowed processing.
    
    Parameters:
    - video_filename: name of video file in DATA folder
    - baseline_frames: number of initial frames for baseline (when method='frames')
    - n_clusters: number of pixel clusters for analysis
    - baseline_method: 'frames' or 'percentile'
    - baseline_percentile: percentile for baseline (when method='percentile')
    - window_size: number of frames to process in each window
    - overlap: number of frames to overlap between windows
    - spatial_downsample: factor to downsample pixels (2=quarter size, 4=sixteenth size)
    """
    # Setup paths
    base_dir = "/Users/nick/Projects/klabCode"
    video_path = os.path.join(base_dir, "DATA", video_filename)
    
    # Create output filenames
    base_name = os.path.splitext(video_filename)[0]
    output_tif_path = os.path.join(base_dir, f"{base_name}_dFoverF.tif")
    output_plot_path = os.path.join(base_dir, f"{base_name}_analysis.png")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    # Run analysis
    try:
        dF_over_F, cluster_map, cluster_labels = compute_dF_over_F(
            video_path, output_tif_path, output_plot_path, 
            baseline_frames, n_clusters, baseline_method, baseline_percentile,
            window_size, overlap, spatial_downsample
        )
        print(f"Analysis complete for {video_filename}")
        return dF_over_F, cluster_map, cluster_labels
    except Exception as e:
        print(f"Error analyzing {video_filename}: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Analyze one of the available videos
    video_file = "Rat-G1-vid1.avi"  # Change this to analyze different videos
    
    print("Starting voltage indicator dF/F analysis...")
    print("=" * 50)
    
    # Windowed processing settings for memory efficiency
    window_size = 500        # Process 500 frames at a time
    overlap = 50             # 50 frame overlap between windows for smooth transitions
    spatial_downsample = 2   # 2x spatial downsampling (1080x1440 -> 540x720)
    
    print(f"Processing settings:")
    print(f"- Window size: {window_size} frames")
    print(f"- Overlap: {overlap} frames") 
    print(f"- Spatial downsampling: {spatial_downsample}x (reduces file size {spatial_downsample**2}x)")
    
    # Example 1: Traditional method using initial frames
    print("\n1. Analysis using initial frames as baseline (windowed processing):")
    result1 = analyze_video(video_file, 
                           baseline_frames=50, 
                           n_clusters=6, 
                           baseline_method='frames',
                           window_size=window_size,
                           overlap=overlap,
                           spatial_downsample=spatial_downsample)
    
    # Example 2: Percentile method using 5th percentile
    print("\n2. Analysis using 5th percentile as baseline (windowed processing):")
    result2 = analyze_video(video_file, 
                           baseline_frames=50, 
                           n_clusters=6,
                           baseline_method='percentile', 
                           baseline_percentile=5,
                           window_size=window_size,
                           overlap=overlap,
                           spatial_downsample=spatial_downsample)
    
    # Compare results if both succeeded
    if result1 is not None and result2 is not None:
        dF_frames, _, _ = result1
        dF_percentile, _, _ = result2
        
        print(f"\nComparison of baseline methods:")
        print(f"Frames method - dF/F range: {np.min(dF_frames):.3f} to {np.max(dF_frames):.3f}")
        print(f"Percentile method - dF/F range: {np.min(dF_percentile):.3f} to {np.max(dF_percentile):.3f}")
        
        # Calculate correlation between methods
        correlation = np.corrcoef(dF_frames.flatten(), dF_percentile.flatten())[0,1]
        print(f"Correlation between methods: {correlation:.3f}")
    
    print("\nWindowed processing benefits:")
    print("- Memory efficient: processes large videos in manageable chunks")
    print("- Scalable: can handle videos of any length")
    print("- Smooth transitions: overlapping windows prevent artifacts")
    print("- Objective movement tolerance: accommodates small movements when")
    print("  microscope field of view is larger than camera sensor")
    print("- Spatial efficiency: downsampling reduces file sizes dramatically")
    print("\nTuning parameters:")
    print(f"- window_size ({window_size}): larger = more memory, fewer processing windows")
    print(f"- overlap ({overlap}): larger = smoother transitions, more computation")
    print(f"- spatial_downsample ({spatial_downsample}): larger = smaller files, less spatial detail")
    print("  Common values: 1 (no downsampling), 2 (quarter size), 4 (sixteenth size)")
    print("\nFile size impact:")
    original_size_gb = (6776 * 1080 * 1440 * 2) / (1024**3)  # Rough estimate for 16-bit TIFF
    downsampled_size_gb = original_size_gb / (spatial_downsample ** 2)
    print(f"- Original estimated size: ~{original_size_gb:.1f} GB")
    print(f"- With {spatial_downsample}x downsampling: ~{downsampled_size_gb:.1f} GB")
    print("\nAvailable baseline methods:")
    print("  - 'frames': Use mean of initial frames (traditional)")
    print("  - 'percentile': Use specified percentile across all frames (robust to drift)")
    print("    Recommended by Chen et al. (2013) Nature 499:295-300")
    print("Common percentiles: 5th (conservative), 10th (moderate), 20th (liberal)")
    print("\nKey references:")
    print("- Chen, T.W. et al. (2013) Ultrasensitive fluorescent proteins for")
    print("  imaging neuronal activity. Nature 499:295-300")
    print("  [Established percentile-based F0 calculation for calcium imaging]")