#!/usr/bin/env python3
"""
Utility functions for organizing and managing dF/F analysis outputs.
"""

import os
import shutil
from datetime import datetime

def setup_output_directories(base_dir="/Users/nick/Projects/klabCode"):
    """
    Create organized directory structure for dF/F analysis outputs.
    
    Structure:
    PROCESSED/
    ├── videos/          # Output videos (MP4)
    ├── plots/           # Analysis plots (PNG)
    ├── data/            # Summary TIFF files
    ├── logs/            # Processing logs
    └── raw_backup/      # Backup of original videos (optional)
    """
    processed_dir = os.path.join(base_dir, "PROCESSED")
    
    subdirs = {
        'videos': 'Output dF/F videos (MP4 format)',
        'plots': 'Analysis plots and visualizations',
        'data': 'Summary data files (TIFF, CSV)',
        'logs': 'Processing logs and metadata',
        'raw_backup': 'Backup copies of original videos'
    }
    
    created_dirs = []
    
    for subdir, description in subdirs.items():
        dir_path = os.path.join(processed_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)
        created_dirs.append(dir_path)
        
        # Create README file for each subdirectory
        readme_path = os.path.join(dir_path, "README.txt")
        if not os.path.exists(readme_path):
            with open(readme_path, 'w') as f:
                f.write(f"# {subdir.upper()} Directory\n\n")
                f.write(f"{description}\n\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Created output directories:")
    for dir_path in created_dirs:
        print(f"  - {dir_path}")
    
    return processed_dir

def generate_output_paths(video_filename, base_dir="/Users/nick/Projects/klabCode", 
                         analysis_type="dFoverF", spatial_downsample=2):
    """
    Generate organized output file paths for a video analysis.
    
    Parameters:
    - video_filename: name of input video file
    - base_dir: base project directory
    - analysis_type: type of analysis (for file naming)
    - spatial_downsample: downsampling factor (for file naming)
    
    Returns:
    - Dictionary with organized output paths
    """
    processed_dir = os.path.join(base_dir, "PROCESSED")
    
    # Extract base name without extension
    base_name = os.path.splitext(video_filename)[0]
    
    # Add analysis metadata to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_suffix = f"_{analysis_type}_ds{spatial_downsample}x_{timestamp}"
    
    output_paths = {
        'video': os.path.join(processed_dir, "videos", f"{base_name}{analysis_suffix}.mp4"),
        'plot': os.path.join(processed_dir, "plots", f"{base_name}{analysis_suffix}_analysis.png"),
        'tiff_summary': os.path.join(processed_dir, "data", f"{base_name}{analysis_suffix}_summary.tif"),
        'log': os.path.join(processed_dir, "logs", f"{base_name}{analysis_suffix}_log.txt"),
        'metadata': os.path.join(processed_dir, "data", f"{base_name}{analysis_suffix}_metadata.txt")
    }
    
    return output_paths

def create_analysis_log(log_path, video_path, analysis_params, results_summary):
    """
    Create a detailed log file for the analysis.
    
    Parameters:
    - log_path: path to save log file
    - video_path: path to input video
    - analysis_params: dictionary of analysis parameters
    - results_summary: summary of results
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, 'w') as f:
        f.write("dF/F Analysis Log\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Video: {video_path}\n\n")
        
        f.write("Analysis Parameters:\n")
        f.write("-" * 20 + "\n")
        for key, value in analysis_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Results Summary:\n")
        f.write("-" * 15 + "\n")
        for key, value in results_summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Output Files:\n")
        f.write("-" * 12 + "\n")
        # This will be filled by the calling function
    
    print(f"Analysis log created: {log_path}")

def save_metadata(metadata_path, video_info, processing_info, output_info):
    """
    Save analysis metadata in a structured format.
    """
    import json
    
    metadata = {
        'video_info': video_info,
        'processing_info': processing_info,
        'output_info': output_info,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {metadata_path}")

def cleanup_temp_files(base_dir="/Users/nick/Projects/klabCode"):
    """
    Clean up temporary files from the main directory.
    """
    temp_patterns = ['*_dFoverF.tif', '*_analysis.png', '*.tmp']
    
    for pattern in temp_patterns:
        import glob
        temp_files = glob.glob(os.path.join(base_dir, pattern))
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"Removed temp file: {temp_file}")
            except:
                pass

def get_analysis_summary(base_dir="/Users/nick/Projects/klabCode"):
    """
    Generate a summary of all analyses in the PROCESSED directory.
    """
    processed_dir = os.path.join(base_dir, "PROCESSED")
    
    if not os.path.exists(processed_dir):
        print("No PROCESSED directory found.")
        return
    
    print("Analysis Summary")
    print("=" * 40)
    
    for subdir in ['videos', 'plots', 'data', 'logs']:
        subdir_path = os.path.join(processed_dir, subdir)
        if os.path.exists(subdir_path):
            files = [f for f in os.listdir(subdir_path) if not f.startswith('.') and f != 'README.txt']
            print(f"{subdir.upper()}: {len(files)} files")
            
            # Calculate total size
            total_size = 0
            for file in files:
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
            
            if total_size > 0:
                size_mb = total_size / (1024**2)
                print(f"  Total size: {size_mb:.1f} MB")
    
    print()

if __name__ == "__main__":
    # Demo of the organization system
    print("Setting up dF/F analysis output directories...")
    setup_output_directories()
    
    # Example usage
    example_paths = generate_output_paths("Rat-G1-vid1.avi", spatial_downsample=4)
    print("\nExample output paths:")
    for key, path in example_paths.items():
        print(f"  {key}: {path}")
    
    print("\nDirectory structure created!")
    get_analysis_summary()
