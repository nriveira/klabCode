#!/usr/bin/env python3
"""
Video integrity checker - tests if video files can be opened and read properly.
"""

import cv2
import os

def check_video_integrity(video_path, sample_frames=10):
    """
    Check if a video file is readable and not corrupted.
    
    Parameters:
    - video_path: path to video file
    - sample_frames: number of frames to test reading
    
    Returns:
    - dict with integrity results
    """
    
    print(f"Checking: {os.path.basename(video_path)}")
    
    if not os.path.exists(video_path):
        return {
            'status': 'missing',
            'readable': False,
            'error': 'File does not exist'
        }
    
    try:
        # Try to open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            cap.release()
            return {
                'status': 'corrupt',
                'readable': False,
                'error': 'Cannot open video file'
            }
        
        # Get basic properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Properties: {total_frames} frames, {height}x{width}, {fps:.1f} fps")
        
        # Test reading frames
        frames_read = 0
        frames_to_test = min(sample_frames, total_frames)
        
        for i in range(frames_to_test):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames_read += 1
            else:
                break
        
        cap.release()
        
        # Determine status
        if frames_read == 0:
            status = 'corrupt'
            readable = False
            error = 'Cannot read any frames'
        elif frames_read < frames_to_test:
            status = 'partial'
            readable = True
            error = f'Could only read {frames_read}/{frames_to_test} test frames'
        else:
            status = 'good'
            readable = True
            error = None
        
        print(f"  Result: {status.upper()} - Read {frames_read}/{frames_to_test} test frames")
        
        return {
            'status': status,
            'readable': readable,
            'total_frames': total_frames,
            'dimensions': [height, width],
            'fps': fps,
            'frames_tested': frames_read,
            'frames_requested': frames_to_test,
            'error': error
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'readable': False,
            'error': str(e)
        }

def check_all_videos(data_folder):
    """
    Check integrity of all video files in the DATA folder.
    """
    
    print("Video Integrity Check")
    print("=" * 40)
    
    video_files = [f for f in os.listdir(data_folder) 
                   if f.endswith('.avi')]
    
    if not video_files:
        print("No AVI files found!")
        return
    
    video_files.sort()
    
    results = {}
    
    for video_file in video_files:
        video_path = os.path.join(data_folder, video_file)
        result = check_video_integrity(video_path)
        results[video_file] = result
        print()
    
    # Summary
    print("=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    good_videos = []
    partial_videos = []
    corrupt_videos = []
    
    for video_file, result in results.items():
        if result['status'] == 'good':
            good_videos.append(video_file)
        elif result['status'] == 'partial':
            partial_videos.append(video_file)
        else:
            corrupt_videos.append(video_file)
    
    print(f"✅ Good videos ({len(good_videos)}):")
    for video in good_videos:
        print(f"   • {video}")
    
    if partial_videos:
        print(f"\n⚠️  Partial videos ({len(partial_videos)}):")
        for video in partial_videos:
            print(f"   • {video} - {results[video]['error']}")
    
    if corrupt_videos:
        print(f"\n❌ Corrupt/Problem videos ({len(corrupt_videos)}):")
        for video in corrupt_videos:
            print(f"   • {video} - {results[video]['error']}")
    
    return results

if __name__ == "__main__":
    data_folder = "/Users/nick/Projects/klabCode/DATA"
    results = check_all_videos(data_folder)
    
    # Check for part files
    part_files = [f for f in os.listdir(data_folder) if f.endswith('.part')]
    if part_files:
        print(f"\n🗑️  Found {len(part_files)} .part file(s):")
        for part_file in part_files:
            print(f"   • {part_file}")
            # Check if corresponding .avi exists
            avi_name = part_file.replace('.part', '')
            if avi_name in results:
                status = results[avi_name]['status']
                print(f"     → Corresponding .avi: {avi_name} ({status})")
