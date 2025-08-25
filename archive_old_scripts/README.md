# Archive - Development History

This folder contains the evolution of the dF/F analysis pipeline development.

## Files

### `dFOverF.py`
- Original windowed processing implementation
- Included K-means clustering functionality
- Used for initial development and testing
- **Status**: Superseded by `analyze_organized.py`

### `test_*.py` Files
- Various test implementations during development
- `test_simple.py`: Basic dF/F calculation tests
- `test_windowed.py`: Windowed processing validation
- `test_dFOverF.py`: Main algorithm testing
- **Status**: Development artifacts, not used in production

### `check_video_integrity.py`
- Video file validation utility
- Checks for corruption or incomplete files
- **Status**: Utility script, may be useful for troubleshooting

## Current Production Code
The active pipeline uses these files in the root directory:
- `analyze_organized.py` - Main batch processing script
- `video_converter.py` - Comparison video creation
- `output_manager.py` - File organization and metadata

## Migration Notes
The transition from windowed processing to the current approach involved:
1. Simplifying from overlapping windows to single-pass processing
2. Adding spatial downsampling for memory efficiency  
3. Implementing organized output structure
4. Adding temporal smoothing and video comparisons
5. Focus on production-ready batch processing vs research flexibility
