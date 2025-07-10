#!/usr/bin/env python3
"""
Script to remove '_16k' suffix from all audio files in the generated_audio directory.
Example: audio_0001_mute_16k.mp3 -> audio_0001_mute.mp3
"""

import os
from pathlib import Path

def remove_16k_suffix(audio_dir="data/generated_audio"):
    """
    Remove '_16k' suffix from all audio files in the specified directory.
    
    Args:
        audio_dir: Directory containing audio files
    """
    # Get the project root (go up one level from misc/)
    project_root = Path(__file__).parent.parent
    audio_path = project_root / audio_dir
    
    if not audio_path.exists():
        print(f"Error: Directory {audio_path} does not exist!")
        return
    
    print(f"Looking for files with '_16k' suffix in: {audio_path}")
    
    # Find all files with '_16k' in the name
    files_to_rename = []
    for file_path in audio_path.glob("*_16k.*"):
        files_to_rename.append(file_path)
    
    if not files_to_rename:
        print("No files found with '_16k' suffix!")
        return
    
    print(f"Found {len(files_to_rename)} files to rename")
    
    # Rename files
    renamed_count = 0
    for file_path in files_to_rename:
        # Create new filename by removing '_16k'
        new_name = file_path.name.replace('_16k', '')
        new_path = file_path.parent / new_name
        
        try:
            # Rename the file
            file_path.rename(new_path)
            print(f"Renamed: {file_path.name} -> {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"Error renaming {file_path.name}: {e}")
    
    print(f"\nDone! Successfully renamed {renamed_count} files")

if __name__ == "__main__":
    remove_16k_suffix()
