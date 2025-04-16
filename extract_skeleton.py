"""
Skeleton Extraction Module

This module processes images to extract human pose landmarks using MediaPipe.
It provides functionality to:
- Extract skeletal landmarks from images
- Save landmarks as JSON files
- Generate visualization of detected skeletons

The module preserves directory structure when saving outputs:
- Original images: DATASET_PATH
- Extracted landmarks: SKELETON_SAVE_PATH
- Visualization previews: SKELETON_PREVIEW
"""

import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple
import mediapipe as mp
import json

# Configuration Constants
DATASET_PATH = os.path.join(os.getcwd(), 'extracted_frames')
SKELETON_SAVE_PATH = os.path.join(os.getcwd(), 'data-skeleton')
SKELETON_PREVIEW = os.path.join(os.getcwd(), 'skeleton-preview')

# Skeleton connection configuration
SKELETON_CONNECTIONS = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle'),
]

def get_image_paths() -> List[str]:
    """
    Retrieve all image file paths from the dataset directory.
    
    Searches recursively for .png and .jpg files in DATASET_PATH.
    
    Returns:
        List[str]: List of absolute paths to image files
    
    Raises:
        FileNotFoundError: If DATASET_PATH doesn't exist
    """
    image_paths = []
    extensions = ['**/*.png', '**/*.jpg']
    
    for ext in extensions:
        image_paths.extend(glob(os.path.join(DATASET_PATH, ext), recursive=True))
    
    return image_paths

def extract_skeleton(image_path: str) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Extract skeleton landmarks from an image using MediaPipe Pose.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Optional[Dict[str, Dict[str, float]]]: Dictionary of landmarks where:
            - Key: landmark name (e.g., 'nose', 'left_shoulder')
            - Value: Dictionary containing 'x', 'y', 'z' coordinates (normalized)
            Returns None if no pose is detected or on error
    
    Example:
        >>> landmarks = extract_skeleton('path/to/image.jpg')
        >>> if landmarks:
        >>>     nose_position = landmarks['nose']
        >>>     x, y, z = nose_position['x'], nose_position['y'], nose_position['z']
    """
    mp_pose = mp.solutions.pose
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize pose detection
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            # Process image
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract landmarks as dictionary
                landmarks = {}
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmark_name = mp_pose.PoseLandmark(idx).name.lower()
                    landmarks[landmark_name] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    }
                return landmarks
            
            return None
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def save_landmark(image_path: str, landmarks: Dict[str, Dict[str, float]]) -> None:
    """
    Save landmarks to a JSON file while preserving directory structure.
    
    Args:
        image_path (str): Original image path
        landmarks (dict): Landmark data to save
    
    Raises:
        IOError: If unable to create directory or write file
    """
    # Get relative path by removing DATASET_PATH
    rel_path = os.path.relpath(image_path, DATASET_PATH)
    
    # Create new path with same structure but in SKELETON_SAVE_PATH
    # Replace image extension with .json
    new_path = os.path.join(SKELETON_SAVE_PATH, 
                           os.path.splitext(rel_path)[0] + '.json')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    
    # Save landmarks as JSON
    with open(new_path, 'w') as f:
        json.dump(landmarks, f, indent=4)

def draw_skeleton(image_path: str, landmarks: Dict[str, Dict[str, float]]) -> None:
    """
    Draw skeleton on image and save visualization to preview directory.
    
    Draws:
        - Green lines (BGR: 0,255,0) connecting body parts
        - Red dots (BGR: 0,0,255) at landmark positions
    
    Args:
        image_path (str): Path to original image
        landmarks (dict): Landmark coordinates (normalized)
    
    Notes:
        - Preserves original image directory structure in SKELETON_PREVIEW
        - Skips silently if image cannot be loaded
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return
    
    height, width = image.shape[:2]
    
    # Draw connections
    for start_point, end_point in SKELETON_CONNECTIONS:
        if start_point in landmarks and end_point in landmarks:
            start = landmarks[start_point]
            end = landmarks[end_point]
            
            start_pos = (int(start['x'] * width), int(start['y'] * height))
            end_pos = (int(end['x'] * width), int(end['y'] * height))
            
            cv2.line(image, start_pos, end_pos, (0, 255, 0), 2)
    
    # Draw landmarks
    for landmark in landmarks.values():
        pos = (int(landmark['x'] * width), int(landmark['y'] * height))
        cv2.circle(image, pos, 3, (0, 0, 255), -1)
    
    # Save image with same structure as source
    rel_path = os.path.relpath(image_path, DATASET_PATH)
    new_path = os.path.join(SKELETON_PREVIEW, rel_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(new_path, image)

def main() -> None:
    """
    Main execution function.
    
    Process flow:
    1. Verify dataset existence
    2. Collect all image paths
    3. For each image:
        - Extract skeleton landmarks
        - Save landmarks as JSON
        - Generate and save visualization
    
    Raises:
        FileNotFoundError: If dataset directory is missing
        ValueError: If no images found in dataset
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError('Dataset not found. Please download the dataset from Kaggle.')
    
    image_paths = get_image_paths()
    
    if not image_paths:
        raise ValueError('No images found in the dataset directory.')
    
    print(f'Number of images found: {len(image_paths)}')
    
    # Process each image and extract skeleton
    for path in image_paths:
        landmarks = extract_skeleton(path)
        if landmarks:
            save_landmark(path, landmarks)
            draw_skeleton(path, landmarks)
            
        # break # remove/activate this line to process all images/stop after one image
    

if __name__ == '__main__':
    main()