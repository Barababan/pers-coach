"""
Utility functions for pose estimation and comparison.
Handles coordinate transformations, landmark mapping, and similarity calculations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation


# MediaPipe Pose Landmarks (33 landmarks)
MEDIAPIPE_POSE_LANDMARKS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]


def normalize_pose_scale(keypoints_3d: np.ndarray, reference_bone: str = 'torso') -> Tuple[np.ndarray, float]:
    """
    Normalize pose by scaling based on a reference bone length.
    
    Args:
        keypoints_3d: Array of shape (N, 3) containing 3D keypoints
        reference_bone: Which bone to use for normalization ('torso', 'femur', etc.)
    
    Returns:
        Tuple of (normalized_keypoints, scale_factor)
    """
    # For now, use torso length (hip to shoulder midpoint)
    # This is a simplified version - adjust indices based on actual landmark format
    
    if keypoints_3d.shape[0] < 25:  # Basic validation
        raise ValueError(f"Expected at least 25 keypoints, got {keypoints_3d.shape[0]}")
    
    # Calculate torso length (approximate)
    # Assuming standard body landmark ordering
    left_shoulder = keypoints_3d[11]  # Approximate index
    right_shoulder = keypoints_3d[12]
    left_hip = keypoints_3d[23]
    right_hip = keypoints_3d[24]
    
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    
    torso_length = np.linalg.norm(shoulder_center - hip_center)
    
    if torso_length < 1e-6:
        raise ValueError("Torso length too small, invalid pose data")
    
    # Normalize so torso length = 1.0
    scale_factor = 1.0 / torso_length
    normalized_keypoints = keypoints_3d * scale_factor
    
    return normalized_keypoints, scale_factor


def center_pose(keypoints_3d: np.ndarray, center_point: str = 'pelvis') -> Tuple[np.ndarray, np.ndarray]:
    """
    Center the pose by translating it so a reference point is at origin.
    
    Args:
        keypoints_3d: Array of shape (N, 3) containing 3D keypoints
        center_point: Which point to use as center ('pelvis', 'root', etc.)
    
    Returns:
        Tuple of (centered_keypoints, translation_vector)
    """
    # Use hip center as pelvis
    left_hip = keypoints_3d[23]
    right_hip = keypoints_3d[24]
    pelvis = (left_hip + right_hip) / 2
    
    translation = -pelvis
    centered_keypoints = keypoints_3d + translation
    
    return centered_keypoints, translation


def normalize_pose(keypoints_3d: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Fully normalize a pose: center and scale.
    
    Args:
        keypoints_3d: Array of shape (N, 3) containing 3D keypoints
    
    Returns:
        Dictionary containing normalized keypoints and transformation parameters
    """
    # First center the pose
    centered, translation = center_pose(keypoints_3d)
    
    # Then scale it
    normalized, scale = normalize_pose_scale(centered)
    
    return {
        'normalized_keypoints': normalized,
        'translation': translation,
        'scale': scale,
        'original_keypoints': keypoints_3d
    }


def calculate_joint_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    """
    Calculate the angle at point_b formed by points a-b-c.
    
    Args:
        point_a: First point (3D coordinates)
        point_b: Middle point (vertex of angle)
        point_c: Third point (3D coordinates)
    
    Returns:
        Angle in degrees
    """
    # Vectors from b to a and b to c
    vec_ba = point_a - point_b
    vec_bc = point_c - point_b
    
    # Normalize vectors
    vec_ba_norm = vec_ba / (np.linalg.norm(vec_ba) + 1e-8)
    vec_bc_norm = vec_bc / (np.linalg.norm(vec_bc) + 1e-8)
    
    # Calculate angle using dot product
    cos_angle = np.clip(np.dot(vec_ba_norm, vec_bc_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_key_joint_angles(keypoints_3d: np.ndarray) -> Dict[str, float]:
    """
    Calculate angles at key joints (elbows, knees, hips, shoulders).
    
    Args:
        keypoints_3d: Array of shape (N, 3) containing 3D keypoints
    
    Returns:
        Dictionary of joint names to angles in degrees
    """
    angles = {}
    
    try:
        # Left elbow: shoulder-elbow-wrist
        angles['left_elbow'] = calculate_joint_angle(
            keypoints_3d[11],  # left shoulder
            keypoints_3d[13],  # left elbow
            keypoints_3d[15]   # left wrist
        )
        
        # Right elbow
        angles['right_elbow'] = calculate_joint_angle(
            keypoints_3d[12],  # right shoulder
            keypoints_3d[14],  # right elbow
            keypoints_3d[16]   # right wrist
        )
        
        # Left knee: hip-knee-ankle
        angles['left_knee'] = calculate_joint_angle(
            keypoints_3d[23],  # left hip
            keypoints_3d[25],  # left knee
            keypoints_3d[27]   # left ankle
        )
        
        # Right knee
        angles['right_knee'] = calculate_joint_angle(
            keypoints_3d[24],  # right hip
            keypoints_3d[26],  # right knee
            keypoints_3d[28]   # right ankle
        )
        
        # Left hip: shoulder-hip-knee
        angles['left_hip'] = calculate_joint_angle(
            keypoints_3d[11],  # left shoulder
            keypoints_3d[23],  # left hip
            keypoints_3d[25]   # left knee
        )
        
        # Right hip
        angles['right_hip'] = calculate_joint_angle(
            keypoints_3d[12],  # right shoulder
            keypoints_3d[24],  # right hip
            keypoints_3d[26]   # right knee
        )
        
    except IndexError as e:
        print(f"Warning: Could not calculate all joint angles: {e}")
    
    return angles


def calculate_pose_similarity(pose1_keypoints: np.ndarray, pose2_keypoints: np.ndarray) -> Dict[str, float]:
    """
    Calculate similarity between two poses.
    
    Args:
        pose1_keypoints: First pose keypoints (N, 3)
        pose2_keypoints: Second pose keypoints (N, 3)
    
    Returns:
        Dictionary with similarity metrics
    """
    # Normalize both poses
    norm1 = normalize_pose(pose1_keypoints)
    norm2 = normalize_pose(pose2_keypoints)
    
    # Calculate positional difference (MPJPE - Mean Per Joint Position Error)
    position_diff = np.linalg.norm(
        norm1['normalized_keypoints'] - norm2['normalized_keypoints'],
        axis=1
    )
    mpjpe = np.mean(position_diff)
    
    # Calculate joint angle differences
    angles1 = calculate_key_joint_angles(pose1_keypoints)
    angles2 = calculate_key_joint_angles(pose2_keypoints)
    
    angle_diffs = {}
    for joint in angles1.keys():
        if joint in angles2:
            angle_diffs[joint] = abs(angles1[joint] - angles2[joint])
    
    mean_angle_diff = np.mean(list(angle_diffs.values())) if angle_diffs else 0.0
    
    # Calculate overall similarity score (0-100)
    # Lower MPJPE and angle differences = higher similarity
    # This is a heuristic - adjust thresholds as needed
    position_score = max(0, 100 - (mpjpe * 100))  # Assuming normalized scale
    angle_score = max(0, 100 - mean_angle_diff)
    
    overall_score = (position_score + angle_score) / 2
    
    return {
        'overall_score': overall_score,
        'mpjpe': mpjpe,
        'mean_angle_difference': mean_angle_diff,
        'angle_differences': angle_diffs,
        'position_score': position_score,
        'angle_score': angle_score
    }


def map_mediapipe_to_sam3d(mediapipe_landmarks: np.ndarray) -> np.ndarray:
    """
    Map MediaPipe 33 landmarks to SAM 3D Body format.
    
    Note: This is a simplified mapping. SAM 3D Body uses MHR (Momentum Human Rig)
    which may have different landmark definitions. This mapping focuses on the
    common body keypoints.
    
    Args:
        mediapipe_landmarks: MediaPipe pose landmarks (33, 3) or (33, 4) with visibility
    
    Returns:
        Mapped keypoints compatible with SAM 3D Body comparison
    """
    # For now, return the first 33 landmarks as-is
    # This may need refinement based on actual MHR landmark ordering
    if mediapipe_landmarks.shape[1] == 4:
        # Remove visibility channel if present
        return mediapipe_landmarks[:, :3]
    return mediapipe_landmarks


def save_pose_sequence(filename: str, pose_data: List[Dict], metadata: Optional[Dict] = None):
    """
    Save a sequence of poses to a file.
    
    Args:
        filename: Output filename (.npz format)
        pose_data: List of dictionaries containing pose information per frame
        metadata: Optional metadata dictionary
    """
    # Prepare data for saving
    save_dict = {
        'num_frames': len(pose_data),
        'pose_data': pose_data,
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    np.savez_compressed(filename, **save_dict)
    print(f"Saved pose sequence to {filename}")


def load_pose_sequence(filename: str) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Load a pose sequence from a file.
    
    Args:
        filename: Input filename (.npz format)
    
    Returns:
        Tuple of (pose_data, metadata)
    """
    data = np.load(filename, allow_pickle=True)
    pose_data = data['pose_data'].tolist()
    metadata = data['metadata'].item() if 'metadata' in data else None
    
    print(f"Loaded {len(pose_data)} frames from {filename}")
    return pose_data, metadata
