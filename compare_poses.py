"""
Compare user pose against trainer reference.
Provides similarity scoring and frame-by-frame feedback.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from pose_utils import (
    load_pose_sequence,
    normalize_pose,
    calculate_pose_similarity,
    calculate_key_joint_angles
)


def align_sequences_dtw(
    trainer_sequence: List[Dict],
    user_sequence: List[Dict],
    distance_metric: str = 'euclidean'
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Align two pose sequences using Dynamic Time Warping.
    Handles sequences of different lengths and speeds.
    
    Args:
        trainer_sequence: Trainer pose sequence
        user_sequence: User pose sequence
        distance_metric: Distance metric to use
    
    Returns:
        Tuple of (alignment_path, dtw_distance)
    """
    # Extract keypoints from sequences (only from frames with detected poses)
    trainer_keypoints = []
    trainer_indices = []
    for i, frame in enumerate(trainer_sequence):
        if frame.get('detected', False) and 'keypoints_3d' in frame:
            trainer_keypoints.append(frame['keypoints_3d'].flatten())
            trainer_indices.append(i)
    
    user_keypoints = []
    user_indices = []
    for i, frame in enumerate(user_sequence):
        if frame.get('detected', False) and 'world_landmarks' in frame:
            # Use world landmarks (remove visibility channel)
            world_lms = frame['world_landmarks'][:, :3]
            user_keypoints.append(world_lms.flatten())
            user_indices.append(i)
    
    if len(trainer_keypoints) == 0 or len(user_keypoints) == 0:
        print("Warning: No valid keypoints found in one or both sequences")
        return [], float('inf')
    
    trainer_keypoints = np.array(trainer_keypoints)
    user_keypoints = np.array(user_keypoints)
    
    # Perform DTW
    print(f"Aligning sequences: trainer={len(trainer_keypoints)} frames, user={len(user_keypoints)} frames")
    distance, path = fastdtw(trainer_keypoints, user_keypoints, dist=euclidean)
    
    # Map back to original indices
    aligned_path = [(trainer_indices[i], user_indices[j]) for i, j in path]
    
    return aligned_path, distance


def compare_frame_pair(
    trainer_frame: Dict,
    user_frame: Dict
) -> Optional[Dict]:
    """
    Compare a single trainer frame with a user frame.
    
    Args:
        trainer_frame: Trainer pose data
        user_frame: User pose data
    
    Returns:
        Comparison results or None if comparison not possible
    """
    if not trainer_frame.get('detected') or not user_frame.get('detected'):
        return None
    
    # Get keypoints
    trainer_kp = trainer_frame.get('keypoints_3d')
    user_world_lms = user_frame.get('world_landmarks')
    
    if trainer_kp is None or user_world_lms is None:
        return None
    
    # Remove visibility channel from user landmarks
    user_kp = user_world_lms[:, :3]
    
    # Ensure same number of keypoints (both should have 33)
    min_kp = min(len(trainer_kp), len(user_kp))
    trainer_kp = trainer_kp[:min_kp]
    user_kp = user_kp[:min_kp]
    
    # Calculate similarity
    similarity = calculate_pose_similarity(trainer_kp, user_kp)
    
    # Add joint angle comparison
    trainer_angles = trainer_frame.get('joint_angles', {})
    user_angles = user_frame.get('joint_angles', {})
    
    angle_comparison = {}
    for joint in trainer_angles.keys():
        if joint in user_angles:
            diff = abs(trainer_angles[joint] - user_angles[joint])
            angle_comparison[joint] = {
                'trainer': trainer_angles[joint],
                'user': user_angles[joint],
                'difference': diff,
                'match': 'good' if diff < 15 else 'fair' if diff < 30 else 'poor'
            }
    
    return {
        'similarity': similarity,
        'angle_comparison': angle_comparison,
        'trainer_frame_idx': trainer_frame.get('frame_idx'),
        'user_frame_idx': user_frame.get('frame_idx')
    }


def compare_sequences(
    trainer_path: str,
    user_path: str,
    use_dtw: bool = True,
    output_path: Optional[str] = None
) -> Dict:
    """
    Compare user pose sequence against trainer reference.
    
    Args:
        trainer_path: Path to trainer pose sequence (.npz)
        user_path: Path to user pose sequence (.npz)
        use_dtw: Whether to use DTW for temporal alignment
        output_path: Optional path to save comparison results
    
    Returns:
        Dictionary with comparison results
    """
    print(f"Loading trainer reference: {trainer_path}")
    trainer_sequence, trainer_meta = load_pose_sequence(trainer_path)
    
    print(f"Loading user sequence: {user_path}")
    user_sequence, user_meta = load_pose_sequence(user_path)
    
    results = {
        'trainer_path': trainer_path,
        'user_path': user_path,
        'trainer_metadata': trainer_meta,
        'user_metadata': user_meta,
        'frame_comparisons': [],
        'overall_metrics': {}
    }
    
    if use_dtw:
        print("\nPerforming temporal alignment with DTW...")
        alignment_path, dtw_distance = align_sequences_dtw(trainer_sequence, user_sequence)
        
        if len(alignment_path) == 0:
            print("Error: Could not align sequences")
            return results
        
        print(f"DTW distance: {dtw_distance:.2f}")
        print(f"Aligned {len(alignment_path)} frame pairs")
        
        # Compare aligned frames
        for trainer_idx, user_idx in alignment_path:
            comparison = compare_frame_pair(
                trainer_sequence[trainer_idx],
                user_sequence[user_idx]
            )
            if comparison:
                results['frame_comparisons'].append(comparison)
        
        results['dtw_distance'] = dtw_distance
        results['alignment_path'] = alignment_path
    else:
        # Simple frame-by-frame comparison (no temporal alignment)
        print("\nPerforming frame-by-frame comparison...")
        min_frames = min(len(trainer_sequence), len(user_sequence))
        
        for i in range(min_frames):
            comparison = compare_frame_pair(trainer_sequence[i], user_sequence[i])
            if comparison:
                results['frame_comparisons'].append(comparison)
    
    # Calculate overall metrics
    if len(results['frame_comparisons']) > 0:
        overall_scores = [c['similarity']['overall_score'] for c in results['frame_comparisons']]
        mpjpe_scores = [c['similarity']['mpjpe'] for c in results['frame_comparisons']]
        angle_diffs = [c['similarity']['mean_angle_difference'] for c in results['frame_comparisons']]
        
        results['overall_metrics'] = {
            'mean_similarity_score': np.mean(overall_scores),
            'std_similarity_score': np.std(overall_scores),
            'min_similarity_score': np.min(overall_scores),
            'max_similarity_score': np.max(overall_scores),
            'mean_mpjpe': np.mean(mpjpe_scores),
            'mean_angle_difference': np.mean(angle_diffs),
            'total_comparisons': len(results['frame_comparisons'])
        }
        
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"Total frame comparisons: {results['overall_metrics']['total_comparisons']}")
        print(f"\nOverall Similarity Score: {results['overall_metrics']['mean_similarity_score']:.1f}/100")
        print(f"  (std: {results['overall_metrics']['std_similarity_score']:.1f})")
        print(f"\nMean Position Error (MPJPE): {results['overall_metrics']['mean_mpjpe']:.4f}")
        print(f"Mean Angle Difference: {results['overall_metrics']['mean_angle_difference']:.1f}°")
        print("="*60)
        
        # Print per-joint angle analysis
        print("\nPer-Joint Angle Analysis:")
        print("-" * 60)
        
        # Aggregate angle differences per joint
        joint_stats = {}
        for comparison in results['frame_comparisons']:
            for joint, data in comparison['angle_comparison'].items():
                if joint not in joint_stats:
                    joint_stats[joint] = []
                joint_stats[joint].append(data['difference'])
        
        for joint, diffs in joint_stats.items():
            mean_diff = np.mean(diffs)
            match_quality = 'GOOD' if mean_diff < 15 else 'FAIR' if mean_diff < 30 else 'POOR'
            print(f"  {joint:20s}: {mean_diff:6.1f}° ± {np.std(diffs):5.1f}° [{match_quality}]")
        
        print("="*60)
    else:
        print("\nWarning: No valid frame comparisons could be made")
    
    # Save results if requested
    if output_path:
        np.savez_compressed(output_path, **results)
        print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare user pose against trainer reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare with DTW alignment
  python compare_poses.py --trainer squat_poses.npz --user user_squat.npz
  
  # Compare without DTW (frame-by-frame)
  python compare_poses.py --trainer squat_poses.npz --user user_squat.npz --no-dtw
  
  # Save comparison results
  python compare_poses.py --trainer squat_poses.npz --user user_squat.npz --output comparison_results.npz
        """
    )
    
    parser.add_argument('--trainer', '-t', required=True, help='Trainer pose sequence (.npz)')
    parser.add_argument('--user', '-u', required=True, help='User pose sequence (.npz)')
    parser.add_argument('--output', '-o', help='Output file for comparison results (.npz)')
    parser.add_argument('--no-dtw', action='store_true', help='Disable DTW temporal alignment')
    
    args = parser.parse_args()
    
    # Check if fastdtw is available
    if not args.no_dtw:
        try:
            import fastdtw
        except ImportError:
            print("Warning: fastdtw not installed. Install with: pip install fastdtw")
            print("Falling back to frame-by-frame comparison")
            args.no_dtw = True
    
    # Perform comparison
    results = compare_sequences(
        trainer_path=args.trainer,
        user_path=args.user,
        use_dtw=not args.no_dtw,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
