"""
Real-time pose comparison system with 3-panel split-screen display.
Left: 3D pose overlay (coach + user)
Center: Coach video playback
Right: User webcam feed
"""

import argparse
import cv2
import numpy as np
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Tuple
from queue import Queue

try:
    import mediapipe as mp
except ImportError:
    print("Error: MediaPipe not installed. Install with: pip install mediapipe")
    exit(1)

from pose_utils import load_pose_sequence, calculate_pose_similarity
from pose_renderer_3d import SimplePose3DRenderer, Pose3DRenderer


class RealtimeComparison:
    """Real-time pose comparison with 3-panel display."""
    
    def __init__(self, coach_poses_path: str, coach_video_path: str,
                 camera_id: int = 0, use_3d_renderer: bool = False):
        """
        Initialize real-time comparison system.
        
        Args:
            coach_poses_path: Path to pre-processed coach poses (.npz)
            coach_video_path: Path to coach video
            camera_id: Webcam device ID
            use_3d_renderer: Use matplotlib 3D renderer (slower but better quality)
        """
        # Load coach data
        print(f"Loading coach poses from {coach_poses_path}...")
        self.coach_poses, self.coach_metadata = load_pose_sequence(coach_poses_path)
        
        # Load coach video
        print(f"Loading coach video from {coach_video_path}...")
        self.coach_video = cv2.VideoCapture(coach_video_path)
        if not self.coach_video.isOpened():
            raise RuntimeError(f"Could not open coach video: {coach_video_path}")
        
        self.coach_fps = self.coach_video.get(cv2.CAP_PROP_FPS)
        self.coach_total_frames = int(self.coach_video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize user webcam
        print(f"Opening webcam {camera_id}...")
        self.user_camera = cv2.VideoCapture(camera_id)
        if not self.user_camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balance between speed and accuracy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize 3D renderer
        if use_3d_renderer:
            print("Using Matplotlib 3D Renderer...")
            self.renderer = Pose3DRenderer(width=640, height=480)
        else:
            print("Using Simple 2D Projection Renderer...")
            self.renderer = SimplePose3DRenderer(width=640, height=480)
        
        # Panel dimensions
        self.panel_width = 640
        self.panel_height = 480
        
        # State
        self.current_coach_frame_idx = 0
        self.is_playing = True
        self.is_looping = True
        self.playback_speed = 1.0
        
        # Similarity tracking
        self.current_similarity = None
        
        # Gray background color
        self.bg_color = (128, 128, 128)
        
        print("Initialization complete!")
    
    def _get_coach_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get coach 3D pose for given frame index."""
        if frame_idx >= len(self.coach_poses):
            return None
        
        pose_data = self.coach_poses[frame_idx]
        if not pose_data.get('detected', False):
            return None
        
        return pose_data.get('keypoints_3d')
    
    def _get_coach_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get coach video frame."""
        self.coach_video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.coach_video.read()
        
        if not ret:
            return None
        
        # Resize to panel size
        frame = cv2.resize(frame, (self.panel_width, self.panel_height))
        return frame
    
    def _process_user_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process user webcam frame.
        
        Returns:
            Tuple of (display_frame, world_landmarks_3d)
        """
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(frame_rgb)
        
        # Extract world landmarks
        world_landmarks = None
        if results.pose_world_landmarks:
            world_landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.pose_world_landmarks.landmark
            ])
        
        # Draw skeleton on frame
        if results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return frame, world_landmarks
    
    def _create_gray_background(self, width: int, height: int) -> np.ndarray:
        """Create gray background."""
        return np.full((height, width, 3), self.bg_color, dtype=np.uint8)
    
    def _add_feedback_overlay(self, frame: np.ndarray, similarity: Optional[Dict] = None):
        """Add similarity score and feedback overlay."""
        if similarity is None:
            return
        
        # Overall score
        score = similarity.get('overall_score', 0)
        score_text = f"Similarity: {score:.1f}/100"
        
        # Color based on score
        if score >= 80:
            color = (0, 255, 0)  # Green
        elif score >= 60:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.putText(frame, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, color, 2)
        
        # Joint angle feedback (show worst 3)
        angle_diffs = similarity.get('angle_differences', {})
        if angle_diffs:
            sorted_joints = sorted(angle_diffs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            y_offset = 70
            for joint, diff in sorted_joints:
                feedback_text = f"{joint}: {diff:.1f}° off"
                feedback_color = (0, 255, 0) if diff < 15 else (0, 255, 255) if diff < 30 else (0, 0, 255)
                
                cv2.putText(frame, feedback_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
                y_offset += 30
    
    def _create_composite_frame(self, pose_3d_frame: np.ndarray,
                                coach_frame: np.ndarray,
                                user_frame: np.ndarray) -> np.ndarray:
        """Create 3-panel composite frame."""
        # Ensure all frames are same height
        target_height = self.panel_height
        
        pose_3d_frame = cv2.resize(pose_3d_frame, (self.panel_width, target_height))
        coach_frame = cv2.resize(coach_frame, (self.panel_width, target_height))
        user_frame = cv2.resize(user_frame, (self.panel_width, target_height))
        
        # Concatenate horizontally
        composite = np.hstack([pose_3d_frame, coach_frame, user_frame])
        
        # Add panel labels
        cv2.putText(composite, "3D Pose Overlay", (10, target_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(composite, "Coach Video", (self.panel_width + 10, target_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(composite, "Your Webcam", (2 * self.panel_width + 10, target_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add similarity feedback at bottom
        if self.current_similarity:
            self._add_feedback_overlay(composite, self.current_similarity)
        
        # Add controls help
        help_text = "Controls: SPACE=Play/Pause | R=Restart | Q=Quit | +/-=Speed"
        cv2.putText(composite, help_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return composite
    
    def run(self):
        """Run the real-time comparison system."""
        print("\nStarting real-time comparison...")
        print("Controls:")
        print("  SPACE - Play/Pause")
        print("  R - Restart")
        print("  Q - Quit")
        print("  +/- - Adjust playback speed")
        print()
        
        window_name = "Real-Time Pose Comparison"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # Get coach data
                coach_pose = self._get_coach_pose(self.current_coach_frame_idx)
                coach_frame = self._get_coach_frame(self.current_coach_frame_idx)
                
                # Check if we reached end of video OR end of processed poses
                video_ended = coach_frame is None
                poses_ended = self.current_coach_frame_idx >= len(self.coach_poses)
                
                if video_ended or poses_ended:
                    if self.is_looping:
                        self.current_coach_frame_idx = 0
                        continue
                    else:
                        print("Coach video/poses ended")
                        break
                
                # Get user frame
                ret, user_frame = self.user_camera.read()
                if not ret:
                    print("Failed to read from webcam")
                    break
                
                user_frame = cv2.resize(user_frame, (self.panel_width, self.panel_height))
                user_frame, user_pose = self._process_user_frame(user_frame)
                
                # Calculate similarity
                if coach_pose is not None and user_pose is not None:
                    try:
                        # Ensure same number of keypoints
                        min_kp = min(len(coach_pose), len(user_pose))
                        self.current_similarity = calculate_pose_similarity(
                            coach_pose[:min_kp],
                            user_pose[:min_kp]
                        )
                    except Exception as e:
                        print(f"Error calculating similarity: {e}")
                        self.current_similarity = None
                
                # Render 3D pose overlay
                pose_3d_frame = self.renderer.render(coach_pose, user_pose)
                
                # Create composite frame
                composite = self._create_composite_frame(pose_3d_frame, coach_frame, user_frame)
                
                # Display
                cv2.imshow(window_name, composite)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord(' '):
                    self.is_playing = not self.is_playing
                    print(f"{'Playing' if self.is_playing else 'Paused'}")
                elif key == ord('r'):
                    self.current_coach_frame_idx = 0
                    print("Restarted")
                elif key == ord('+') or key == ord('='):
                    self.playback_speed = min(2.0, self.playback_speed + 0.1)
                    print(f"Speed: {self.playback_speed:.1f}x")
                elif key == ord('-') or key == ord('_'):
                    self.playback_speed = max(0.1, self.playback_speed - 0.1)
                    print(f"Speed: {self.playback_speed:.1f}x")
                
                # Advance frame
                if self.is_playing:
                    self.current_coach_frame_idx += int(self.playback_speed)
                
                # Frame rate control
                time.sleep(1.0 / 30)  # Target 30 FPS
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        self.coach_video.release()
        self.user_camera.release()
        cv2.destroyAllWindows()
        self.pose.close()
        if hasattr(self.renderer, 'close'):
            self.renderer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time pose comparison with 3-panel display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python realtime_comparison.py --coach-poses squat_poses.npz --coach-video squat.mp4
  
  # Use 3D renderer (slower but better quality)
  python realtime_comparison.py --coach-poses squat_poses.npz --coach-video squat.mp4 --use-3d
  
  # Different camera
  python realtime_comparison.py --coach-poses squat_poses.npz --coach-video squat.mp4 --camera 1
        """
    )
    
    parser.add_argument('--coach-poses', required=True, help='Path to coach poses (.npz)')
    parser.add_argument('--coach-video', required=True, help='Path to coach video')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--use-3d', action='store_true', help='Use matplotlib 3D renderer')
    
    args = parser.parse_args()
    
    # Create and run comparison system
    system = RealtimeComparison(
        coach_poses_path=args.coach_poses,
        coach_video_path=args.coach_video,
        camera_id=args.camera,
        use_3d_renderer=args.use_3d
    )
    
    system.run()


if __name__ == '__main__':
    main()
