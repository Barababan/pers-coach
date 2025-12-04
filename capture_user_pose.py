"""
Capture user movements using MediaPipe 3D Pose Estimation.
Extracts global world landmarks for comparison with trainer reference.
"""

import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, List, Dict

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Error: MediaPipe not installed. Install with: pip install mediapipe")
    exit(1)

from pose_utils import save_pose_sequence, calculate_key_joint_angles, map_mediapipe_to_sam3d


class UserPoseCapture:
    """Capture user pose using MediaPipe."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize MediaPipe Pose Landmarker.
        
        Args:
            model_path: Path to pose landmarker model. If None, will try to download.
        """
        # For MediaPipe Solutions API (simpler, but may not have world landmarks in all versions)
        # We'll use the legacy solution for better compatibility
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detector with world landmarks enabled
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0, 1, or 2. Higher = more accurate but slower
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("MediaPipe Pose initialized")
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process a single frame and extract pose landmarks.
        
        Args:
            frame: BGR image from camera
        
        Returns:
            Dictionary with pose data or None if no pose detected
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks or not results.pose_world_landmarks:
            return None
        
        # Extract image landmarks (normalized 0-1)
        image_landmarks = []
        for landmark in results.pose_landmarks.landmark:
            image_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        image_landmarks = np.array(image_landmarks)
        
        # Extract world landmarks (in meters, relative to hips)
        world_landmarks = []
        for landmark in results.pose_world_landmarks.landmark:
            world_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        world_landmarks = np.array(world_landmarks)
        
        # Calculate joint angles using world landmarks
        try:
            # Remove visibility channel for angle calculation
            world_coords_3d = world_landmarks[:, :3]
            angles = calculate_key_joint_angles(world_coords_3d)
        except Exception as e:
            print(f"Warning: Could not calculate joint angles: {e}")
            angles = {}
        
        return {
            'detected': True,
            'image_landmarks': image_landmarks,      # (33, 4) - x, y, z, visibility
            'world_landmarks': world_landmarks,      # (33, 4) - x, y, z (meters), visibility
            'joint_angles': angles
        }
    
    def draw_landmarks(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """
        Draw pose landmarks on frame.
        
        Args:
            frame: BGR image
            pose_data: Pose data from process_frame
        
        Returns:
            Frame with landmarks drawn
        """
        if not pose_data or not pose_data.get('detected'):
            return frame
        
        # Convert landmarks back to MediaPipe format for drawing
        # We need to reconstruct the landmark list
        from mediapipe.framework.formats import landmark_pb2
        
        pose_landmarks = landmark_pb2.NormalizedLandmarkList()
        image_lms = pose_data['image_landmarks']
        
        for i in range(len(image_lms)):
            landmark = pose_landmarks.landmark.add()
            landmark.x = image_lms[i][0]
            landmark.y = image_lms[i][1]
            landmark.z = image_lms[i][2]
            landmark.visibility = image_lms[i][3]
        
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        return frame
    
    def capture_from_webcam(
        self,
        duration: Optional[float] = None,
        output_path: Optional[str] = None,
        show_preview: bool = True,
        camera_id: int = 0
    ) -> List[Dict]:
        """
        Capture pose data from webcam.
        
        Args:
            duration: Duration in seconds (None = until 'q' pressed)
            output_path: Path to save pose sequence (.npz)
            show_preview: Whether to show live preview
            camera_id: Camera device ID
        
        Returns:
            List of pose data dictionaries
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Get camera properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default if not available
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nCamera info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        if duration:
            print(f"  Duration: {duration}s")
        else:
            print(f"  Press 'q' to stop recording")
        print()
        
        pose_sequence = []
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Check duration
                elapsed = time.time() - start_time
                if duration and elapsed >= duration:
                    break
                
                # Process frame
                pose_data = self.process_frame(frame)
                
                if pose_data:
                    pose_data['frame_idx'] = frame_count
                    pose_data['timestamp'] = elapsed
                    pose_sequence.append(pose_data)
                else:
                    # No pose detected
                    pose_sequence.append({
                        'frame_idx': frame_count,
                        'timestamp': elapsed,
                        'detected': False
                    })
                
                # Draw visualization
                if show_preview:
                    if pose_data and pose_data.get('detected'):
                        frame = self.draw_landmarks(frame, pose_data)
                        
                        # Add status text
                        status = f"Recording: {elapsed:.1f}s | Frames: {frame_count}"
                        cv2.putText(frame, status, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Add joint angles
                        if 'joint_angles' in pose_data:
                            y_offset = 60
                            for joint, angle in pose_data['joint_angles'].items():
                                text = f"{joint}: {angle:.1f}°"
                                cv2.putText(frame, text, (10, y_offset),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                y_offset += 25
                    else:
                        # No pose detected
                        cv2.putText(frame, "No pose detected", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow('User Pose Capture', frame)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break
                
                frame_count += 1
        
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Save if output path provided
        if output_path and len(pose_sequence) > 0:
            metadata = {
                'camera_id': camera_id,
                'fps': fps,
                'resolution': (width, height),
                'duration': time.time() - start_time,
                'total_frames': len(pose_sequence)
            }
            save_pose_sequence(output_path, pose_sequence, metadata)
        
        # Print summary
        detected_frames = sum(1 for p in pose_sequence if p.get('detected', False))
        print(f"\nCapture complete!")
        print(f"  Total frames: {len(pose_sequence)}")
        print(f"  Frames with pose: {detected_frames}")
        print(f"  Duration: {time.time() - start_time:.2f}s")
        if output_path:
            print(f"  Saved to: {output_path}")
        
        return pose_sequence
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    parser = argparse.ArgumentParser(
        description="Capture user pose using MediaPipe 3D Pose Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture for 10 seconds
  python capture_user_pose.py --duration 10 --output user_pose.npz
  
  # Capture until 'q' pressed
  python capture_user_pose.py --output user_pose.npz
  
  # Capture without preview (headless)
  python capture_user_pose.py --duration 5 --output user_pose.npz --no-preview
        """
    )
    
    parser.add_argument('--output', '-o', help='Output .npz file path')
    parser.add_argument('--duration', '-d', type=float, help='Duration in seconds (default: until q pressed)')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--no-preview', action='store_true', help='Disable live preview')
    
    args = parser.parse_args()
    
    # Initialize capture
    capture = UserPoseCapture()
    
    # Capture from webcam
    pose_sequence = capture.capture_from_webcam(
        duration=args.duration,
        output_path=args.output,
        show_preview=not args.no_preview,
        camera_id=args.camera
    )
    
    print(f"\nCaptured {len(pose_sequence)} frames")


if __name__ == '__main__':
    main()
