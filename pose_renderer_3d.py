"""
3D Pose Renderer for real-time visualization.
Renders coach and user 3D skeletons overlaid on same canvas.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg


# MediaPipe/MHR pose connections (skeleton structure)
POSE_CONNECTIONS = [
    # Torso
    (11, 12),  # shoulders
    (11, 23),  # left shoulder to left hip
    (12, 24),  # right shoulder to right hip
    (23, 24),  # hips
    
    # Left arm
    (11, 13),  # left shoulder to left elbow
    (13, 15),  # left elbow to left wrist
    
    # Right arm
    (12, 14),  # right shoulder to right elbow
    (14, 16),  # right elbow to right wrist
    
    # Left leg
    (23, 25),  # left hip to left knee
    (25, 27),  # left knee to left ankle
    
    # Right leg
    (24, 26),  # right hip to right knee
    (26, 28),  # right knee to right ankle
    
    # Head/neck (if available)
    (0, 11),   # nose to left shoulder
    (0, 12),   # nose to right shoulder
]


class Pose3DRenderer:
    """Render 3D pose skeletons with matplotlib."""
    
    def __init__(self, width: int = 640, height: int = 480, dpi: int = 100):
        """
        Initialize 3D pose renderer.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            dpi: DPI for rendering
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        
        # Colors
        self.coach_color = (0.2, 0.4, 0.8)  # Blue
        self.user_color = (0.2, 0.8, 0.4)   # Green
        self.bg_color = (0.5, 0.5, 0.5)     # Gray
        
        # Create figure
        self.fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, facecolor=self.bg_color)
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor=self.bg_color)
        
        # Set up canvas
        self.canvas = FigureCanvasAgg(self.fig)
        
        # Configure 3D plot
        self._setup_plot()
    
    def _setup_plot(self):
        """Configure 3D plot appearance."""
        # Set viewing angle
        self.ax.view_init(elev=10, azim=45)
        
        # Set axis limits (will be updated based on data)
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        
        # Labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Grid
        self.ax.grid(True, alpha=0.3)
        
        # Background
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
    
    def _draw_skeleton(self, keypoints_3d: np.ndarray, color: Tuple[float, float, float], 
                       label: str, alpha: float = 0.8):
        """
        Draw a 3D skeleton.
        
        Args:
            keypoints_3d: Array of shape (N, 3) with 3D keypoints
            color: RGB color tuple (0-1 range)
            label: Label for legend
            alpha: Transparency
        """
        if keypoints_3d is None or len(keypoints_3d) == 0:
            return
        
        # Draw joints
        self.ax.scatter(
            keypoints_3d[:, 0],
            keypoints_3d[:, 1],
            keypoints_3d[:, 2],
            c=[color],
            s=50,
            alpha=alpha,
            label=f'{label} joints'
        )
        
        # Draw bones (connections)
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(keypoints_3d) and end_idx < len(keypoints_3d):
                start = keypoints_3d[start_idx]
                end = keypoints_3d[end_idx]
                
                self.ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    c=color,
                    linewidth=2,
                    alpha=alpha
                )
    
    def render(self, coach_keypoints: Optional[np.ndarray] = None,
               user_keypoints: Optional[np.ndarray] = None,
               auto_scale: bool = True) -> np.ndarray:
        """
        Render coach and user skeletons.
        
        Args:
            coach_keypoints: Coach 3D keypoints (N, 3)
            user_keypoints: User 3D keypoints (N, 3)
            auto_scale: Automatically scale axes to fit data
        
        Returns:
            Rendered image as numpy array (H, W, 3) in BGR format
        """
        # Clear previous frame
        self.ax.cla()
        self._setup_plot()
        
        # Draw skeletons
        if coach_keypoints is not None:
            self._draw_skeleton(coach_keypoints, self.coach_color, 'Coach')
        
        if user_keypoints is not None:
            self._draw_skeleton(user_keypoints, self.user_color, 'User', alpha=0.7)
        
        # Auto-scale axes
        if auto_scale:
            all_points = []
            if coach_keypoints is not None:
                all_points.append(coach_keypoints)
            if user_keypoints is not None:
                all_points.append(user_keypoints)
            
            if all_points:
                all_points = np.vstack(all_points)
                margin = 0.2
                
                x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
                y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
                z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
                
                x_range = x_max - x_min
                y_range = y_max - y_min
                z_range = z_max - z_min
                
                self.ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
                self.ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
                self.ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)
        
        # Add legend
        if coach_keypoints is not None or user_keypoints is not None:
            self.ax.legend(loc='upper right')
        
        # Render to canvas
        self.canvas.draw()
        
        # Convert to numpy array
        buf = self.canvas.buffer_rgba()
        image = np.asarray(buf)
        
        # Convert RGBA to BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        return image_bgr
    
    def close(self):
        """Clean up resources."""
        plt.close(self.fig)


class SimplePose3DRenderer:
    """Simpler 3D renderer using OpenCV 2D projection."""
    
    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize simple 3D renderer.
        
        Args:
            width: Canvas width
            height: Canvas height
        """
        self.width = width
        self.height = height
        
        # Colors (BGR format for OpenCV)
        self.coach_color = (200, 100, 50)   # Blue
        self.user_color = (50, 200, 100)    # Green
        self.bg_color = (128, 128, 128)     # Gray
        
        # Camera parameters for projection
        self.focal_length = 500
        self.camera_center = (width // 2, height // 2)
    
    def _project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D using simple perspective projection.
        
        Args:
            points_3d: Array of shape (N, 3)
        
        Returns:
            Array of shape (N, 2) with 2D coordinates
        """
        if points_3d is None or len(points_3d) == 0:
            return np.array([])
        
        # Simple perspective projection
        # Assume Z is depth (away from camera)
        points_2d = np.zeros((len(points_3d), 2))
        
        for i, (x, y, z) in enumerate(points_3d):
            # Add offset to avoid division by zero
            z_proj = z + 2.0
            
            # Project
            x_2d = (x * self.focal_length / z_proj) + self.camera_center[0]
            y_2d = (y * self.focal_length / z_proj) + self.camera_center[1]
            
            points_2d[i] = [x_2d, y_2d]
        
        return points_2d.astype(int)
    
    def _draw_skeleton_2d(self, canvas: np.ndarray, points_2d: np.ndarray, 
                          color: Tuple[int, int, int], thickness: int = 2):
        """Draw skeleton on 2D canvas."""
        if points_2d is None or len(points_2d) == 0:
            return
        
        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(points_2d) and end_idx < len(points_2d):
                start = tuple(points_2d[start_idx])
                end = tuple(points_2d[end_idx])
                
                # Check if points are within canvas
                if (0 <= start[0] < self.width and 0 <= start[1] < self.height and
                    0 <= end[0] < self.width and 0 <= end[1] < self.height):
                    cv2.line(canvas, start, end, color, thickness)
        
        # Draw joints
        for point in points_2d:
            if 0 <= point[0] < self.width and 0 <= point[1] < self.height:
                cv2.circle(canvas, tuple(point), 5, color, -1)
    
    def render(self, coach_keypoints: Optional[np.ndarray] = None,
               user_keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render coach and user skeletons.
        
        Args:
            coach_keypoints: Coach 3D keypoints (N, 3)
            user_keypoints: User 3D keypoints (N, 3)
        
        Returns:
            Rendered image as numpy array (H, W, 3) in BGR format
        """
        # Create gray background
        canvas = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        
        # Project and draw coach skeleton
        if coach_keypoints is not None:
            coach_2d = self._project_3d_to_2d(coach_keypoints)
            self._draw_skeleton_2d(canvas, coach_2d, self.coach_color, thickness=3)
        
        # Project and draw user skeleton (thinner, on top)
        if user_keypoints is not None:
            user_2d = self._project_3d_to_2d(user_keypoints)
            self._draw_skeleton_2d(canvas, user_2d, self.user_color, thickness=2)
        
        # Add labels
        if coach_keypoints is not None:
            cv2.putText(canvas, "Coach", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, self.coach_color, 2)
        if user_keypoints is not None:
            cv2.putText(canvas, "User", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, self.user_color, 2)
        
        return canvas


# Test function
if __name__ == '__main__':
    # Test with sample data
    print("Testing 3D Pose Renderer...")
    
    # Create sample skeletons (simplified)
    coach_pose = np.random.randn(33, 3) * 0.3
    user_pose = coach_pose + np.random.randn(33, 3) * 0.1  # Similar but slightly different
    
    # Test matplotlib renderer
    print("Testing Matplotlib 3D Renderer...")
    renderer_3d = Pose3DRenderer(width=640, height=480)
    frame = renderer_3d.render(coach_pose, user_pose)
    cv2.imwrite('test_3d_render.png', frame)
    print(f"Saved test_3d_render.png ({frame.shape})")
    renderer_3d.close()
    
    # Test simple renderer
    print("Testing Simple 2D Projection Renderer...")
    renderer_simple = SimplePose3DRenderer(width=640, height=480)
    frame = renderer_simple.render(coach_pose, user_pose)
    cv2.imwrite('test_simple_render.png', frame)
    print(f"Saved test_simple_render.png ({frame.shape})")
    
    print("Test complete!")
