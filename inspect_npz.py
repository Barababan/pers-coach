import numpy as np
import sys

try:
    data = np.load('squat_poses.npz', allow_pickle=True)
    print("Keys:", data.files)
    if 'pose_data' in data:
        print("Pose data shape:", data['pose_data'].shape)
        # Check first frame data
        first_frame = data['pose_data'][0]
        if isinstance(first_frame, dict):
            print("Keys in first frame:", first_frame.keys())
            if 'pred_keypoints_3d' in first_frame:
                kp = first_frame['pred_keypoints_3d']
                print("Keypoints 3D shape:", kp.shape)
                print("Sample keypoints (Head/Feet):", kp[0], kp[-1])
                print("Range X:", kp[:, 0].min(), kp[:, 0].max())
                print("Range Y:", kp[:, 1].min(), kp[:, 1].max())
            if 'error' in first_frame:
                print("Error in first frame:", first_frame['error'])
    if 'metadata' in data:
        print("Metadata:", data['metadata'])
except Exception as e:
    print("Error loading file:", e)
