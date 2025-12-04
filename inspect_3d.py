import json

with open('output/poses/trainer_poses.json') as f:
    data = json.load(f)

for frame_idx in [60, 100, 200]:
    print(f'=== Frame {frame_idx} ===')
    p = data['poses'][frame_idx]
    j = p['joints_3d']
    
    print(f'L_shoulder (5): x={j[5][0]:.3f}, y={j[5][1]:.3f}, z={j[5][2]:.3f}')
    print(f'R_shoulder (6): x={j[6][0]:.3f}, y={j[6][1]:.3f}, z={j[6][2]:.3f}')
    print(f'L_hip (9):      x={j[9][0]:.3f}, y={j[9][1]:.3f}, z={j[9][2]:.3f}')
    print(f'R_hip (10):     x={j[10][0]:.3f}, y={j[10][1]:.3f}, z={j[10][2]:.3f}')
    print(f'L_ankle (13):   x={j[13][0]:.3f}, y={j[13][1]:.3f}, z={j[13][2]:.3f}')
    print(f'R_ankle (14):   x={j[14][0]:.3f}, y={j[14][1]:.3f}, z={j[14][2]:.3f}')
    print(f'L_wrist (62):   x={j[62][0]:.3f}, y={j[62][1]:.3f}, z={j[62][2]:.3f}')
    print(f'R_wrist (41):   x={j[41][0]:.3f}, y={j[41][1]:.3f}, z={j[41][2]:.3f}')
    
    shoulder_y = (j[5][1] + j[6][1]) / 2
    hip_y = (j[9][1] + j[10][1]) / 2
    ankle_y = (j[13][1] + j[14][1]) / 2
    print(f'Avg Y: shoulder={shoulder_y:.3f}, hip={hip_y:.3f}, ankle={ankle_y:.3f}')
    print(f'L_shoulder X > R_shoulder X? {j[5][0] > j[6][0]} (means L is to the right of R in image)')
    print()
