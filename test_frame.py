import cv2
import mediapipe as mp
import numpy as np
import sys
import os

def process_frame(input_path, output_path, frame_num=300):
    print(f"Extracting and processing frame {frame_num} from {input_path}...")
    
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, image = cap.read()
    
    if not success:
        print(f"Error: Could not read frame {frame_num}")
        return

    height, width = image.shape[:2]
    
    # Create gray background
    bg_image = np.zeros((height, width, 3), dtype=np.uint8)
    bg_image[:] = (128, 128, 128)

    # Process with MediaPipe
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image_rgb)
        
        mask = results.segmentation_mask
        
        # Apply slight blur to mask for better blending
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        mask_3d = np.stack((mask,) * 3, axis=-1)
        
        # Alpha blending
        output_image = image * mask_3d + bg_image * (1 - mask_3d)
        output_image = output_image.astype(np.uint8)
        
        # Save result
        cv2.imwrite(output_path, output_image)
        print(f"Saved processed frame to {output_path}")

if __name__ == "__main__":
    input_file = "squat.mp4"
    output_file = "preview.jpg"
    frame_to_test = 300 # A frame where the trainer is likely in the middle of the squat
    
    if len(sys.argv) > 1:
        frame_to_test = int(sys.argv[1])
        
    if os.path.exists(input_file):
        process_frame(input_file, output_file, frame_to_test)
    else:
        print(f"File {input_file} not found.")
