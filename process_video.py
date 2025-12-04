import cv2
import numpy as np
import sys
import os
from PIL import Image
from rembg import remove, new_session

def process_video(input_path, output_path):
    print(f"Processing {input_path} -> {output_path}...")
    
    # Initialize Rembg session with human segmentation model
    try:
        session = new_session("u2net_human_seg")
    except Exception as e:
        print(f"Error initializing Rembg session: {e}")
        return

    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create gray background
    bg_color = (128, 128, 128) # RGB
    
    frame_count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}/{total_frames}...", end='\r')

        # Convert BGR (OpenCV) to RGB (PIL)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        
        # Remove background using Rembg
        # This returns an RGBA image
        output_pil = remove(img_pil, session=session)
        
        # Create background image
        bg = Image.new('RGBA', output_pil.size, (*bg_color, 255))
        
        # Composite foreground over background
        result_pil = Image.alpha_composite(bg, output_pil)
        
        # Convert back to BGR (OpenCV)
        result_np = np.array(result_pil)
        output_image = cv2.cvtColor(result_np, cv2.COLOR_RGBA2BGR)

        # Write frame
        out.write(output_image)

    cap.release()
    out.release()
    print(f"\nDone! Saved to {output_path}")

if __name__ == "__main__":
    # Default paths
    input_file = "squat.mp4"
    output_file = "squat_gray.mp4"
    
    # Allow command line args
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please make sure 'squat.mp4' is in the current directory or provide a path.")
    else:
        process_video(input_file, output_file)
