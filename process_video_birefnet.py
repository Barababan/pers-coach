import cv2
import numpy as np
import sys
import os
from PIL import Image
from rembg import remove, new_session
import time

def process_video_birefnet(input_path, output_path, bg_color=(128, 128, 128)):
    """
    Process video with BiRefNet background removal.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        bg_color: Background color as RGB tuple (default: gray 128,128,128)
    """
    print(f"Processing {input_path} -> {output_path}...")
    print("Using BiRefNet model (high quality, slower processing)")
    
    # Initialize BiRefNet session
    print("Initializing BiRefNet session...")
    start_time = time.time()
    try:
        session = new_session("birefnet-general")
        print(f"Session initialized in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Error initializing BiRefNet session: {e}")
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
    estimated_time = (total_frames * 44) / 60  # Based on 44s per frame from test
    print(f"Estimated processing time: ~{estimated_time:.1f} minutes")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_inference_time = 0
    start_processing = time.time()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Convert BGR (OpenCV) to RGB (PIL)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        
        # Remove background using BiRefNet
        frame_start = time.time()
        output_pil = remove(img_pil, session=session)
        inference_time = time.time() - frame_start
        total_inference_time += inference_time
        
        # Create background image
        bg = Image.new('RGBA', output_pil.size, (*bg_color, 255))
        
        # Composite foreground over background
        result_pil = Image.alpha_composite(bg, output_pil)
        
        # Convert back to BGR (OpenCV)
        result_np = np.array(result_pil)
        output_image = cv2.cvtColor(result_np, cv2.COLOR_RGBA2BGR)

        # Write frame
        out.write(output_image)
        
        # Progress update
        avg_time = total_inference_time / frame_count
        remaining_frames = total_frames - frame_count
        eta_seconds = remaining_frames * avg_time
        eta_minutes = eta_seconds / 60
        
        print(f"Frame {frame_count}/{total_frames} | "
              f"Inference: {inference_time:.2f}s | "
              f"Avg: {avg_time:.2f}s/frame | "
              f"ETA: {eta_minutes:.1f}min", end='\r')

    cap.release()
    out.release()
    
    total_time = time.time() - start_processing
    print(f"\n\nProcessing complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average inference time: {total_inference_time/frame_count:.2f}s per frame")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    # Default paths
    input_file = "squat.mp4"
    output_file = "squat_gray_birefnet.mp4"
    
    # Allow command line args
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please make sure 'squat.mp4' is in the current directory or provide a path.")
    else:
        process_video_birefnet(input_file, output_file)
