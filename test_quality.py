import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image
import io

# Try to import rembg
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError as e:
    REMBG_AVAILABLE = False
    print(f"Warning: rembg not installed or import failed. Error: {e}")
    print("Run 'pip install rembg'")

# Initialize session with u2net_human_seg (specialized for humans)
if REMBG_AVAILABLE:
    try:
        rembg_session = new_session("u2net_human_seg")
    except Exception as e:
        print(f"Error creating rembg session: {e}")
        REMBG_AVAILABLE = False

def process_mediapipe(image, bg_color=(128, 128, 128)):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    height, width = image.shape[:2]
    bg_image = np.zeros((height, width, 3), dtype=np.uint8)
    bg_image[:] = bg_color

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image_rgb)
        
        mask = results.segmentation_mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask_3d = np.stack((mask,) * 3, axis=-1)
        
        output_image = image * mask_3d + bg_image * (1 - mask_3d)
        return output_image.astype(np.uint8)

def process_rembg(image, bg_color=(128, 128, 128)):
    if not REMBG_AVAILABLE:
        return image
        
    # Convert cv2 (BGR) to PIL (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Remove background using human segmentation model
    output_pil = remove(img_pil, session=rembg_session)
    
    # Create background
    bg = Image.new('RGBA', output_pil.size, (*bg_color, 255))
    
    # Composite
    result = Image.alpha_composite(bg, output_pil)
    
    # Convert back to cv2 (BGR)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

def generate_comparison(input_path, frames=[100, 300, 500]):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Segmentation Comparison</title>
        <style>
            body { background: #222; color: white; font-family: sans-serif; padding: 20px; }
            .row { display: flex; margin-bottom: 20px; gap: 10px; }
            .col { flex: 1; text-align: center; }
            img { max-width: 100%; border: 1px solid #555; }
            h3 { margin: 5px 0; }
        </style>
    </head>
    <body>
        <h1>Comparison: MediaPipe vs Rembg</h1>
    """

    for frame_num in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, image = cap.read()
        if not success:
            continue
            
        print(f"Processing frame {frame_num}...")
        
        # Original
        cv2.imwrite(f"frame_{frame_num}_orig.jpg", image)
        
        # MediaPipe
        mp_res = process_mediapipe(image)
        cv2.imwrite(f"frame_{frame_num}_mp.jpg", mp_res)
        
        # Rembg
        if REMBG_AVAILABLE:
            rembg_res = process_rembg(image)
            cv2.imwrite(f"frame_{frame_num}_rembg.jpg", rembg_res)
        
        html_content += f"""
        <div class="row">
            <div class="col">
                <h3>Original {frame_num}</h3>
                <img src="frame_{frame_num}_orig.jpg">
            </div>
            <div class="col">
                <h3>MediaPipe {frame_num}</h3>
                <img src="frame_{frame_num}_mp.jpg">
            </div>
            <div class="col">
                <h3>Rembg {frame_num}</h3>
                <img src="frame_{frame_num}_rembg.jpg">
            </div>
        </div>
        """

    html_content += "</body></html>"
    
    with open("preview.html", "w") as f:
        f.write(html_content)
        
    print("Done! Open preview.html to see results.")
    cap.release()

if __name__ == "__main__":
    generate_comparison("squat.mp4", frames=[100, 250, 400])
