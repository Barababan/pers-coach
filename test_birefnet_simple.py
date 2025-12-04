import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
import time

def test_birefnet():
    print("Initializing BiRefNet session...")
    start_time = time.time()
    try:
        session = new_session("birefnet-general")
        print(f"Session initialized in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Failed to initialize session: {e}")
        return

    # Create a dummy image or load one
    print("Creating dummy image...")
    img = Image.new('RGB', (500, 500), color = 'red')
    
    print("Running inference...")
    start_time = time.time()
    try:
        output = remove(img, session=session)
        print(f"Inference successful in {time.time() - start_time:.2f}s")
        output.save("test_birefnet_output.png")
        print("Saved test_birefnet_output.png")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    test_birefnet()
