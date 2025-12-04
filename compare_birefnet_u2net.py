import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
import time

def process_frame_comparison(input_path, frame_nums=[100, 250, 400]):
    """
    Compare BiRefNet vs U2Net Human Seg on sample frames.
    
    Args:
        input_path: Path to input video
        frame_nums: List of frame numbers to test
    """
    print("Comparing segmentation models on sample frames...")
    print(f"Testing frames: {frame_nums}")
    
    # Initialize sessions
    print("\nInitializing BiRefNet session...")
    start = time.time()
    birefnet_session = new_session("birefnet-general")
    print(f"BiRefNet loaded in {time.time() - start:.2f}s")
    
    print("Initializing U2Net Human Seg session...")
    start = time.time()
    u2net_session = new_session("u2net_human_seg")
    print(f"U2Net loaded in {time.time() - start:.2f}s")
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return
    
    bg_color = (128, 128, 128)  # Gray background
    
    results = []
    
    for frame_num in frame_nums:
        print(f"\n--- Processing frame {frame_num} ---")
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, image = cap.read()
        if not success:
            print(f"Failed to read frame {frame_num}")
            continue
        
        # Save original
        cv2.imwrite(f"compare_frame_{frame_num}_original.jpg", image)
        
        # Convert to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        
        # Process with BiRefNet
        print("  Processing with BiRefNet...")
        start = time.time()
        birefnet_output = remove(img_pil, session=birefnet_session)
        birefnet_time = time.time() - start
        
        # Composite BiRefNet
        bg = Image.new('RGBA', birefnet_output.size, (*bg_color, 255))
        birefnet_result = Image.alpha_composite(bg, birefnet_output)
        birefnet_np = cv2.cvtColor(np.array(birefnet_result), cv2.COLOR_RGBA2BGR)
        cv2.imwrite(f"compare_frame_{frame_num}_birefnet.jpg", birefnet_np)
        print(f"  BiRefNet: {birefnet_time:.2f}s")
        
        # Process with U2Net
        print("  Processing with U2Net Human Seg...")
        start = time.time()
        u2net_output = remove(img_pil, session=u2net_session)
        u2net_time = time.time() - start
        
        # Composite U2Net
        bg = Image.new('RGBA', u2net_output.size, (*bg_color, 255))
        u2net_result = Image.alpha_composite(bg, u2net_output)
        u2net_np = cv2.cvtColor(np.array(u2net_result), cv2.COLOR_RGBA2BGR)
        cv2.imwrite(f"compare_frame_{frame_num}_u2net.jpg", u2net_np)
        print(f"  U2Net: {u2net_time:.2f}s")
        
        results.append({
            'frame': frame_num,
            'birefnet_time': birefnet_time,
            'u2net_time': u2net_time,
            'speedup': birefnet_time / u2net_time
        })
    
    cap.release()
    
    # Generate HTML comparison
    html = """<!DOCTYPE html>
<html>
<head>
    <title>BiRefNet vs U2Net Comparison</title>
    <style>
        body { 
            background: #1a1a1a; 
            color: #fff; 
            font-family: 'Segoe UI', sans-serif; 
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 { 
            text-align: center; 
            color: #00ff88;
            margin-bottom: 10px;
        }
        .summary {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .summary table {
            width: 100%;
            border-collapse: collapse;
        }
        .summary th, .summary td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #444;
        }
        .summary th {
            color: #00ff88;
        }
        .frame-section {
            margin-bottom: 40px;
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
        }
        .frame-section h2 {
            color: #00ff88;
            margin-top: 0;
        }
        .comparison {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }
        .image-box {
            text-align: center;
        }
        .image-box img {
            width: 100%;
            border-radius: 4px;
            border: 2px solid #444;
        }
        .image-box h3 {
            margin: 10px 0 5px 0;
            font-size: 16px;
        }
        .image-box .time {
            color: #888;
            font-size: 14px;
        }
        .faster {
            color: #00ff88;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>🎯 BiRefNet vs U2Net Human Seg - Quality Comparison</h1>
    
    <div class="summary">
        <h2>⏱️ Performance Summary</h2>
        <table>
            <tr>
                <th>Frame</th>
                <th>BiRefNet Time</th>
                <th>U2Net Time</th>
                <th>Speed Difference</th>
            </tr>
"""
    
    for r in results:
        html += f"""
            <tr>
                <td>Frame {r['frame']}</td>
                <td>{r['birefnet_time']:.2f}s</td>
                <td class="faster">{r['u2net_time']:.2f}s</td>
                <td>U2Net is {r['speedup']:.1f}x faster</td>
            </tr>
"""
    
    html += """
        </table>
    </div>
"""
    
    for frame_num in frame_nums:
        html += f"""
    <div class="frame-section">
        <h2>Frame {frame_num}</h2>
        <div class="comparison">
            <div class="image-box">
                <h3>Original</h3>
                <img src="compare_frame_{frame_num}_original.jpg">
            </div>
            <div class="image-box">
                <h3>BiRefNet</h3>
                <img src="compare_frame_{frame_num}_birefnet.jpg">
                <div class="time">High quality, slower</div>
            </div>
            <div class="image-box">
                <h3>U2Net Human Seg</h3>
                <img src="compare_frame_{frame_num}_u2net.jpg">
                <div class="time">Good quality, faster</div>
            </div>
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open("comparison_birefnet_vs_u2net.html", "w") as f:
        f.write(html)
    
    print("\n" + "="*60)
    print("✅ Comparison complete!")
    print("="*60)
    print("\nPerformance Summary:")
    for r in results:
        print(f"  Frame {r['frame']}: BiRefNet={r['birefnet_time']:.2f}s, U2Net={r['u2net_time']:.2f}s (U2Net {r['speedup']:.1f}x faster)")
    
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"\n  Average: U2Net is {avg_speedup:.1f}x faster than BiRefNet")
    print("\n📊 Open 'comparison_birefnet_vs_u2net.html' to compare quality")

if __name__ == "__main__":
    process_frame_comparison("squat.mp4", frame_nums=[100, 250, 400])
