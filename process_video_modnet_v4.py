import os
import sys
import cv2
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Add MODNet src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MODNet'))
from src.models.modnet import MODNet


def hard_threshold_mask(mask, low_thresh=0.1, high_thresh=0.9):
    """
    Apply hard thresholds to eliminate semi-transparent areas.
    Values below low_thresh become 0, above high_thresh become 1.
    Middle values are linearly scaled.
    """
    result = np.zeros_like(mask)
    
    # Below low threshold -> 0
    # Above high threshold -> 1
    # In between -> linear scale from 0 to 1
    mask_mid = (mask >= low_thresh) & (mask <= high_thresh)
    mask_high = mask > high_thresh
    
    result[mask_high] = 1.0
    result[mask_mid] = (mask[mask_mid] - low_thresh) / (high_thresh - low_thresh)
    
    return result


def remove_fringe(image, mask, fringe_width=3):
    """
    Remove color fringe/halo from edges by replacing edge pixels 
    with colors from inside the mask.
    """
    # Find edge region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fringe_width*2+1, fringe_width*2+1))
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Eroded mask (definitely inside)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    
    # Edge region = original mask - eroded
    edge_region = (mask_uint8 > 128) & (eroded < 128)
    
    if not np.any(edge_region):
        return image
    
    # For edge pixels, sample color from nearby inside pixels
    result = image.copy()
    
    # Use inpainting to fill edge region with colors from inside
    inpaint_mask = edge_region.astype(np.uint8) * 255
    result = cv2.inpaint(result, inpaint_mask, fringe_width + 2, cv2.INPAINT_TELEA)
    
    return result


def detect_and_clean_halo(image, mask, bg_color=(128, 128, 128), threshold=30):
    """
    Detect remaining halo - LIGHT version.
    Only targets obvious bright spots, doesn't aggressively erode.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find outer edge region (just outside the mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    outer_edge = (dilated > 128) & (mask_uint8 < 128)
    
    bg_array = np.array(bg_color, dtype=np.float32)
    result_mask = mask.copy()
    
    if np.any(outer_edge):
        # Only check brightness difference (most reliable for halo)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        brightness_diff = np.abs(gray - bg_color[0])
        
        # Only detect very obvious halo (high threshold)
        halo_pixels = outer_edge & (brightness_diff > threshold)
        
        if np.any(halo_pixels):
            # Light erosion only where halo detected
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded_local = cv2.erode(mask_uint8, kernel_erode, iterations=1)
            
            # Apply local erosion only in halo regions
            result_uint8 = mask_uint8.copy()
            result_uint8[halo_pixels] = eroded_local[halo_pixels]
            result_mask = result_uint8.astype(np.float32) / 255.0
    
    return result_mask


def remove_bright_artifacts(image, mask, bg_color=(128, 128, 128), 
                           edge_width=8, brightness_threshold=40, color_threshold=50):
    """
    Remove bright artifacts (like ear leftovers) near mask edges.
    Targets pixels that are:
    1. Near the mask edge (within edge_width pixels)
    2. Significantly brighter OR differently colored than the background
    3. But outside the main mask (semi-transparent or just outside)
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    h, w = mask.shape
    
    # Find edge region - both inside and outside the mask boundary
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width*2+1, edge_width*2+1))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width+1, edge_width+1))
    
    dilated = cv2.dilate(mask_uint8, kernel_dilate, iterations=1)
    eroded = cv2.erode(mask_uint8, kernel_erode, iterations=1)
    
    # Edge band = expanded area minus core
    edge_band = (dilated > 128) & (eroded < 128)
    
    if not np.any(edge_band):
        return image, mask
    
    # Calculate how much pixels differ from background
    bg_array = np.array(bg_color, dtype=np.float32)
    image_f = image.astype(np.float32)
    
    # Brightness difference
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    brightness_diff = np.abs(gray - bg_color[0])
    
    # Color difference (Euclidean distance in RGB)
    color_diff = np.sqrt(np.sum((image_f - bg_array) ** 2, axis=-1))
    
    # Find artifact pixels: in edge band, not fully inside mask, 
    # and significantly different from background
    artifact_candidates = edge_band & (
        (brightness_diff > brightness_threshold) | 
        (color_diff > color_threshold)
    )
    
    # Additional filter: only target pixels that are NOT inside the solid mask
    # This preserves body parts while removing floating artifacts
    solid_mask = mask_uint8 > 200  # Definitely inside
    artifact_pixels = artifact_candidates & ~solid_mask
    
    result_image = image.copy()
    result_mask = mask.copy()
    
    if np.any(artifact_pixels):
        # Replace artifact pixels with background color
        artifact_3d = np.stack([artifact_pixels] * 3, axis=-1)
        result_image = np.where(artifact_3d, 
                               np.array(bg_color, dtype=np.uint8), 
                               result_image)
        
        # Zero out mask in artifact areas
        result_mask[artifact_pixels] = 0.0
    
    return result_image, result_mask


def fix_edge_artifacts_inpaint(image, mask, bg_color=(128, 128, 128), 
                               edge_width=5, inpaint_radius=3):
    """
    Fix edge artifacts using inpainting from background direction.
    Creates smooth transition at mask edges.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find thin edge region just inside the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width, edge_width))
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    
    # Edge band inside mask
    inner_edge = (mask_uint8 > 128) & (eroded < 128)
    
    if not np.any(inner_edge):
        return image
    
    # Create inpaint mask
    inpaint_mask = inner_edge.astype(np.uint8) * 255
    
    # Inpaint edge pixels
    result = cv2.inpaint(image, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
    
    return result


def enhance_output_quality(image, mask, 
                           sharpen_amount=0.3,
                           contrast_boost=1.05,
                           saturation_boost=1.1,
                           denoise=True):
    """
    Enhance final output quality with color correction and sharpening.
    Only applies to the person (masked area), not background.
    """
    result = image.copy().astype(np.float32)
    
    # Convert to different color spaces for processing
    # 1. Sharpening using unsharp mask
    if sharpen_amount > 0:
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        sharpened = cv2.addWeighted(result, 1 + sharpen_amount, blurred, -sharpen_amount, 0)
        result = sharpened
    
    # 2. Contrast boost (apply to luminance)
    if contrast_boost != 1.0:
        # Convert to LAB
        result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Boost L channel contrast
        l_channel = lab[:, :, 0]
        l_mean = np.mean(l_channel)
        lab[:, :, 0] = np.clip((l_channel - l_mean) * contrast_boost + l_mean, 0, 255)
        
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    
    # 3. Saturation boost
    if saturation_boost != 1.0:
        result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    
    # 4. Light denoising (only on person area to preserve detail)
    if denoise:
        result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
        # Fast bilateral filter for edge-preserving denoise
        denoised = cv2.bilateralFilter(result_uint8, d=5, sigmaColor=20, sigmaSpace=20)
        
        # Apply only to masked area
        mask_3d = np.stack([mask] * 3, axis=-1)
        result = mask_3d * denoised.astype(np.float32) + (1 - mask_3d) * result
    
    return np.clip(result, 0, 255).astype(np.uint8)


def sharpen_alpha_aggressive(matte, threshold=0.5, steepness=15):
    """
    Aggressively sharpen alpha matte using steep sigmoid.
    Higher steepness = sharper edges, less semi-transparency.
    """
    # Sigmoid with adjustable steepness
    # steepness=15 is much sharper than edge_width=0.08 (which is ~12.5)
    sharpened = 1 / (1 + np.exp(-steepness * (matte - threshold)))
    return sharpened


def apply_morphological_cleanup(mask, erode_size=3, dilate_size=2):
    """
    Apply morphological operations to clean mask.
    Light erosion to remove halo, preserve body proportions.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Light erosion - just enough to remove edge artifacts
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    mask_uint8 = cv2.erode(mask_uint8, kernel_erode, iterations=1)
    
    # Same size dilation to restore
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    mask_uint8 = cv2.dilate(mask_uint8, kernel_dilate, iterations=1)
    
    # Close small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    
    return mask_uint8.astype(np.float32) / 255.0


def remove_small_components(mask, min_area_ratio=0.01):
    """Remove small disconnected components from mask."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    if num_labels <= 1:
        return mask
    
    # Calculate minimum area threshold
    total_pixels = mask.shape[0] * mask.shape[1]
    min_area = int(total_pixels * min_area_ratio)
    
    # Keep only large components
    result = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255
    
    return result.astype(np.float32) / 255.0


def temporal_smooth_with_outlier_rejection(current_mask, history, max_history=3, outlier_thresh=0.3):
    """
    Temporal smoothing with outlier rejection.
    If current frame differs too much from history, reduce its influence.
    """
    if len(history) == 0:
        return current_mask
    
    # Calculate average of history
    history_avg = np.mean(history, axis=0)
    
    # Calculate difference from history
    diff = np.abs(current_mask - history_avg)
    mean_diff = np.mean(diff)
    
    # If current frame is an outlier, weight it less
    if mean_diff > outlier_thresh:
        # Outlier detected - use more history, less current
        alpha = 0.3  # Only 30% of current frame
    else:
        # Normal frame - balanced smoothing
        alpha = 0.6  # 60% current, 40% history
    
    smoothed = alpha * current_mask + (1 - alpha) * history_avg
    return smoothed


def guided_filter_fast(image, alpha, radius=4, eps=0.01):
    """Fast guided filter implementation."""
    try:
        import cv2.ximgproc as ximgproc
        image_f = image.astype(np.float32) / 255.0
        alpha_f = alpha.astype(np.float32)
        return ximgproc.guidedFilter(image_f, alpha_f, radius, eps)
    except:
        # Fallback - just blur slightly
        return cv2.GaussianBlur(alpha, (3, 3), 0)


def process_video_modnet_v4(
    input_path, 
    output_path, 
    bg_color=(128, 128, 128),
    quality_mode='high',  # 'fast', 'balanced', 'high'
    temporal_smooth=True,
    resolution_scale=1.0,
    defringe=True,
    enhance=True  # Enable output enhancement
):
    """
    Process video with MODNet v4 - focus on eliminating background artifacts.
    
    Quality modes:
    - 'fast': Minimal processing, faster but may have artifacts
    - 'balanced': Good quality/speed tradeoff
    - 'high': Maximum quality, aggressive artifact removal
    """
    print(f"Processing {input_path} -> {output_path}")
    print(f"Quality mode: {quality_mode}, defringe: {defringe}, enhance: {enhance}")
    
    # Quality presets - сбалансированные настройки
    if quality_mode == 'fast':
        sharpen_steepness = 12
        low_thresh, high_thresh = 0.2, 0.8
        min_component_area = 0.005
        use_guided_filter = False
        erode_size, dilate_size = 2, 2
        enhance_params = {'sharpen_amount': 0.2, 'contrast_boost': 1.0, 'saturation_boost': 1.0, 'denoise': False}
    elif quality_mode == 'balanced':
        sharpen_steepness = 18
        low_thresh, high_thresh = 0.15, 0.85
        min_component_area = 0.01
        use_guided_filter = True
        erode_size, dilate_size = 2, 2
        enhance_params = {'sharpen_amount': 0.25, 'contrast_boost': 1.03, 'saturation_boost': 1.05, 'denoise': True}
    else:  # high
        sharpen_steepness = 22
        low_thresh, high_thresh = 0.15, 0.85
        min_component_area = 0.01
        use_guided_filter = True
        erode_size, dilate_size = 2, 2  # Мягкий erode чтобы не сужать тело
        enhance_params = {'sharpen_amount': 0.3, 'contrast_boost': 1.05, 'saturation_boost': 1.1, 'denoise': True}
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU")
    
    # Load MODNet
    print("\nLoading MODNet model...")
    pretrained_ckpt = 'MODNet/pretrained/modnet_webcam_portrait_matting.ckpt'
    
    if not os.path.exists(pretrained_ckpt):
        print(f"❌ Model not found: {pretrained_ckpt}")
        return
    
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    
    if device.type == 'cpu':
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
    else:
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
        modnet = modnet.to(device)
    
    modnet.eval()
    print("✅ Model loaded")
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    
    # Calculate processing resolution (higher for better quality)
    base_size = int(512 * resolution_scale)
    if quality_mode == 'high':
        base_size = int(672 * resolution_scale)  # Higher res for high quality
    
    if width >= height:
        rh = base_size
        rw = int(width / height * base_size)
    else:
        rw = base_size
        rh = int(height / width * base_size)
    rh = rh - rh % 32
    rw = rw - rw % 32
    
    print(f"Processing size: {rw}x{rh}")
    
    # Video writer with high quality codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Transform
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Temporal smoothing history
    mask_history = []
    max_history = 3
    
    # Process frames
    print("\nProcessing frames...")
    frame_count = 0
    total_inference_time = 0
    start_time = time.time()
    
    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Prepare frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (rw, rh), cv2.INTER_AREA)
            
            frame_pil = Image.fromarray(frame_resized)
            frame_tensor = torch_transforms(frame_pil)
            frame_tensor = frame_tensor[None, :, :, :]
            
            if device.type != 'cpu':
                frame_tensor = frame_tensor.to(device)
            
            # Inference
            inference_start = time.time()
            with torch.no_grad():
                _, _, matte_tensor = modnet(frame_tensor, True)
            total_inference_time += time.time() - inference_start
            
            # Get matte as numpy
            matte_np = matte_tensor[0, 0].data.cpu().numpy()
            
            # === V4 PROCESSING PIPELINE ===
            
            # 1. Aggressive sigmoid sharpening
            matte_np = sharpen_alpha_aggressive(matte_np, threshold=0.5, steepness=sharpen_steepness)
            
            # 2. Hard threshold to eliminate semi-transparency
            matte_np = hard_threshold_mask(matte_np, low_thresh, high_thresh)
            
            # 3. Morphological cleanup (erode to remove halo)
            matte_np = apply_morphological_cleanup(matte_np, erode_size=erode_size, dilate_size=dilate_size)
            
            # 4. Remove small disconnected components
            matte_np = remove_small_components(matte_np, min_area_ratio=min_component_area)
            
            # 5. Guided filter for edge refinement (optional)
            if use_guided_filter:
                matte_np = guided_filter_fast(frame_resized, matte_np, radius=3, eps=0.001)
            
            # 5.5 Detect and clean remaining halo
            matte_np = detect_and_clean_halo(frame_resized, matte_np, bg_color, threshold=40)
            
            # 5.6 Remove bright artifacts near mask edges (ear, etc.)
            frame_resized, matte_np = remove_bright_artifacts(
                frame_resized, matte_np, bg_color, 
                edge_width=10, brightness_threshold=35, color_threshold=45
            )
            
            # 6. Temporal smoothing with outlier rejection
            if temporal_smooth and len(mask_history) > 0:
                matte_np = temporal_smooth_with_outlier_rejection(
                    matte_np, 
                    mask_history, 
                    max_history=max_history,
                    outlier_thresh=0.25
                )
            
            # Update history
            mask_history.append(matte_np.copy())
            if len(mask_history) > max_history:
                mask_history.pop(0)
            
            # Final clip
            matte_np = np.clip(matte_np, 0, 1)
            
            # === COMPOSITING ===
            
            # 7. Remove color fringe from edges
            if defringe:
                frame_defringed = remove_fringe(frame_resized, matte_np, fringe_width=3)
            else:
                frame_defringed = frame_resized
            
            # 8. Enhance output quality (sharpening, color correction)
            if enhance:
                frame_enhanced = enhance_output_quality(
                    frame_defringed, matte_np, **enhance_params
                )
            else:
                frame_enhanced = frame_defringed
            
            # Expand to 3 channels
            matte_3d = np.stack([matte_np] * 3, axis=-1)
            
            # Apply background
            bg = np.full(frame_resized.shape, bg_color, dtype=np.float32)
            result_np = matte_3d * frame_enhanced.astype(np.float32) + (1 - matte_3d) * bg
            result_np = np.clip(result_np, 0, 255).astype(np.uint8)
            
            # Resize back to original
            result_np = cv2.resize(result_np, (width, height), cv2.INTER_LANCZOS4)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            out.write(result_bgr)
            
            # Update progress
            if frame_count % 10 == 0:
                avg_time = total_inference_time / frame_count
                remaining = total_frames - frame_count
                eta_seconds = remaining * avg_time
                pbar.set_postfix({
                    'avg': f'{avg_time:.2f}s/f',
                    'ETA': f'{eta_seconds/60:.1f}min'
                })
            
            pbar.update(1)
    
    cap.release()
    out.release()
    
    # Summary
    total_time = time.time() - start_time
    avg_inference = total_inference_time / frame_count
    
    print(f"\n✅ Processing complete!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Average inference: {avg_inference:.3f}s per frame")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video with MODNet v4 (artifact removal)')
    parser.add_argument('--input', '-i', type=str, default='squat.mp4')
    parser.add_argument('--output', '-o', type=str, default='squat_gray_modnet_v4.mp4')
    parser.add_argument('--bg-color', type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument('--quality', '-q', choices=['fast', 'balanced', 'high'], default='high',
                        help='Quality mode: fast, balanced, or high')
    parser.add_argument('--no-temporal', action='store_true', help='Disable temporal smoothing')
    parser.add_argument('--no-defringe', action='store_true', help='Disable color fringe removal')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Resolution scale (1.0=default, 1.5=higher quality)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    process_video_modnet_v4(
        args.input, 
        args.output, 
        tuple(args.bg_color),
        quality_mode=args.quality,
        temporal_smooth=not args.no_temporal,
        resolution_scale=args.resolution,
        defringe=not args.no_defringe
    )
