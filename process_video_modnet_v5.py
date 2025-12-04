#!/usr/bin/env python3
"""
MODNet Video Processing v5 - with AI Quality Enhancement

Features:
- MODNet background removal
- GFPGAN face enhancement (optional)
- Real-ESRGAN upscaling (optional)
- Professional Lighting (3D LUT, studio light simulation)
- All v4 artifact removal features
"""

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

# Import v4 functions
from process_video_modnet_v4 import (
    sharpen_alpha_aggressive,
    hard_threshold_mask,
    apply_morphological_cleanup,
    remove_small_components,
    guided_filter_fast,
    remove_fringe,
    detect_and_clean_halo,
    remove_bright_artifacts,
    enhance_output_quality,
    temporal_smooth_with_outlier_rejection
)

# Import professional lighting
from professional_lighting import (
    LightingStyle,
    professional_lighting_pipeline,
    apply_fitness_lighting,
    apply_broadcast_lighting,
    apply_cinematic_lighting
)


# AI Enhancement classes
class GFPGANEnhancer:
    """GFPGAN Face Enhancement wrapper."""
    
    def __init__(self, model_path='gfpgan/weights/GFPGANv1.4.pth', upscale=1):
        self.enabled = False
        self.upscale = upscale
        
        try:
            from gfpgan import GFPGANer
            
            # Check if model exists, download if not
            if not os.path.exists(model_path):
                model_path = 'GFPGANv1.4.pth'  # Will download automatically
            
            self.enhancer = GFPGANer(
                model_path=model_path,
                upscale=upscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None  # We handle background separately
            )
            self.enabled = True
            print("✅ GFPGAN loaded successfully")
        except Exception as e:
            print(f"⚠️  GFPGAN not available: {e}")
            self.enhancer = None
    
    def enhance(self, img):
        """Enhance face in image. Input: BGR numpy array."""
        if not self.enabled or self.enhancer is None:
            return img
        
        try:
            # GFPGAN expects BGR
            _, _, restored = self.enhancer.enhance(
                img, 
                has_aligned=False, 
                only_center_face=False,
                paste_back=True
            )
            return restored
        except Exception as e:
            print(f"⚠️  GFPGAN enhance failed: {e}")
            return img


class RealESRGANEnhancer:
    """Real-ESRGAN upscaling wrapper."""
    
    def __init__(self, model_name='RealESRGAN_x2plus', scale=2, tile=400):
        self.enabled = False
        self.scale = scale
        self.tile = tile
        
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # Model configurations
            model_configs = {
                'RealESRGAN_x2plus': {
                    'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    'num_block': 23,
                    'scale': 2
                },
                'RealESRGAN_x4plus': {
                    'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                    'num_block': 23,
                    'scale': 4
                },
                'realesr-general-x4v3': {
                    'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
                    'num_block': 23,
                    'scale': 4
                }
            }
            
            if model_name not in model_configs:
                print(f"⚠️  Unknown model: {model_name}, using RealESRGAN_x2plus")
                model_name = 'RealESRGAN_x2plus'
            
            config = model_configs[model_name]
            self.scale = config['scale']
            
            # Build model
            model = RRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=config['num_block'], 
                num_grow_ch=32, 
                scale=self.scale
            )
            
            # Download model if needed
            model_path = f'weights/{model_name}.pth'
            if not os.path.exists(model_path):
                os.makedirs('weights', exist_ok=True)
                print(f"📥 Downloading {model_name}...")
                import urllib.request
                urllib.request.urlretrieve(config['url'], model_path)
            
            self.enhancer = RealESRGANer(
                scale=self.scale,
                model_path=model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=False  # Use FP32 for compatibility
            )
            self.enabled = True
            print(f"✅ Real-ESRGAN ({model_name}, {self.scale}x) loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Real-ESRGAN not available: {e}")
            self.enhancer = None
    
    def enhance(self, img):
        """Upscale image. Input: BGR numpy array."""
        if not self.enabled or self.enhancer is None:
            return img
        
        try:
            output, _ = self.enhancer.enhance(img, outscale=self.scale)
            return output
        except Exception as e:
            print(f"⚠️  Real-ESRGAN enhance failed: {e}")
            return img


def apply_neural_color_correction(image, mode='warm'):
    """
    Apply neural-inspired color correction (LUT-like effect).
    Uses color science principles rather than AI models.
    
    Modes:
    - 'warm': Warmer skin tones, more pleasing
    - 'cool': Cooler, more professional look
    - 'vivid': Enhanced colors
    - 'cinematic': Film-like color grading
    - 'natural': Natural enhancement
    """
    img = image.astype(np.float32)
    
    if mode == 'warm':
        # Warm color shift - boost red/yellow, reduce blue
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.05, 0, 255)  # Red
        img[:, :, 1] = np.clip(img[:, :, 1] * 1.02, 0, 255)  # Green
        img[:, :, 0] = np.clip(img[:, :, 0] * 0.95, 0, 255)  # Blue
        
    elif mode == 'cool':
        # Cool color shift - boost blue, reduce red
        img[:, :, 2] = np.clip(img[:, :, 2] * 0.95, 0, 255)  # Red
        img[:, :, 1] = np.clip(img[:, :, 1] * 1.0, 0, 255)   # Green
        img[:, :, 0] = np.clip(img[:, :, 0] * 1.05, 0, 255)  # Blue
        
    elif mode == 'vivid':
        # Convert to HSV, boost saturation
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)  # Saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)  # Value
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        
    elif mode == 'cinematic':
        # Cinematic look: lift shadows, compress highlights, teal-orange
        # Convert to LAB for better control
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Lift shadows (increase L in dark areas)
        l_channel = lab[:, :, 0]
        shadow_mask = l_channel < 80
        lab[:, :, 0] = np.where(shadow_mask, l_channel * 1.1 + 10, l_channel)
        
        # Teal in shadows, orange in highlights
        a_channel = lab[:, :, 1]  # Green-Red
        b_channel = lab[:, :, 2]  # Blue-Yellow
        
        highlight_mask = l_channel > 150
        lab[:, :, 1] = np.where(highlight_mask, a_channel + 5, a_channel - 3)
        lab[:, :, 2] = np.where(highlight_mask, b_channel + 8, b_channel - 5)
        
        lab = np.clip(lab, 0, 255)
        img = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        
    elif mode == 'natural':
        # Subtle natural enhancement
        # Slight contrast curve
        img = img / 255.0
        img = np.power(img, 0.95)  # Slight gamma lift
        img = img * 255.0
        
        # Very subtle warm shift
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.02, 0, 255)
        img[:, :, 0] = np.clip(img[:, :, 0] * 0.98, 0, 255)
    
    return np.clip(img, 0, 255).astype(np.uint8)


def process_video_modnet_v5(
    input_path, 
    output_path, 
    bg_color=(128, 128, 128),
    quality_mode='high',
    temporal_smooth=True,
    resolution_scale=1.0,
    defringe=True,
    enhance=True,
    # AI enhancement options
    use_gfpgan=False,
    use_realesrgan=False,
    realesrgan_model='RealESRGAN_x2plus',
    color_grade=None,  # 'warm', 'cool', 'vivid', 'cinematic', 'natural'
    # Professional lighting options
    lighting_style=None,  # 'fitness', 'broadcast', 'cinematic', 'warm_studio', 'cool_studio', 'daylight'
    normalize_light=False,
    enhance_skin=False,
    virtual_studio_light=False
):
    """
    Process video with MODNet v5 - with optional AI quality enhancement.
    
    Lighting styles:
    - 'fitness': Optimized for fitness videos (contrast + skin tones)
    - 'broadcast': TV-style neutral with slight warmth
    - 'cinematic': Film-like with lifted shadows and teal-orange split
    - 'warm_studio': Warm studio lighting
    - 'cool_studio': Cool modern look
    - 'daylight': Natural daylight simulation
    """
    print(f"Processing {input_path} -> {output_path}")
    print(f"Quality mode: {quality_mode}")
    print(f"AI Features: GFPGAN={use_gfpgan}, Real-ESRGAN={use_realesrgan}, Color={color_grade}")
    print(f"Lighting: style={lighting_style}, normalize={normalize_light}, skin={enhance_skin}, studio={virtual_studio_light}")
    
    # Quality presets
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
        erode_size, dilate_size = 2, 2
        enhance_params = {'sharpen_amount': 0.3, 'contrast_boost': 1.05, 'saturation_boost': 1.1, 'denoise': True}
    
    # Initialize AI enhancers if needed
    gfpgan_enhancer = None
    realesrgan_enhancer = None
    
    if use_gfpgan:
        gfpgan_enhancer = GFPGANEnhancer()
    
    if use_realesrgan:
        realesrgan_enhancer = RealESRGANEnhancer(model_name=realesrgan_model)
    
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
    
    # Calculate processing resolution
    base_size = int(512 * resolution_scale)
    if quality_mode == 'high':
        base_size = int(672 * resolution_scale)
    
    if width >= height:
        rh = base_size
        rw = int(width / height * base_size)
    else:
        rw = base_size
        rh = int(height / width * base_size)
    rh = rh - rh % 32
    rw = rw - rw % 32
    
    print(f"Processing size: {rw}x{rh}")
    
    # Output size (may be larger if using Real-ESRGAN)
    output_width = width
    output_height = height
    if use_realesrgan and realesrgan_enhancer and realesrgan_enhancer.enabled:
        output_width = width * realesrgan_enhancer.scale
        output_height = height * realesrgan_enhancer.scale
        print(f"Output size (with Real-ESRGAN): {output_width}x{output_height}")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
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
            
            # === V4/V5 PROCESSING PIPELINE ===
            
            # 1. Aggressive sigmoid sharpening
            matte_np = sharpen_alpha_aggressive(matte_np, threshold=0.5, steepness=sharpen_steepness)
            
            # 2. Hard threshold
            matte_np = hard_threshold_mask(matte_np, low_thresh, high_thresh)
            
            # 3. Morphological cleanup
            matte_np = apply_morphological_cleanup(matte_np, erode_size=erode_size, dilate_size=dilate_size)
            
            # 4. Remove small components
            matte_np = remove_small_components(matte_np, min_area_ratio=min_component_area)
            
            # 5. Guided filter
            if use_guided_filter:
                matte_np = guided_filter_fast(frame_resized, matte_np, radius=3, eps=0.001)
            
            # 5.5. Halo cleanup
            matte_np = detect_and_clean_halo(frame_resized, matte_np, bg_color, threshold=40)
            
            # 5.6. Remove bright artifacts
            frame_resized, matte_np = remove_bright_artifacts(
                frame_resized, matte_np, bg_color,
                edge_width=10, brightness_threshold=35, color_threshold=45
            )
            
            # 6. Temporal smoothing
            if temporal_smooth and len(mask_history) > 0:
                matte_np = temporal_smooth_with_outlier_rejection(
                    matte_np, mask_history, max_history=max_history, outlier_thresh=0.25
                )
            
            mask_history.append(matte_np.copy())
            if len(mask_history) > max_history:
                mask_history.pop(0)
            
            matte_np = np.clip(matte_np, 0, 1)
            
            # === COMPOSITING ===
            
            # 7. Defringe
            if defringe:
                frame_processed = remove_fringe(frame_resized, matte_np, fringe_width=3)
            else:
                frame_processed = frame_resized
            
            # 8. Quality enhancement
            if enhance:
                frame_processed = enhance_output_quality(frame_processed, matte_np, **enhance_params)
            
            # Composite
            matte_3d = np.stack([matte_np] * 3, axis=-1)
            bg = np.full(frame_resized.shape, bg_color, dtype=np.float32)
            result_np = matte_3d * frame_processed.astype(np.float32) + (1 - matte_3d) * bg
            result_np = np.clip(result_np, 0, 255).astype(np.uint8)
            
            # Resize to original
            result_np = cv2.resize(result_np, (width, height), cv2.INTER_LANCZOS4)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            
            # === AI ENHANCEMENT ===
            
            # GFPGAN face enhancement
            if use_gfpgan and gfpgan_enhancer and gfpgan_enhancer.enabled:
                result_bgr = gfpgan_enhancer.enhance(result_bgr)
            
            # Professional Lighting (3D LUT + studio simulation)
            if lighting_style or normalize_light or enhance_skin or virtual_studio_light:
                # Map string to LightingStyle enum
                style_map = {
                    'fitness': LightingStyle.FITNESS,
                    'broadcast': LightingStyle.BROADCAST,
                    'cinematic': LightingStyle.CINEMATIC,
                    'warm_studio': LightingStyle.WARM_STUDIO,
                    'cool_studio': LightingStyle.COOL_STUDIO,
                    'daylight': LightingStyle.DAYLIGHT,
                    'neutral': LightingStyle.NEUTRAL,
                }
                style = style_map.get(lighting_style, LightingStyle.NEUTRAL) if lighting_style else LightingStyle.NEUTRAL
                
                # Resize mask for lighting if using virtual studio
                mask_for_lighting = None
                if virtual_studio_light:
                    mask_resized = cv2.resize(matte_np, (result_bgr.shape[1], result_bgr.shape[0]))
                    mask_for_lighting = mask_resized
                
                result_bgr = professional_lighting_pipeline(
                    result_bgr,
                    mask=mask_for_lighting,
                    style=style,
                    normalize_light=normalize_light,
                    enhance_skin=enhance_skin,
                    virtual_studio=virtual_studio_light,
                    adaptive_exposure_enabled=lighting_style is not None
                )
            
            # Simple color grading (legacy, use lighting_style instead)
            if color_grade:
                result_bgr = apply_neural_color_correction(result_bgr, mode=color_grade)
            
            # Real-ESRGAN upscaling (apply last)
            if use_realesrgan and realesrgan_enhancer and realesrgan_enhancer.enabled:
                result_bgr = realesrgan_enhancer.enhance(result_bgr)
            
            out.write(result_bgr)
            
            # Update progress
            if frame_count % 10 == 0:
                avg_time = total_inference_time / frame_count
                remaining = total_frames - frame_count
                eta_seconds = remaining * avg_time * (2 if use_gfpgan else 1) * (3 if use_realesrgan else 1)
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
    parser = argparse.ArgumentParser(description='Process video with MODNet v5 (AI enhancement + Professional Lighting)')
    parser.add_argument('--input', '-i', type=str, default='squat.mp4')
    parser.add_argument('--output', '-o', type=str, default='squat_gray_modnet_v5.mp4')
    parser.add_argument('--bg-color', type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument('--quality', '-q', choices=['fast', 'balanced', 'high'], default='high')
    parser.add_argument('--no-temporal', action='store_true')
    parser.add_argument('--no-defringe', action='store_true')
    parser.add_argument('--resolution', type=float, default=1.0)
    
    # AI enhancement options
    parser.add_argument('--gfpgan', action='store_true', help='Enable GFPGAN face enhancement')
    parser.add_argument('--realesrgan', action='store_true', help='Enable Real-ESRGAN upscaling')
    parser.add_argument('--realesrgan-model', type=str, default='RealESRGAN_x2plus',
                        choices=['RealESRGAN_x2plus', 'RealESRGAN_x4plus', 'realesr-general-x4v3'])
    parser.add_argument('--color', type=str, choices=['warm', 'cool', 'vivid', 'cinematic', 'natural'],
                        help='Apply simple color grading (legacy)')
    
    # Professional Lighting options
    parser.add_argument('--lighting', '-l', type=str, 
                        choices=['fitness', 'broadcast', 'cinematic', 'warm_studio', 'cool_studio', 'daylight'],
                        help='Professional lighting style (3D LUT)')
    parser.add_argument('--normalize-light', action='store_true',
                        help='Normalize uneven lighting across frame')
    parser.add_argument('--enhance-skin', action='store_true',
                        help='Enhance skin tones for healthier look')
    parser.add_argument('--studio-light', action='store_true',
                        help='Apply virtual 3-point studio lighting')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    process_video_modnet_v5(
        args.input, 
        args.output, 
        tuple(args.bg_color),
        quality_mode=args.quality,
        temporal_smooth=not args.no_temporal,
        resolution_scale=args.resolution,
        defringe=not args.no_defringe,
        use_gfpgan=args.gfpgan,
        use_realesrgan=args.realesrgan,
        realesrgan_model=args.realesrgan_model,
        color_grade=args.color,
        # Professional lighting
        lighting_style=args.lighting,
        normalize_light=args.normalize_light,
        enhance_skin=args.enhance_skin,
        virtual_studio_light=args.studio_light
    )
