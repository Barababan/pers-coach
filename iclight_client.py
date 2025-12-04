#!/usr/bin/env python3
"""
IC-Light Cloud Client - облачный relighting через HuggingFace Spaces.

Использует IC-Light от lllyasviel для профессионального relighting.
Работает через Gradio API - не требует локального GPU.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
from PIL import Image
import tempfile

try:
    from gradio_client import Client, handle_file
except ImportError:
    print("Установите gradio_client: pip install gradio_client")
    sys.exit(1)


class ICLightClient:
    """Клиент для IC-Light через HuggingFace Spaces."""
    
    # Доступные пресеты освещения
    LIGHTING_PRESETS = {
        'left': 'Left Light',
        'right': 'Right Light',
        'top': 'Top Light', 
        'bottom': 'Bottom Light',
        'none': 'None (ambient)',
    }
    
    # Рекомендуемые промпты для фитнес-видео
    FITNESS_PROMPTS = {
        'studio': 'fitness trainer, professional studio lighting, soft light, clean background',
        'bright': 'fitness trainer, bright natural lighting, energetic atmosphere',
        'dramatic': 'fitness trainer, dramatic side lighting, professional photoshoot',
        'neon': 'fitness trainer, neon RGB lighting, cyberpunk gym atmosphere, colorful',
        'warm': 'fitness trainer, warm golden hour lighting, motivational atmosphere',
        'cool': 'fitness trainer, cool professional lighting, modern gym',
    }
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Инициализация клиента.
        
        Args:
            hf_token: HuggingFace token (опционально, для избежания rate limits)
        """
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.client = None
        self._connect()
    
    def _connect(self):
        """Подключаемся к IC-Light Space."""
        print("Connecting to IC-Light Space...")
        try:
            if self.hf_token:
                self.client = Client("lllyasviel/IC-Light", hf_token=self.hf_token)
            else:
                self.client = Client("lllyasviel/IC-Light")
            print("✅ Connected to IC-Light")
        except Exception as e:
            print(f"⚠️ Connection error: {e}")
            print("Trying alternative space...")
            # Можно попробовать другие копии space
            raise
    
    def view_api(self):
        """Показать доступные API endpoints."""
        if self.client:
            self.client.view_api()
    
    def relight_image(
        self,
        image_path: str,
        prompt: str = "professional studio lighting, soft light",
        lighting_preference: str = 'none',
        num_samples: int = 1,
        steps: int = 25,
        guidance_scale: float = 2.0,
        seed: int = 12345,
        lowres_denoise: float = 0.9,
        highres_denoise: float = 0.5,
        image_width: int = 512,
        image_height: int = 640,
        highres_scale: float = 1.5,
    ) -> List[Image.Image]:
        """
        Применяет relighting к изображению.
        
        Args:
            image_path: Путь к изображению (желательно с удалённым фоном)
            prompt: Текстовое описание желаемого освещения
            lighting_preference: 'left', 'right', 'top', 'bottom', 'none'
            num_samples: Количество вариантов (1-12)
            steps: Количество шагов диффузии (1-100)
            guidance_scale: CFG scale (1-32)
            seed: Random seed
            lowres_denoise: Сила обработки на низком разрешении (0.1-1.0)
            highres_denoise: Сила обработки на высоком разрешении (0.1-1.0)
            image_width: Ширина выходного изображения (256-1024)
            image_height: Высота выходного изображения (256-1024)
            highres_scale: Масштаб для высокого разрешения (1.0-3.0)
        
        Returns:
            Список PIL.Image с результатами
        """
        if not self.client:
            raise RuntimeError("Not connected to IC-Light")
        
        # Маппинг lighting preference
        lighting_map = {
            'none': 'None',
            'left': 'Left Light',
            'right': 'Right Light',
            'top': 'Top Light',
            'bottom': 'Bottom Light',
        }
        bg_source = lighting_map.get(lighting_preference, 'None')
        
        # Положительный и негативный промпты
        a_prompt = "best quality"
        n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
        
        print(f"🎨 Relighting image...")
        print(f"   Prompt: {prompt}")
        print(f"   Lighting: {bg_source}")
        print(f"   Size: {image_width}x{image_height}")
        
        start_time = time.time()
        
        try:
            # API: /process_relight
            # Parameters: input_fg, prompt, image_width, image_height, num_samples, 
            #             seed, steps, a_prompt, n_prompt, cfg, highres_scale, 
            #             highres_denoise, lowres_denoise, bg_source
            result = self.client.predict(
                handle_file(image_path),  # input_fg
                prompt,                    # prompt
                image_width,               # image_width
                image_height,              # image_height
                num_samples,               # num_samples
                seed,                      # seed
                steps,                     # steps
                a_prompt,                  # a_prompt
                n_prompt,                  # n_prompt
                guidance_scale,            # cfg
                highres_scale,             # highres_scale
                highres_denoise,           # highres_denoise
                lowres_denoise,            # lowres_denoise
                bg_source,                 # bg_source
                api_name="/process_relight"
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Done in {elapsed:.1f}s")
            
            # Result: (preprocessed_foreground, outputs)
            # outputs is List[Dict(image: filepath, caption: str | None)]
            if isinstance(result, tuple):
                preprocessed, outputs = result
                print(f"   Preprocessed: {preprocessed}")
                
                images = []
                if isinstance(outputs, list):
                    for item in outputs:
                        if isinstance(item, dict) and 'image' in item:
                            images.append(Image.open(item['image']))
                        elif isinstance(item, str):
                            images.append(Image.open(item))
                elif isinstance(outputs, str):
                    images.append(Image.open(outputs))
                
                return images
            else:
                # Fallback
                if isinstance(result, str):
                    return [Image.open(result)]
                return []
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def relight_with_preset(
        self,
        image_path: str,
        preset: str = 'studio',
        lighting: str = 'none',
    ) -> List[Image.Image]:
        """
        Применяет relighting с готовым пресетом.
        
        Args:
            image_path: Путь к изображению
            preset: Один из FITNESS_PROMPTS ключей
            lighting: Направление света
        """
        prompt = self.FITNESS_PROMPTS.get(preset, self.FITNESS_PROMPTS['studio'])
        return self.relight_image(image_path, prompt=prompt, lighting_preference=lighting)


def process_video_frames(
    video_path: str,
    output_dir: str,
    client: ICLightClient,
    prompt: str = "professional studio lighting",
    frame_skip: int = 1,
    max_frames: Optional[int] = None,
):
    """
    Обрабатывает видео покадрово.
    
    Args:
        video_path: Путь к видео
        output_dir: Директория для результатов
        client: ICLightClient instance
        prompt: Промпт для освещения
        frame_skip: Обрабатывать каждый N-й кадр
        max_frames: Максимум кадров для обработки
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {total_frames} frames at {fps} FPS")
    
    frame_idx = 0
    processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        if max_frames and processed >= max_frames:
            break
        
        # Сохраняем кадр во временный файл
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            tmp_path = tmp.name
        
        try:
            # Обрабатываем
            print(f"\nFrame {frame_idx}/{total_frames}")
            results = client.relight_image(tmp_path, prompt=prompt)
            
            if results:
                # Сохраняем результат
                output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
                results[0].save(output_path)
                print(f"  Saved: {output_path}")
        
        finally:
            os.unlink(tmp_path)
        
        frame_idx += 1
        processed += 1
    
    cap.release()
    print(f"\n✅ Processed {processed} frames")


def test_single_image(image_path: str, output_path: str = "iclight_result.png"):
    """Тестирует IC-Light на одном изображении."""
    client = ICLightClient()
    
    # Смотрим API
    print("\n📡 Available API:")
    client.view_api()
    
    # Тестируем
    print("\n🎨 Testing relighting...")
    try:
        results = client.relight_with_preset(image_path, preset='studio')
        if results:
            results[0].save(output_path)
            print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nНужно проверить API endpoints через client.view_api()")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="IC-Light Cloud Client")
    parser.add_argument('--image', '-i', help='Input image path')
    parser.add_argument('--output', '-o', default='iclight_result.png')
    parser.add_argument('--prompt', '-p', default='professional studio lighting, soft light')
    parser.add_argument('--preset', choices=list(ICLightClient.FITNESS_PROMPTS.keys()))
    parser.add_argument('--lighting', '-l', default='none',
                       choices=list(ICLightClient.LIGHTING_PRESETS.keys()))
    parser.add_argument('--view-api', action='store_true', help='Show API info')
    
    args = parser.parse_args()
    
    if args.view_api:
        client = ICLightClient()
        client.view_api()
    elif args.image:
        test_single_image(args.image, args.output)
    else:
        print("Usage:")
        print("  python iclight_client.py --view-api")
        print("  python iclight_client.py -i image.png -o result.png")
        print("  python iclight_client.py -i image.png --preset neon")
