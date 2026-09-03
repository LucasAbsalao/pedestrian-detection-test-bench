'''
To add a new distortion you should modificate this three parts:
    - Import modules from transformation.utils
    - Change apply_function to call this new function with its parameters
    - Change create distortion parameters to add this new function's parameters (You should delete the config file parameters.yaml if it already exists)
    - Add its name in the get_transformation function

TODO
    Occlusion
'''

import cv2
import numpy as np
from typing import Any
import os
from numpy.typing import NDArray
import yaml
import sys
from enum import Enum
from pathlib import Path
from tqdm import tqdm
import torch
import subprocess
import time

from computer_vision.utils.transformations import (
    gaussian_blur, gaussian_noise, fog, 
    salt_and_pepper, gaussian_noise_conv,
    salt_and_pepper_conv, 
    time_decaying_artifacts
)

from .config import MODELS_DIR, DISTORTION_DIR, CONFIG_DIR, DEPTH_ANYTHING_DIR

if str(DEPTH_ANYTHING_DIR) not in sys.path:
    sys.path.append(str(DEPTH_ANYTHING_DIR))

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

class DistortionType(Enum):
    GAUSSIAN_NOISE = "gaussian_noise"
    CONVOLVED_GAUSSIAN_NOISE = "gaussian_noise_conv"
    GAUSSIAN_BLUR = "gaussian_blur"
    SALT_AND_PEPPER = "salt_and_pepper"
    CONVOLVED_SALT_AND_PEPPER = "salt_and_pepper_conv"
    FOG = "fog"
    SMOKE = "smoke"
    RAIN = "rain"
    DIRT = "dirt"

class DistortionHandler:
    def __init__(self,
                 video : Path,
                 output_path : Path = DISTORTION_DIR,
                 codec : str = 'libx265',
                 distortion : DistortionType | None = None
                 ):
        
        self.video = video.resolve()

        output_path.mkdir(exist_ok=True, parents=True)
        self.output_path = output_path.resolve()

        # if codec is libx265, we will first save in a codec without loss, FFV1, and then transform it to h265 via ffmpeg
        self.codec = codec if codec != 'libx265' else 'FFV1'

        self.params = self._load_parameters()

        if distortion is not None:
            self.init_for_distortion(distortion, '')

    def init_for_distortion(self, distortion : DistortionType, name : str):
        self.name = name

        self.distortion = distortion

        self.model = self._init_depth_model() if self.distortion == DistortionType.FOG or self.distortion == DistortionType.SMOKE else None

        self.encoder_details = True if self.distortion ==DistortionType.GAUSSIAN_NOISE or self.distortion == DistortionType.SALT_AND_PEPPER else False

    def apply_function(self, image : NDArray, parameters : dict):
    
            if self.distortion == DistortionType.GAUSSIAN_NOISE:
                new_image = gaussian_noise(image, **parameters.get(self.distortion.value, {}))
            
            elif self.distortion == DistortionType.CONVOLVED_GAUSSIAN_NOISE:
                new_image = gaussian_noise_conv(image, **parameters.get(self.distortion.value, {}))
    
            elif self.distortion == DistortionType.GAUSSIAN_BLUR:
                new_image = gaussian_blur(image, **parameters.get(self.distortion.value, {}))
    
            elif self.distortion == DistortionType.SALT_AND_PEPPER:
                new_image = salt_and_pepper(image, **parameters.get(self.distortion.value, {}))
    
            elif self.distortion == DistortionType.CONVOLVED_SALT_AND_PEPPER:
                new_image = salt_and_pepper_conv(image, **parameters.get(self.distortion.value, {}))

            elif self.distortion == DistortionType.SMOKE:
                if self.model is None:
                    raise ValueError("A depth model needs to be instanced to apply the smoke distortion. Recommended: DepthAnything2")
                depth = self.model.infer_image(image)
                new_image = fog(image = image, depth_map = depth, **parameters.get(self.distortion.value, {}))
                new_image = np.clip(new_image,0,255).astype(np.uint8)

            elif self.distortion == DistortionType.FOG:
                if self.model is None:
                    raise ValueError("A depth model needs to be instanced to apply the fog distortion. Recommended: DepthAnything2")
                depth = self.model.infer_image(image)
                new_image = fog(image = image, depth_map = depth, **parameters.get(self.distortion.value, {}))
                new_image = np.clip(new_image,0,255).astype(np.uint8)
    
            elif self.distortion == DistortionType.RAIN:
                new_image, old_noise = time_decaying_artifacts(image, **parameters.get(self.distortion.value, {}))
                parameters['rain']['old_noise'] = old_noise
    
            elif self.distortion == DistortionType.DIRT:
                new_image, old_noise = time_decaying_artifacts(image, **parameters.get(self.distortion.value, {}))
                parameters['dirt']['old_noise'] = old_noise
    
            else:
                raise ValueError(f"There is no distortion function implemented for {self.distortion.value}")
    
            return new_image

    def transform(self, distortion : DistortionType, name : str):
        self.init_for_distortion(distortion=distortion, name=name)

        if len(self.name) == 0:
            dest_file = self.output_path / f'{self.video.stem}_{self.distortion}.mp4'
        else:
            dest_file = self.output_path / f"{self.name}.mp4"

        temp_file = dest_file.with_name(f"{self.video.stem}_{self.distortion}_temp.avi")
        video_write_path = temp_file if self.codec == 'libx265' else dest_file

        cap = cv2.VideoCapture(str(self.video))

        assert cap.isOpened(), "Error reading file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Using codec {self.codec}")

        if self.codec == 'libx265_rawvideo':
            process = self._init_libx265_rawvideo_process(final_video_path=video_write_path, width=w, height=h, fps=fps)
        else:
            video_writer = cv2.VideoWriter(str(video_write_path), cv2.VideoWriter_fourcc(*self.codec), fps, (w,h))
        count_frames = 0

        start_time = time.perf_counter()

        try:
            for i in tqdm(range(total_frames), desc=f"Applying {self.distortion.value} to video: "):
                if cap.isOpened():
                    success, im0 = cap.read()
                    
                    if not success:
                        break
                    
                    new_img = self.apply_function(image = im0, parameters = self.params)

                    if self.codec == "libx265_rawvideo":
                        if process.stdin is not None:
                            process.stdin.write(new_img.tobytes())
                        else:
                            raise BrokenPipeError("Couldn't send bytes to subprocess' stdin PIPE")
                    else:
                        video_writer.write(new_img)

                    count_frames+=1

                else:
                    break

        except KeyboardInterrupt:
            print(f"\n[INFO] Execution interrupted by user at frame {count_frames}. Cleaning up...")

        finally:
            cap.release()

            if self.codec == 'libx265_rawvideo':
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except Exception:
                        pass
                
                process.terminate() 
                process.wait()
                print("[INFO] FFmpeg subprocess successfully terminated.")
            else:       
                video_writer.release()

        if self.codec == 'libx265':
            print('Transforming to H265')
            self._transform_any_codec_to_libx265(temp_video_path=video_write_path, final_video_path=dest_file)

        end_time = time.perf_counter()
        with open(self.output_path / 'time.txt', "a") as time_file:
            time_file.write(f"{self.distortion.value}  {(end_time - start_time)/count_frames}\n")


    def _init_depth_model(self):
        encoder = 'vitl' # or 'vits', 'vitb'
        dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth = 80 # 20 for indoor model, 80 for outdoor model

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available else 'cpu')
        model_path = MODELS_DIR / f'depth_anything_v2_metric_{dataset}_{encoder}.pth'


        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        model = model.to(device).eval()

        return model
    
    def _load_parameters(self) -> dict[str, Any]:
        params_file = CONFIG_DIR / 'parameters.yaml'

        if not params_file.exists():
            self.create_distortions_parameters()

        with open(str(params_file), 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)

        if not params:
            raise ValueError("Couldn't parse yaml file named parameters.yaml with distortion's parameters. It could be empty or corrupted. Try running create_distortion_parameters before continue.")
        
        return params

    def _transform_any_codec_to_libx265(self, temp_video_path:Path, final_video_path:Path):
        subprocess_commands = [
            'ffmpeg', 
            '-y', # Sobrescreve o arquivo final se ele já existir
            '-loglevel', 'error', # Esconde os textos chatos do ffmpeg, mostra só erros
            '-hwaccel', 'cuda',
            '-i', str(temp_video_path), 

            # --- GPU ENCODER SETTINGS ---
            '-c:v', 'hevc_nvenc', 
            '-preset', 'p6', 
            '-cq', '17',           # NVENC Constant Quality

            '-c:a', 'copy',
            
            str(final_video_path)
        ]

        subprocess.run(subprocess_commands, check=True)

        if temp_video_path.exists():
            os.remove(str(temp_video_path))

    def _init_libx265_rawvideo_process(self, final_video_path : Path, width : int, height : int, fps : int):
        pix_fmt = 'yuv420p'
        rate_control = ['-rc', 'vbr', '-cq', '20']
        b_frames = []
        profile = ['-profile:v', 'main']

        # Used for preserving high frequency details in video. In this project, for Salt and Pepper and Gaussian Noise distortions 
        if self.encoder_details:
            pix_fmt = 'yuv420p'

            rate_control = ['-rc', 'constqp', '-qp', '17'] 
            b_frames = [] #['-bf', '0']
            profile = ['-profile:v', 'rext']


        comando_ffmpeg = [
            'ffmpeg',
            '-y',
            # --- INPUT SETTINGS (From Python/OpenCV) ---
            '-f', 'rawvideo',       
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',             
            '-pix_fmt', 'bgr24',          
            '-r', str(fps),               
            '-i', '-',                    

            # --- COLORSPACE ---
            '-color_primaries', 'bt709',
            '-colorspace', 'bt709',
            '-color_trc', 'bt709',
            '-sws_flags', 'accurate_rnd+bitexact',
            
            # --- GPU ENCODER SETTINGS ---
            '-c:v', 'hevc_nvenc',
            '-preset', 'p6',          
            '-pix_fmt', pix_fmt,

            # NVENC specific flags to disable adaptive quantization smoothing 
            '-spatial-aq', '0',
            '-temporal-aq', '0', 

        ]
        comando_ffmpeg.extend(rate_control)
        comando_ffmpeg.extend(b_frames)
        comando_ffmpeg.extend(profile)
        
        comando_ffmpeg.append(str(final_video_path))

        process = subprocess.Popen(comando_ffmpeg, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return process 

    @staticmethod
    def get_transformations():
        return ['gaussian_noise', 
                'gaussian_noise_conv', 
                'gaussian_blur',
                'smoke', 
                'salt_and_pepper', 
                'rain',
                'dirt',
                'fog']

    def create_distortions_parameters(self):
        params = {
            "gaussian_noise": {
                "mean": 0.0,
                "stdev": 10. # tests jusqu'a 10
            },
            "gaussian_noise_conv":{
                "mean": 0.0,
                "stdev": 10,
                "kernel_size": 9
            },
            "gaussian_blur": {
                "kernel_size": 11,
                "stdev": 0.
            },
            "fog": {
                "minimum_distance": 10,
                "airlight": None,
                "per_channel_airlight": False 
            },
            "salt_and_pepper": {
                "salt_prob": 0.03,
                "pepper_prob": 0.03
            },
            "salt_and_pepper_conv": {
                "salt_prob": 0.0003,
                "pepper_prob": 0.0003,
                "kernel_size": 9,
                "sigma": 1.23
            },
            "rain": {
                "old_noise": None,
                "event_probability": 0.2,
                "decay_factor": 0.98,
                "kernel_size": 201,
                "kernel_factor": 150,
                "sigma": 39,
                "additive": True
            },
            "dirt": {
                "old_noise": None,
                "event_probability": 0.033,
                "decay_factor": 1.0,
                "kernel_size": 151,
                "kernel_factor": 160,
                "sigma": 29,
                "additive": False
            }
        }
        yaml_path = CONFIG_DIR / 'parameters.yaml'

        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(str(yaml_path), "w") as yaml_file:
            yaml.dump(params, yaml_file, default_flow_style=False)
    