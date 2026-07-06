'''
Example of running script:

python3 video_augmentation.py --video /home/lucas/Documents/computer_vision/videos/marcher_180.mp4 \
    --dest /home/lucas/Documents/computer_vision/videos/distortion \
    --name marcher_gaussian_noise \
    --distortion gaussian_noise

    Salt and Pepper
    Bruit Gaussien avec Lissage
'''


import cv2
import numpy as np
import typing
from typing import Any
from numpy.typing import NDArray
import os
import yaml
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

from utils.transformations import gaussian_blur, gaussian_noise, fog, salt_and_pepper

ROOT_DIR = Path(__file__).resolve().parents[2]
DEPTH_ANYTHING_DIR = ROOT_DIR / "Depth-Anything-V2"

if str(DEPTH_ANYTHING_DIR) not in sys.path:
    sys.path.append(str(DEPTH_ANYTHING_DIR))

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

DEFAULT_TRANSF_VIDEOS_DIR = ROOT_DIR / "videos" / "distortions"
DEFAULT_DISTORTION_CONFIG_PATH = ROOT_DIR / 'src' / 'computer_vision' / 'config'
DEFAULT_MODELS_PATH = ROOT_DIR / 'src' / 'computer_vision' / 'models'

def parse_args(arg_list=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Applying a transofrmation in each frame of a video'
    )

    parser.add_argument("--video", 
                        type=Path, 
                        required=True, 
                        help="Video that will be transformed"
    )
    parser.add_argument("--dest", 
                        type=Path, 
                        default = DEFAULT_TRANSF_VIDEOS_DIR, 
                        help="Name of the folder where the new video will be saved"
    )
    parser.add_argument("--name", 
                        type=str, 
                        default = "", 
                        help = "Name of the video"
    )
    parser.add_argument("--distortion",
                        type=str,
                        required=True,
                        choices=['gaussian_noise', 'gaussian_blur', 'fog'],
                        help="Distortion to be applied in the image"
    )

    return parser.parse_args(arg_list)

def generate_depth_model():
    encoder = 'vitl' # or 'vits', 'vitb'
    dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available else 'cpu')
    model_path = DEFAULT_MODELS_PATH / f'depth_anything_v2_metric_{dataset}_{encoder}.pth'


    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model = model.to(device).eval()

    return model

def apply_function(image : NDArray, distortion_str : str, parameters : dict, depth_model=None):
    dist_name = distortion_str.lower()

    if dist_name == 'gaussian_noise':
        new_image = gaussian_noise(image, **parameters.get('gaussian_noise', {}))

    elif dist_name == 'gaussian_blur':
        new_image = gaussian_blur(image, **parameters.get('gaussian_blur', {}))

    elif dist_name == 'salt_and_pepper':
        new_image = salt_and_pepper(image, **parameters.get('salt_and_pepper', {}))

    elif dist_name == 'fog':
        if depth_model is None:
            raise ValueError("A depth model needs to be instanced to apply the fog distortion. Recommended: DepthAnything2")
        depth = depth_model.infer_image(image)
        new_image = fog(image = image, depth_map = depth, **parameters.get('fog', {}))
        new_image = np.clip(new_image,0,255).astype(np.uint8)

    else:
        raise ValueError(f"There is no distortion function implemented for {dist_name}")

    return new_image


def create_distortions_parameters():
    params = {
        "gaussian_noise": {
            "mean": 0.0,
            "stdev": 60.
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
        }
    }
    yaml_path = DEFAULT_DISTORTION_CONFIG_PATH / 'parameters.yaml'

    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(yaml_path), "w") as yaml_file:
        yaml.dump(params, yaml_file, default_flow_style=False)


def load_parameters() -> dict[str, Any]:
    params_file = DEFAULT_DISTORTION_CONFIG_PATH / 'parameters.yaml'

    if not params_file.exists():
        create_distortions_parameters()

    with open(str(params_file), 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)

    if not params:
        raise FileNotFoundError("Couldn't parse yaml file named parameters.yaml with distortion's parameters. It could be empty or corrupted. Try running create_distortion_parameters before continue.")
    
    return params

def get_transformations():
    return ['gaussian_noise', 'gaussian_blur', 'fog', 'salt_and_pepper']

def transform(arg_list=None):
    args = parse_args(arg_list)

    video_path = args.video.resolve()

    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    dest_folder = dest.resolve()

    if len(args.name) == 0:
        dest_file = dest_folder / str(video_path.stem) + f'_{args.distorion}.mp4'
    else:
        dest_file = dest_folder / f"{args.name}.mp4"

    cap = cv2.VideoCapture(str(video_path))

    assert cap.isOpened(), "Error reading file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_writer = cv2.VideoWriter(str(dest_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    params = load_parameters()

    model = generate_depth_model() if args.distortion == 'fog' else None

    count_frames = 0

    for i in tqdm(range(total_frames), desc=f"Applying {args.distortion} to video: "):
        if cap.isOpened():
            success, im0 = cap.read()
            
            if not success:
                break
            
            new_img = apply_function(image = im0, distortion_str = args.distortion, parameters = params, depth_model=model)

            video_writer.write(new_img)

            count_frames+=1

        else:
            break

    cap.release()
    video_writer.release()


if __name__ == "__main__":
    transform()