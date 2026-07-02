'''
Example of running script:

python3 video_augmentation.py --video /home/lucas/Documents/computer_vision/videos/marcher_180.mp4 \
    --dest /home/lucas/Documents/computer_vision/videos/distortion \
    --name marcher_gaussian_noise \
    --distortion gaussian_noise

'''


import cv2
import numpy as np
import typing
from numpy.typing import NDArray
import os
import sys
import argparse
from pathlib import Path
import torch

from utils.transformations import gaussian_blur, gaussian_noise, fog

ROOT_DIR = Path(__file__).resolve().parents[2]
DEPTH_ANYTHING_DIR = ROOT_DIR / "Depth-Anything-V2"

if str(DEPTH_ANYTHING_DIR) not in sys.path:
    sys.path.append(str(DEPTH_ANYTHING_DIR))

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

DEFAULT_TRANSF_VIDEOS_DIR = ROOT_DIR / "videos" / "distortions"

def parse_args() -> argparse.Namespace:
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

    return parser.parse_args()

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

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()

    return model

def apply_function(image : NDArray, distortion_str : str, parameters : dict):
    if distortion_str.lower() == 'gaussian_noise':
        new_image = gaussian_noise(image, **parameters['gaussian_noise'])

    elif distortion_str.lower() == 'gaussian_blur':
        new_image = gaussian_blur(image, **parameters['gaussian_blur'])

    elif distortion_str.lower() == 'fog':
        depth = parameters["depth_model"].infer_image(image)
        new_image = fog(image = image, depth_map = depth, **parameters['fog'])
        new_image = np.clip(new_image,0,255).astype(np.uint8)

    else:
        raise Exception("There is no distortion function implemented for this string")

    return new_image


def set_parameters():
    params = {
        "gaussian_noise": {
            "mean": 0.0,
            "stdev": 60.
        },
        "gaussian_blur": {
            "kernel_size": 11,
            "stdev": 0.
        },
        "depth_model": DepthAnythingV2(),
        "fog": {
            "minimum_distance": 10,
            "airlight": None,
            "per_channel_airlight": False 
        }
    }
    return params

def get_transformations():
    return ['gaussian_noise', 'gaussian_blur', 'fog']

def main():
    args = parse_args()

    video_path = args.video.resolve()

    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    dest_folder = dest.resolve()

    dest_file = dest_folder / f"{args.name}.mp4"

    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened, "Error reading file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_writer = cv2.VideoWriter(str(dest_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    params = set_parameters()

    if args.distortion == 'fog':
        model = generate_depth_model()
        params['depth_model'] = model

    count_frames = 0

    while cap.isOpened():
        success, im0 = cap.read()

        if not success:
            print("Video frame is either empty or processing is complete")
            break
        
        new_img = apply_function(image = im0, distortion_str = args.distortion, parameters = params)

        video_writer.write(new_img)

        count_frames+=1
        print(f"frame {count_frames}/{total_frames}")

    cap.release()
    video_writer.release()


if __name__ == "__main__":
    main()