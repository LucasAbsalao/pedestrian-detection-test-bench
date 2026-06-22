import cv2
import numpy as np
import typing
import os
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEPTH_ANYTHING_DIR = ROOT_DIR / "Depth-Anything-V2"

if str(DEPTH_ANYTHING_DIR) not in sys.path:
    sys.path.append(str(DEPTH_ANYTHING_DIR))


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
                        help="Distortion to be applied in the image"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    video_path = args.video.resolve()

    dest = args.dest
    if not dest.exists():
        os.makedirs(dest)
    dest_folder = dest.reolve()

    dest_file = dest_folder / f"{args.name}.mp4"

    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened, "Error reading file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(str(dest_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    while cap.isOpened():
        success, im0 = cap.read()

        if not success:
            print("Video frame is either empty or processing is complete")
            break

        im0 = im0.copy() # Change here

        video_writer.write(im0)

    cap.release()
    video_writer.release()


if __name__ == "__main__":
    main()