import numpy as np
import cv2
import os
import sys
import argparse
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_DATA_DIR = ROOT_DIR / 'data'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main script that grabs all videos from a folder, makes all annotations and apply distortions"
    )

    return parser.parse_args()


def main():
    print('\n\n')
    print("-"*30 + "STARTING MAIN EXECUTION" + "-"*30)
    video_dir = DEFAULT_DATA_DIR / 'videos'
    video_dir = video_dir.resolve()

    mp4_files = list(video_dir.glob("*.mp4"))
    print("List of videos to analyse:")
    for file in mp4_files:
        print(str(file)) 

    print("List of distortions that will be applied to each video:")
    


if __name__ == '__main__':
    main()