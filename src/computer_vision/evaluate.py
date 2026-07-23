import numpy as np
import sys
import argparse
import yaml
from pathlib import Path

import scripts.zone_counter as zone_counter


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_DATA_DIR = ROOT_DIR / 'data'
DEFAULT_ANNOTATION_DIR = DEFAULT_DATA_DIR / 'annotations'
DEFAULT_VIDEO_DIR = DEFAULT_DATA_DIR / 'videos'
DEFAULT_DISTORTION_DIR = DEFAULT_VIDEO_DIR / 'distortions'
DEFAULT_CSV_DIR = ROOT_DIR / 'evaluations'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script responsible for testing all videos in the data and outputing its values"
    )
    parser.add_argument(
        "--name",
        type = str,
        default = "eval",
        help="Name of this evaluation run. Results will be saved in a csv with the same name."
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = "dataset_engins_de_chantier",
        help="Name of the dataset. Will look for an yaml with this name."
    )
    return parser.parse_args()


def evaluate():

    args = parse_args()

    print('\n\n')
    print("-"*30 + " STARTING DATASET EVALUATION " + "-"*30)
    video_dir = DEFAULT_VIDEO_DIR.resolve()

    yaml_file = DEFAULT_DATA_DIR / f'{args.dataset}.yaml'
    yaml_file = yaml_file.resolve()

    if not yaml_file.exists():
        raise FileExistsError(f"The yaml file {args.dataset}.yaml does not exist. Try adding a yaml file to the data via generate_data module")
    else:
        with open(str(yaml_file), 'r') as yaml_file:
            general_data = yaml.safe_load(yaml_file)

    mp4_files = list(video_dir.rglob("*.mp4"))
    print("List of videos to be evaluated:")
    for file in mp4_files:
        print(str(file)) 

    # CSV file
    csv_path = DEFAULT_CSV_DIR / f'{args.name}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    count_videos = 1
    for file in mp4_files:
        print("="*30 + f" EVALUATING VIDEO {count_videos}: {file.stem.upper()} " + "="*30 + "\n")
        print(f"The annotation file used for {file} is {general_data['data'][str(file)]}")

        bbox_points = general_data['bbox_points'][str(file)]

        print(f"\nVideo Path: {file}")
        evaluate_args = [
            '--video', str(file),
            '--predictions', general_data['data'][str(file)],
            '--draw', 'True',
            '--csv', str(csv_path),
            '--point-d', str(bbox_points[0]), str(bbox_points[1]), 
            '--point-u', str(bbox_points[2]), str(bbox_points[3])  
        ]
        zone_counter.count_zones(evaluate_args)

        count_videos += 1



if __name__ == '__main__':
    evaluate()