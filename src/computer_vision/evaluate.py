import argparse
import yaml
from pathlib import Path

from core.config import DATA_DIR, EVALUATIONS_DIR, VIDEO_DIR, ALARM_CONFIG
from core.zone_detection import ZoneDetector



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
    parser.add_argument(
        "--alarm",
        type=str,
        required=True,
        help="Name of the alarm. This will be used to search for the alarm detection frequency"
    )
    parser.add_argument(
        "--distortion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set to true, every single distorted video will also be put to test."
    )
    return parser.parse_args()


def evaluate():

    args = parse_args()

    print('\n\n')
    print("-"*30 + " STARTING DATASET EVALUATION " + "-"*30)
    video_dir = VIDEO_DIR.resolve()

    yaml_file = DATA_DIR / f'{args.dataset}.yaml'
    yaml_file = yaml_file.resolve()

    if not yaml_file.exists():
        raise FileExistsError(f"The yaml file {args.dataset}.yaml does not exist. Try adding a yaml file to the data via generate_data module")
    else:
        with open(str(yaml_file), 'r') as yaml_file:
            general_data = yaml.safe_load(yaml_file)

    if args.distortion:
        mp4_files = list(video_dir.rglob("*.mp4"))
    else:
        mp4_files = list(video_dir.glob("*.mp4"))

    print("List of videos to be evaluated:")
    for file in mp4_files:
        print(str(file)) 

    # CSV file
    csv_path = EVALUATIONS_DIR / args.name / f'{args.name}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    count_videos = 1

    zone_detector = ZoneDetector(alarm=args.alarm, show = False, csv = csv_path, delay_window=90, close_detection_gaps=30)

    for file in mp4_files:
        print("="*30 + f" EVALUATING VIDEO {count_videos}: {file.stem.upper()} " + "="*30 + "\n")
        print(f"The annotation file used for {file} is {general_data['data'][str(file)]}")

        bbox_points = general_data['bbox_points'][str(file)]

        print(f"\nVideo Path: {file}")

        zone_detector.detect_zones(video = file,
                                   predictions = Path(general_data['data'][str(file)]),
                                   point_d = (bbox_points[0], bbox_points[1]),
                                   point_u = (bbox_points[2], bbox_points[3]))

        count_videos += 1


if __name__ == '__main__':
    evaluate()