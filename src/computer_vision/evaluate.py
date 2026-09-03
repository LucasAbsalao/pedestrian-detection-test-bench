'''
python3 evaluate.py --name brigade_petit_test --alarm brigade_test 2>&1 | tee log_brigade_test.txt

python3 evaluate.py --name blaxtair_new_test --alarm blaxtair 2>&1 | tee log/log_blaxtair_new.txt
'''
import argparse
import yaml
from pathlib import Path
import time

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
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If passed, skips videos that already exist in the output CSV."
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

    # CSV file
    csv_path = EVALUATIONS_DIR / args.name / f'{args.name}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    evaluated_videos = set()
    
    if args.resume and csv_path.exists() and csv_path.stat().st_size > 0:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        video_column_header = 'Name' 
        
        if video_column_header in df.columns:
            evaluated_videos = set(df[video_column_header].astype(str))
        else:
            print(f"Warning: '{video_column_header}' not found in {csv_path.name}.")
            print("Cannot filter resumed videos properly. Starting from scratch or check column name.")

    videos_to_evaluate = []
    for file in mp4_files:
        if args.resume and file.name in evaluated_videos:
            pass
        else:
            videos_to_evaluate.append(file)
            
    mp4_files = videos_to_evaluate

    print(f"List of {len(mp4_files)} videos to be evaluated:")
    for file in mp4_files:
        print(str(file)) 

    count_videos = 1

    zone_detector = ZoneDetector(alarm=args.alarm, show = False, csv = csv_path, delay_window=100, close_detection_gaps=30, close_audio_gaps=45)

    start_time = time.perf_counter()
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
    final_time = time.perf_counter()

    with open(csv_path.parent / "time.txt", 'w') as time_file:
        time_file.write(f'Total Elapsed Time: {final_time - start_time}')

if __name__ == '__main__':
    evaluate()