'''
To execute:
    python3 evaluate_1_video.py \
    --video /home/lucas/Documents/computer_vision/data/videos/distortions/GX010080_00_2_gaussian_noise.mp4 \
    --predictions /home/lucas/Documents/computer_vision/data/annotations/yolo_GX010080_00_2.txt \
    --point-d 274 1039     --point-u 702 310 \
    --csv test_gn/results.csv \
    --alarm blaxtair
'''
import argparse
from pathlib import Path

from computer_vision.core.zone_detection import ZoneDetector



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script responsible for testing all videos in the data and outputing its values"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Video file to display." 
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the predictions txt file from lines_prediction.py"
    )
    parser.add_argument(
        "--csv",
        type = Path,
        default='results.csv',
        help="Path to the csv where this run is going to be saved."
    )
    parser.add_argument(
        "--point-d",
        type=int,
        nargs=2,
        default=[400, 1000],
        help="Lower trapezoid point (x y)"
    )
    parser.add_argument(
        "--point-u",
        type=int,
        nargs=2,
        default=[550, 600],
        help="Upper trapezoid point (x y)"
    )
    parser.add_argument(
        "--alarm",
        type=str,
        required=True,
        help="Name of the alarm. This will be used to search for the alarm detection frequency."
    )
    return parser.parse_args()


def evaluate():

    args = parse_args()

    print('\n\n')
    print("-"*30 + " STARTING DATASET EVALUATION " + "-"*30)
    video_path = args.video
    video_path = video_path.resolve()

    # CSV file
    csv_path = args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    zone_detector = ZoneDetector(alarm=args.alarm, show = False, csv = csv_path, save_audio=True, close_audio_gaps=45)

    predictions = args.predictions
    predictions = predictions.resolve()


    print("="*30 + f" EVALUATING VIDEO {video_path}: {video_path.stem.upper()} " + "="*30 + "\n")
    print(f"The annotation file used for {video_path.stem} is {csv_path}")

    point_d = args.point_d
    point_u = args.point_u

    zone_detector.detect_zones(video = video_path,
                               predictions = predictions,
                               point_d = (point_d[0], point_d[1]),
                               point_u = (point_u[0], point_u[1]))


if __name__ == '__main__':
    evaluate()