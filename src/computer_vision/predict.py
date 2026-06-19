from ultralytics import YOLO
from pathlib import Path
import sys
import argparse

def parse_args() ->argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediction using a yolo model"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default='yolo26n.pt',
        help="Path to the model used to make predictions"
    )

    return parser.parse_args()

PATH_FILE = Path(__file__).resolve().parents[1]

args = parse_args()

model_path = args.model_path.resolve()

model = YOLO(model_path)

results = model(source=0, stream=True, show=True)

try:
    for result in results:
        pass
except KeyboardInterrupt:
    print("User exited the program")
    print(results)
    sys.exit(1)