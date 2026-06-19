'''

To execute:
python3 predict_video.py --model_path yolo26l.pt --video /app/videos/cad42.mp4 --name cad42_video_large --save_txt True --save True --show False 


'''


from ultralytics import YOLO
from pathlib import Path
import sys
import argparse

ROOT_DIRECTORY = Path(__file__).resolve().parents[2]

DEFAULT_PROJECT_PATH = ROOT_DIRECTORY / "src" / "computer_vision" / "predict"

def str2bool(v: str | bool) -> bool:
    """Auxiliar function for argparse to read a boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args() ->argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediction using a yolo model"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default='yolo26n.pt',
        help="Path to the model used to make predictions."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Video used for prediction."
    )
    parser.add_argument(
        "--project",
        type = Path,
        default=DEFAULT_PROJECT_PATH,
        help="Name of the project directory where prediction outputs are saved."
    )
    parser.add_argument(
        "--name",
        type=str,
        default='yolo26_video',
        help="Name of the prediction run. Used to save outputs in a subdirectory with the same name"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs='+',
        default=[0],
        help="Filters prediction to a set of class IDs. Default is [1], since it's the person's ID"
    )
    parser.add_argument(
        "--all-classes",
        action="store_true", 
        help="If set, ignores --classes and detects all objects."
    )
    parser.add_argument(
        "--stream",
        type=str2bool,
        default=True,
        help="Enables emmemory-efficient processing for vidoes or numerous images returning a generator."
    )
    parser.add_argument(
        "--save",
        type=str2bool,
        default=True,
        help="Saving the annotated files (images or video)"
    )
    parser.add_argument(
        "--save_txt",
        type=str2bool,
        default=False,
        help="Saves detection results in a txt file, following the format [class] [x_center] [y_center] [width] [height] [confidence]"
    )
    parser.add_argument(
        "--show",
        type=str2bool,
        default=True,
        help="Displays the annotated files in a window"
    )

    return parser.parse_args()


args = parse_args()

model_path = args.model_path.resolve() if args.model_path.exists() else str(args.model_path)
video_path = args.video.resolve()
project_path = args.project.resolve()

if args.all_classes:
    classes_to_detect = None 
else:
    classes_to_detect = args.classes


model = YOLO(model_path)

yolo_kwargs = {
    "source": video_path,
    "classes": classes_to_detect,
    "project": str(project_path),
    "name": args.name,
    "stream": args.stream,
    "save": args.save,
    "save_txt": args.save_txt,
    "save_conf": args.save_txt,
    "show": args.show
}

print("Model's parameters: ")
print(yolo_kwargs)

results = model(**yolo_kwargs)

try:
    for result in results:
        pass
except KeyboardInterrupt:
    print("User exited the program")
    print(results)
    sys.exit(1)