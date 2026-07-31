'''

To execute:
python3 lines_prediction.py \
    --model_path models/yolo26x.pt \
    --video /home/lucas/Documents/computer_vision/videos/GoLiveTogether.mp4 \
    --name GoLiveTogether_meters \
    --save_txt False \
    --save False \
    --show False \
    --stream False \
    --point-d 400 1000     --point-u 550 600


'''


import cv2
import argparse
from pathlib import Path
import sys
from ultralytics import YOLO
from numpy.typing import NDArray
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import utils.draw as draw

ROOT_DIRECTORY = Path(__file__).resolve().parents[3]
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

DEFAULT_PROJECT_PATH = ROOT_DIRECTORY / "src" / 'computer_vision' / "predict" / "annotation" 
DEFAULT_TXT_PATH = ROOT_DIRECTORY / "data" / "annotations"

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

def parse_args(args_list=None) -> argparse.Namespace:
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
        help="Filters prediction to a set of class IDs. Default is [0], since it's the person's ID"
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
        help="Enables memory-efficient processing for videos or numerous images returning a generator."
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
    parser.add_argument(
        "--txt-path",
        type=Path,
        default=DEFAULT_TXT_PATH,
        help="Directory where the bounding boxes by frames will be saved"
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
    return parser.parse_args(args_list)

def continuous_morphological_closing(detection_array : NDArray, closing_se_size : int):
    assert closing_se_size%2==1, "closing_se_size has to be odd"

    closed_detection_array = np.copy(detection_array)
    min_value = np.min(detection_array)
    max_value = np.max(detection_array)

    pad = int(closing_se_size // 2)

    #Dilate
    detection_array_padded_min = np.pad(detection_array, (pad, pad), mode='constant', constant_values=min_value)

    for i in range(len(detection_array)):
        closed_detection_array[i] = np.max(detection_array_padded_min[i:i+closing_se_size])

    #Erode
    detection_array_padded_max = np.pad(detection_array, (pad, pad), mode='constant', constant_values=max_value)
    
    for i in range(len(detection_array)):
        closed_detection_array[i] = np.max(detection_array_padded_max[i:i+closing_se_size])



def predict(args_list = None):
    args = parse_args(args_list)

    model_path = args.model_path if args.model_path.exists() else str(args.model_path)
    video_path = args.video.resolve()
    project_path = args.project.resolve()

    if args.all_classes:
        classes_to_detect = None
    else:
        classes_to_detect = args.classes

    print("Starting prediction of video: ")
    print(video_path)
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened(), "Error reading video file"


    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(str(project_path / f"{str(args.name)}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    model = YOLO(model_path)

    yolo_kwargs = {
        "classes": classes_to_detect,
        "project": str(project_path),
        "name": args.name,
        "stream": args.stream,
        "save": args.save,
        "save_txt": args.save_txt,
        "save_conf": args.save_txt,
        "show": args.show,
        "verbose": False
    }

    #Points of depth perspective
    point_1 = args.point_d
    point_2 = args.point_u

    trapezes = draw.generate_trapezes(point_d=point_1, point_u=point_2, width=w)

    count_frames = 0

    txt_path = args.txt_path.resolve()

    with open(str(txt_path / f"yolo_{args.name}.txt"), "w") as f:
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = model(source = im0, **yolo_kwargs)

            # print("Results Obtained: ")
            f.write(f"{count_frames}\n")
            for r in results:
                # print("------------------------------------- Bounding Boxes -------------------------------------")

                im0 = draw.write_lines(im0, point_1, point_2, w)
                
                zone = draw.draw_bbox(image = im0, bbox = r.boxes.xyxy.cpu().numpy(), trapezes = trapezes, rectangle=True)

                video_writer.write(im0)

                bbox = r.boxes.xyxy.cpu().numpy()
                for bbox_idx in range(len(bbox)):
                    f.write(f"{zone[bbox_idx]} ")
                    for coord in bbox[bbox_idx]:
                        f.write(f"{coord} ")
                    f.write("\n")

            count_frames += 1

    cap.release()
    video_writer.release()

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    predict()
