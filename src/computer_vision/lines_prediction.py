'''

To execute:
python3 meters_prediction.py \
    --model_path yolo26x.pt \
    --video /home/lucas/Documents/computer_vision/videos/GoLiveTogether.mp4 \
    --name GoLiveTogether_meters \
    --save_txt False \
    --save False \
    --show False \
    --stream False


'''


import cv2
import argparse
from pathlib import Path
import sys
from ultralytics import YOLO

ROOT_DIRECTORY = Path(__file__).resolve().parents[2]
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

DEFAULT_PROJECT_PATH = ROOT_DIRECTORY / "src" / 'computer_vision' / "predict"

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
        help="Enables memory-efficient processing for vidoes or numerous images returning a generator."
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

def generate_trapezes(point_d, point_u, width):
    point_r1 = point_d
    point_r2 = (width - point_d[0], point_d[1])

    point_g1 = point_u
    point_g2 = (width - point_u[0], point_u[1])

    point_r3 = ((point_u[0]-point_d[0])//3 + point_d[0], point_d[1] - (point_d[1]-point_u[1])//3)
    point_r4 = (width - ((point_u[0]-point_d[0])//3 + point_d[0]), point_d[1] - (point_d[1]-point_u[1])//3)

    point_o1 = (2*(point_u[0]-point_d[0])//3 + point_d[0], point_d[1] - 2*(point_d[1]-point_u[1])//3)  
    point_o2 = (width - (2 * (point_u[0]-point_d[0])//3 + point_d[0]), point_d[1] - 2*(point_d[1]-point_u[1])//3)

    trapezes = [[point_r1, point_r3, point_r4, point_r2], 
                [point_r3, point_o1, point_o2, point_r4], 
                [point_o1, point_g1, point_g2, point_o2]]

    return trapezes

def intersect(point, trapezes, verbose = False):
    i = 0
    for trapeze in trapezes:
        ld, lu, ru, rd = trapeze
        if verbose:
            print("Trapeze: ", ld, lu, ru, rd, "| Point: ", point)
        if (point[0]>ld[0] and point[0]<rd[0]) and (point[1]<ld[1] and point[1]>lu[1]):
            ang_coef = (lu[1]-ld[1])/(lu[0]-ld[0])
            if lu[0] <= point[0] <= ru[0]:
                return i
            elif point[0] < lu[0]:
                lateral_limit_l = ld[1] + ang_coef * (point[0]-ld[0])
                if point[1]>lateral_limit_l:
                    return i
            elif point[0]>ru[0]:
                lateral_limit_l = ru[1] - ang_coef * (point[0]-ru[0])
                if point[1]>lateral_limit_l:
                    return i
        i+=1
                
    return i


def write_bbox(image, bbox, trapezes, rectangle = True):
    colors = [(0,0,255), (0,150,255), (0,255,0), (255,0,0)]
    print("Writing Bounding Box in frame: ")
    for box in bbox:
        print("BBOX")
        points = [(box[0], box[1]), (box[0], box[3]), (box[2], box[3]), (box[2], box[1])]
        colors_circles = []
        for p in points:
            colors_circles.append(intersect(p, trapezes))

        if rectangle:
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=colors[min(colors_circles)], thickness=3)
        else:
            cv2.circle(image, center=(int(box[0]), int(box[1])), radius=6, color=colors[colors_circles[0]], thickness=-1)
            cv2.circle(image, center=(int(box[0]), int(box[3])), radius=6, color=colors[colors_circles[1]], thickness=-1)
            cv2.circle(image, center=(int(box[2]), int(box[3])), radius=6, color=colors[colors_circles[2]], thickness=-1)
            cv2.circle(image, center=(int(box[2]), int(box[1])), radius=6, color=colors[colors_circles[3]], thickness=-1)

    

def write_lines(image, point_d, point_u, width):
    point_r1 = point_d
    point_r2 = (width - point_d[0], point_d[1])

    point_g1 = point_u
    point_g2 = (width - point_u[0], point_u[1])

    point_r3 = ((point_u[0]-point_d[0])//3 + point_d[0], point_d[1] - (point_d[1]-point_u[1])//3)
    point_r4 = (width - ((point_u[0]-point_d[0])//3 + point_d[0]), point_d[1] - (point_d[1]-point_u[1])//3)

    point_o1 = (2*(point_u[0]-point_d[0])//3 + point_d[0], point_d[1] - 2*(point_d[1]-point_u[1])//3)  
    point_o2 = (width - (2 * (point_u[0]-point_d[0])//3 + point_d[0]), point_d[1] - 2*(point_d[1]-point_u[1])//3)
    
    image = cv2.line(image, point_r1, point_r2, color=(0,0,255), thickness=3)
    image = cv2.line(image, point_r3, point_r4, color = (0,0,255), thickness=3)
    image = cv2.line(image, point_r1, point_r3, color = (0,0,255), thickness=3)
    image = cv2.line(image, point_r2, point_r4, color = (0,0,255), thickness=3)

    image = cv2.line(image, point_r3, point_o1, color=(0,150,255), thickness=3)
    image = cv2.line(image, point_r4, point_o2, color=(0,150,255), thickness=3)
    image = cv2.line(image, point_o1, point_o2, color=(0,150,255), thickness=3)

    image = cv2.line(image, point_o1, point_g1, color=(0,255,0), thickness=3)
    image = cv2.line(image, point_o2, point_g2, color=(0,255,0), thickness=3)
    image = cv2.line(image, point_g1, point_g2, color=(0,255,0), thickness=3)
    
    return image 

args = parse_args()

model_path = args.model_path if args.model_path.exists() else str(args.model_path)
video_path = args.video.resolve()
project_path = args.project.resolve()

if args.all_classes:
    classes_to_detect = None
else:
    classes_to_detect = args.classes

print(video_path)
cap = cv2.VideoCapture(video_path)

assert cap.isOpened(), "Error reading video file"


w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(str(project_path / f"{str(args.name)}.avi"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

model = YOLO(model_path)

yolo_kwargs = {
    "classes": classes_to_detect,
    "project": str(project_path),
    "name": args.name,
    "stream": args.stream,
    "save": args.save,
    "save_txt": args.save_txt,
    "save_conf": args.save_txt,
    "show": args.show
}

#Points of depth perspective
point_1 = (300, 700)
point_2 = (500, 300)

trapezes = generate_trapezes(point_d=point_1, point_u=point_2, width=w)

while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = model(source = im0, **yolo_kwargs)

    print("Results Obtained: ")
    for r in results:
        print("------------------------------------- Bounding Boxes -------------------------------------")

        im0 = write_lines(im0, point_1, point_2, w)
        
        write_bbox(image = im0, bbox = r.boxes.xyxy.cpu().numpy(), trapezes = trapezes, rectangle=True)

        video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
