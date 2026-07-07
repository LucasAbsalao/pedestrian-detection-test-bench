import sys
import argparse
import yaml
from pathlib import Path

import lines_prediction
import video_augmentation
import draw_trapeze_points


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_DATA_DIR = ROOT_DIR / 'data'
DEFAULT_ANNOTATION_DIR = DEFAULT_DATA_DIR / 'annotations'
DEFAULT_VIDEO_DIR = DEFAULT_DATA_DIR / 'videos'
DEFAULT_DISTORTION_DIR = DEFAULT_VIDEO_DIR / 'distortions'
DEFAULT_MODELS_DIR = ROOT_DIR / 'src' / 'computer_vision' / 'models'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main script that grabs all videos from a folder, makes all annotations and apply distortions"
    )
    parser.add_argument(
        "--name",
        type = str,
        default = "dataset_engins_de_chantier",
        help="Name of the generated dataset. Will be present at yaml."
    )
    parser.add_argument(
        '--model',
        type = str,
        default = str(DEFAULT_MODELS_DIR / 'yolo26x.pt'),
        help = 'Model used to annotate images.'
    )
    parser.add_argument(
        '--points',
        type=Path,
        default = DEFAULT_DATA_DIR / "points.yaml",
        help="Path to the yaml file storing points used for prioritized zone detection"
    )
    parser.add_argument(
        "--force",
        type=bool,
        default=False,
        help="If set, erases any precedent annotation file for these videos."
    )
    return parser.parse_args()


def get_points_from_yaml(points_file_path : Path) -> dict | None :
    if points_file_path.exists():
        with open(str(points_file_path), 'r') as yaml_file:
            point_data = yaml.safe_load(yaml_file)
        return point_data 
    else:
        return None


def add_points_to_file(file : Path, yaml_path : Path):
    print(f"Adding point to {file}")
    points_args = [
        "--name", yaml_path.stem,
        "--config", str(yaml_path.parent),
        "--video", str(file)
    ]
    draw_trapeze_points.mark_points(points_args)


def predict(file : Path, point_data : dict , args : argparse.Namespace):

    # Check if this file has a point associated to it
    if str(file) not in list(point_data.keys()):
        add_points_to_file(file, args.points)
        point_data = get_points_from_yaml(args.points) or {}
        if point_data is None:
            raise FileExistsError("Couldn't create yaml file")
    
    points_xyxy = point_data[str(file)]

    predict_args = [
        '--model_path', args.model,
        '--video', str(file),
        '--project', 'predict',
        '--name', file.stem,
        '--stream', 'False',
        '--save', 'False',
        '--save_txt', 'False',
        '--show', 'False',   
        '--txt-path', str(DEFAULT_ANNOTATION_DIR),
        '--point-d', str(points_xyxy[0]), str(points_xyxy[1]), 
        '--point-u', str(points_xyxy[2]), str(points_xyxy[3])
    ]
    lines_prediction.predict(predict_args) 

    print("Annotations saved!!!\n\n")

def apply_distortion(file : Path, distortion : str, args : argparse.Namespace) -> Path:
    distortion_video_name = file.stem + '_' + distortion

    distortion_path = DEFAULT_DISTORTION_DIR / f"{distortion_video_name}.mp4"
    if distortion_path.exists() and not args.force:
        return distortion_path

    print('='*15 + distortion + '='*15)
    
    distortion_args = [
        "--video", str(file),
        "--dest", str(DEFAULT_DISTORTION_DIR),
        "--name", distortion_video_name,
        "--distortion", distortion
    ]
    video_augmentation.transform(distortion_args)

    return distortion_path
    

def generate():

    args = parse_args()

    print('\n\n')
    print("-"*30 + " STARTING DATASET GENERATION " + "-"*30)
    video_dir = DEFAULT_VIDEO_DIR.resolve()

    mp4_files = list(video_dir.glob("*.mp4"))
    print("List of videos to analyse:")
    for file in mp4_files:
        print(str(file)) 

    print('\n')
    print("List of distortions that will be applied to each video:")
    distortions_str = video_augmentation.get_transformations()
    for distortion in distortions_str:
        print(distortion)



    # Every information will be saved in this dictionary
    general_data = {
        'name': args.name,
        'distortions': distortions_str,
        'data': {}
    }


    # Getting points' data
    point_data = get_points_from_yaml(args.points)

    # Create a point data dictionary if it doesnt exist
    if point_data is None:
        for file in mp4_files:
            add_points_to_file(file, args.points)

        point_data = get_points_from_yaml(args.points)
        if point_data is None:
            raise FileExistsError("Couldn't create point yaml file")

    
    for file in mp4_files:
            
        txt_file = "yolo_" + file.stem + ".txt"
        txt_path = DEFAULT_ANNOTATION_DIR / txt_file

        # Check if there is already some annotation to this file
        if txt_path.exists() and not args.force:
            print(f"The video {file} already has an annotation. If you want to change the annotation, set --force to True")
            print("Skipping video prediction")

        else:
            print(f"Annotations will be saved in {txt_path}")

            predict(file=file,
                    point_data=point_data,
                    args=args)

            print("Annotations saved!!!\n\n")

            if txt_path.exists():
                general_data['data'][str(file)] = str(txt_path)
            else:
                raise FileExistsError(f"Couldn't create annotation file for this video {str(file)}")
            
            print(f"Applying distortions to video {file}")

        for distortion in distortions_str:

            distortion_path = apply_distortion(file=file,
                                               distortion=distortion,
                                               args=args)
            
            general_data['data'][str(distortion_path)] = str(txt_path)


    with open(str(DEFAULT_DATA_DIR / f'{args.name}.yaml'), "w") as yaml_file:
        yaml.dump(general_data, yaml_file, default_flow_style=False)

    print("\n\n\nGenerated YAML:\n")
    print(yaml.dump(general_data))


if __name__ == '__main__':
    generate()