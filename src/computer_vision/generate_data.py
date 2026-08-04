import sys
import argparse
import yaml
from pathlib import Path

from core.config import DEFAULT_YOLO_MODEL, POINTS_CONFIG, ANNOTATION_DIR, DISTORTION_DIR, VIDEO_DIR, DATA_DIR

from core.annotation import VideoAnnotator
from core.augmentation import DistortionHandler, DistortionType
from core.trapezoid import TrapezoidMarker


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
        default = str(DEFAULT_YOLO_MODEL),
        help = 'Model used to annotate images.'
    )
    parser.add_argument(
        '--points',
        type=Path,
        default = POINTS_CONFIG,
        help="Path to the yaml file storing points used for prioritized zone detection"
    )
    parser.add_argument(
        "--force",
        type=bool,
        default=False,
        help="If set, erases any precedent annotation file for these videos."
    )
    return parser.parse_args()


def get_points_from_yaml(points_file_path : Path) -> dict :
    if points_file_path.exists():
        with open(str(points_file_path), 'r') as yaml_file:
            point_data = yaml.safe_load(yaml_file)
        return point_data 
    else:
        return {}


def add_points_to_file(file : Path, yaml_path : Path):
    print(f"Adding point to {file}")

    trapezoid_marker = TrapezoidMarker(video_path = file, config = yaml_path.parent, name = yaml_path.stem)
    trapezoid_marker.mark_trapezoids()


def annotate(file : Path, point_data : dict , point_path : Path, model : str, txt_path : Path):

    # Check if this file has a point associated to it
    if str(file) not in point_data:
        add_points_to_file(file, point_path)
        point_data = get_points_from_yaml(point_path) or {}
        if point_data is None:
            raise FileNotFoundError("Couldn't create yaml file")
    
    points_xyxy = point_data[str(file)]

    annotator = VideoAnnotator(video = file,
                               point_d = (points_xyxy[0], points_xyxy[1]),
                               point_u = (points_xyxy[2], points_xyxy[3]),
                               model_path = model,
                               name = file.stem,
                               stream = False,
                               save = False,
                               save_txt_yolo = True,
                               show=False,
                               txt_output_path=txt_path
                               )
    
    annotator.predict()

def apply_distortion(distortion_handler : DistortionHandler, file : Path, distortion : str, force : bool) -> Path:
    distortion_video_name = file.stem + '_' + distortion

    distortion_path = distortion_handler.output_path / f"{distortion_video_name}.mp4"
    if distortion_path.exists() and not force:
        return distortion_path

    print('='*15 + distortion + '='*15)

    distortion_handler.transform(distortion = DistortionType(distortion), name = distortion_video_name)

    return distortion_path
    

def generate():

    args = parse_args()

    print('\n\n')
    print("-"*30 + " STARTING DATASET GENERATION " + "-"*30)
    video_dir = VIDEO_DIR.resolve()

    mp4_files = list(video_dir.glob("*.mp4"))
    print("List of videos to analyse:")
    for file in mp4_files:
        print(str(file)) 

    print('\n')
    print("List of distortions that will be applied to each video:")
    distortions_str = DistortionHandler.get_transformations()
    for distortion in distortions_str:
        print(distortion)

    # Every information will be saved in this dictionary
    general_data = {
        'name': args.name,
        'distortions': distortions_str,
        'data': {},
        'bbox_points':{}
    }

    point_path = args.points.resolve()


    # Getting points' data
    point_data = get_points_from_yaml(point_path)

    # Create a point data dictionary if it doesnt exist
    if not point_data:
        for file in mp4_files:
            add_points_to_file(file, point_path)
        point_data = get_points_from_yaml(point_path)
    # Check if every single video has its own points setted
    else:
        file_missing = False
        for file in mp4_files:
            if str(file) not in point_data:
                add_points_to_file(file, point_path)
                file_missing = True
        if file_missing:
            point_data = get_points_from_yaml(point_path)
                
    if not point_data:
        raise FileNotFoundError("Couldn't create point yaml file")
    

    for file in mp4_files:
            
        txt_file = "yolo_" + file.stem + ".txt"
        txt_path = ANNOTATION_DIR / txt_file

        # Check if there is already some annotation to this file
        if txt_path.exists() and not args.force:
            print(f"The video {file} already has an annotation. If you want to change the annotation, set --force to True")
            print("Skipping video prediction")

        else:
            print(f"Annotations will be saved in {txt_path}")

            annotate(file = file,
                    point_data = point_data,
                    point_path = point_path,
                    model = args.model,
                    txt_path = txt_path
                    )

            print("Annotations saved!!!\n\n")

        if txt_path.exists():
            general_data['data'][str(file)] = str(txt_path)
            general_data['bbox_points'][str(file)] = list(point_data[str(file)])
        else:
            raise FileNotFoundError(f"Couldn't create annotation file for this video {str(file)}")
            
        print(f"Applying distortions to video {file}")

        distortion_handler = DistortionHandler(video = file, codec = 'libx265_rawvideo')

        for distortion in distortions_str:

            distortion_path = apply_distortion(distortion_handler=distortion_handler,
                                               file=file,
                                               distortion=distortion,
                                               force=args.force)
            
            general_data['data'][str(distortion_path)] = str(txt_path)
            general_data['bbox_points'][str(distortion_path)] = list(point_data[str(file)])


    with open(str(DATA_DIR / f'{args.name}.yaml'), "w") as yaml_file:
        yaml.dump(general_data, yaml_file, default_flow_style=False)


if __name__ == '__main__':
    generate()