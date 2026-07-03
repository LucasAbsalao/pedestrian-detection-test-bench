import sys
import argparse
import yaml
from pathlib import Path

import lines_prediction
import video_augmentation


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_DATA_DIR = ROOT_DIR / 'data'
DEFAULT_ANNOTATION_DIR = DEFAULT_DATA_DIR / 'annotations'
DEFAULT_VIDEO_DIR = DEFAULT_DATA_DIR / 'videos'
DEFAULT_DISTORTION_DIR = DEFAULT_VIDEO_DIR / 'distortions'


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
        "--force",
        type=bool,
        default=True,
        help="If set, erases any precedent annotation file for these videos."
    )
    return parser.parse_args()


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

    general_data = {
        'name': args.name,
        'distortions': distortions_str,
        'data': {}
    }

    for file in mp4_files:
        txt_file = "yolo_" + file.stem + ".txt"
        txt_path = DEFAULT_ANNOTATION_DIR / txt_file

        # Check if there is already some annotation to this file
        if txt_path.exists() and not args.force:
            print(f"The video {file} already has an annotation. If you want to change the annotation, set --force to True")
            print("Skipping video")
            continue

        print(f"Annotations will be saved in {txt_path}")


        predict_args = [
            '--model_path', 'models/yolo26x.pt',
            '--video', str(file),
            '--project', 'predict',
            '--name', file.stem,
            '--stream', 'False',
            '--save', 'False',
            '--save_txt', 'False',
            '--show', 'False',   
            '--txt-path', str(DEFAULT_ANNOTATION_DIR),
            '--point-d', '400', '1000', 
            '--point-u', '550', '600'   
        ]
        lines_prediction.predict(predict_args) 

        print("Annotations saved!!!\n\n")

        if txt_path.exists():
            general_data['data'][str(file)] = str(txt_path)
        else:
            raise FileExistsError(f"Couldn't create annotation file for this video {str(file)}")
        
        print(f"Applying distortions to video {file}")

        for distortion in distortions_str:

            distortion_video_name = file.stem + '_' + distortion
            print('='*15 + distortion + '='*15)
            
            distortion_args = [
                "--video", str(file),
                "--dest", str(DEFAULT_DISTORTION_DIR),
                "--name", distortion_video_name,
                "--distortion", distortion
            ]
            video_augmentation.transform(distortion_args)
            
            distortion_path = DEFAULT_DISTORTION_DIR / f"{distortion_video_name}.mp4"
            general_data['data'][str(distortion_path)] = str(txt_path)




    with open(str(DEFAULT_DATA_DIR / f'data_{args.name}.yaml'), "w") as yaml_file:
        yaml.dump(general_data, yaml_file, default_flow_style=False)

    print("\n\n\nGenerated YAML:\n")
    print(yaml.dump(general_data))


if __name__ == '__main__':
    generate()