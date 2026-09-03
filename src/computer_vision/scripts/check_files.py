from pathlib import Path
from computer_vision.core.config import VIDEO_DIR, ANNOTATION_DIR


mp4_files = list(VIDEO_DIR.glob('*.mp4'))
mp4_names = [vid.stem for vid in mp4_files]

annotation_files = list(ANNOTATION_DIR.glob("*.txt"))
annotation_names = [annot.stem.strip('yolo_') for annot in annotation_files]

for annotation in annotation_names:
    if annotation not in mp4_names:
        print(annotation)