import os
import json
from pathlib import Path
import subprocess

from core.config import FRAMES_DIR, PREDICT_DIR, VIDEO_DIR

CLASSES = ["person"] 

IMG_WIDTH = 1920   
IMG_HEIGHT = 1080 

OUTPUT_JSON = "./predictions.json" # Output file name

# If uploading frames manually via web interface, 
# leave only the filename. 
# If using Local Storage in Label Studio, adjust the corresponding path.
IMAGE_URL_PREFIX = "/data/local-files/?d=meus_frames/" 
# ==========================================

def separate_frames(video_path : Path):
    output_path = FRAMES_DIR / video_path.stem
    output_path.mkdir(exist_ok=True, parents=True)
    separate_frames_command = f'''ffmpeg -i {video_path} -qscale:v 2 "{FRAMES_DIR}/{video_path.stem}/{video_path.stem}-%04d.jpg"'''
    proc = subprocess.run(separate_frames_command, check=True)


name = "GX020083_00_2.mp4"

yolo_dir = PREDICT_DIR / name
video_path = VIDEO_DIR / name

separate_frames(video_path)

image_url = FRAMES_DIR / ""
label_studio_json = []

for txt_filename in os.listdir(yolo_dir):
    if not txt_filename.endswith(".txt"):
        continue
        
    # Assume image shares the same name as the txt file, but with a .jpg extension
    image_filename = txt_filename.replace(".txt", ".jpg")
    txt_path = os.path.join(yolo_dir, txt_filename)
    
    results = []
    with open(txt_path, "r") as f:
        for line_number, line in enumerate(f.readlines()):
            parts = line.strip().split()
            
            # Ignore empty or poorly formatted lines
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Mathematical conversion: YOLO (center, 0-1) to Label Studio (top-left corner, 0-100)
            ls_width = width * 100
            ls_height = height * 100
            ls_x = (x_center - width / 2) * 100
            ls_y = (y_center - height / 2) * 100
            
            # Get the class name or use the ID as a string if not in the list
            class_name = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
            
            result_item = {
                "original_width": IMG_WIDTH,
                "original_height": IMG_HEIGHT,
                "image_rotation": 0,
                "value": {
                    "x": ls_x,
                    "y": ls_y,
                    "width": ls_width,
                    "height": ls_height,
                    "rotation": 0,
                    "rectanglelabels": [class_name]
                },
                "id": f"bbox_{line_number}",
                "from_name": "label",      # Must match the XML of your Label Studio project
                "to_name": "image",        # Must match the XML of your Label Studio project
                "type": "rectanglelabels"
            }
            results.append(result_item)
            
    # Create task for this specific frame
    task = {
        "data": {
            "image": IMAGE_URL_PREFIX + image_filename
        },
        "predictions": [{
            "result": results
        }]
    }
    label_studio_json.append(task)

# Save final JSON file
with open(OUTPUT_JSON, "w") as f:
    json.dump(label_studio_json, f, indent=2)

print(f"✅ Conversion complete! File saved as: {OUTPUT_JSON}")
print(f"Processed {len(label_studio_json)} frames.")