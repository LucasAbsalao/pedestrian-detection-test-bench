import cv2
import argparse
from pathlib import Path
import sys
from ultralytics import YOLO
from .config import PREDICT_DIR, ANNOTATION_DIR
import computer_vision.utils.draw as draw
import computer_vision.utils.video as utilv

class VideoAnnotator:
    def __init__(self, 
                 video : Path,
                 point_d : tuple[int, int] ,
                 point_u : tuple[int, int] ,
                 model_path : Path | str = 'yolo26n.pt', 
                 predict_output_path : Path = PREDICT_DIR,
                 name : str = "yolo26_video",
                 classes : list[int] | None = [0], 
                 stream : bool = True,
                 save : bool = True,
                 save_txt_yolo : bool = False,
                 show : bool = True,
                 txt_output_path : Path = ANNOTATION_DIR,
                 ):
        
        self.video = video.resolve()
        self.name = name
        self.txt_output_path = txt_output_path.resolve()

        self.point_d = point_d
        self.point_u = point_u


        self.model = YOLO(str(model_path))
        self.project_path = predict_output_path.resolve()
        self.classes = classes
        self.stream = stream
        self.save = save
        self.save_txt = save_txt_yolo
        self.show = show

        w, h, fps = utilv.get_video_parameters(self.video)
        self.width = int(w)
        self.height = int(h)
        self.fps = fps

        self.trapezes = draw.generate_trapezes(point_d=point_d,
                                               point_u=point_u,
                                               width=self.width)


    def predict(self):

        print("Starting prediction of video: ")
        print(self.video)
        cap = cv2.VideoCapture(self.video)
    
        assert cap.isOpened(), "Error reading video file"
    
        video_writer = cv2.VideoWriter(str(self.project_path / f"{self.name}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width,self.height))
    
        yolo_kwargs = {
            "classes": self.classes,
            "project": str(self.project_path),
            "name": self.name,
            "stream": self.stream,
            "save": self.save,
            "save_txt": self.save_txt,
            "save_conf": self.save_txt,
            "show": self.show,
            "verbose": False
        }
    
        count_frames = 0

        if self.txt_output_path.suffix:
            annotation_file_path = self.txt_output_path
        else:
            self.txt_output_path.mkdir(exist_ok=True, parents=True)
            annotation_file_path = self.txt_output_path / f"yolo_{self.name}.txt"
    
        with open(annotation_file_path, "w") as f:
            while cap.isOpened():
                success, im0 = cap.read()
    
                if not success:
                    print("Video frame is empty or processing is complete.")
                    break
    
                results = self.model(source = im0, **yolo_kwargs)
    
                # print("Results Obtained: ")
                f.write(f"{count_frames}\n")
                for r in results:
                    # print("------------------------------------- Bounding Boxes -------------------------------------")
    
                    im0 = draw.write_lines(im0, trapezes=self.trapezes)
                    
                    zone = draw.draw_bbox(image = im0, bbox = r.boxes.xyxy.cpu().numpy(), trapezes = self.trapezes, rectangle=True)
    
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
    
        if self.show:
            cv2.destroyAllWindows()