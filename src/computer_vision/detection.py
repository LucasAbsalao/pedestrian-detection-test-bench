from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(data='coco8.yaml', imgsz=640, epochs=50, device=0)

path = model.export(format='onnx', simplify=True, half=True, device = 0)