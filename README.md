# UGE

To adjust detections, it's recommended to use CVAT.
How it works:
- Create an account and use the local installation
- Use sudo docker compose up -d to run cvat in localhost detached, i.e. It will be running on background
- Open localhost on port 8080 (That's the default port for running but you can change it if needed)
- Click in the + button, then in create new project, choose a name for this file and press in submit and open
- Click on the new + button and go to create new task.
- Give a name to your task and upload the video that you want to adjust. Then, press submit & open 



docker build --build-arg DEV=true -t yolo:1.1 .

X11
xhost +local:docker && docker run --name yolo_detection \
--device /dev/video0 \
-v $(pwd):/app \
-e DISPLAY=$DISPLAY \
-e QT_X11_NO_MITSHM=1 \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v ~/.Xauthority:/root/.Xauthority:rw \
--net=host \
--device nvidia.com/gpu=all \
-it --ipc=host --rm \
yolo:1.3


ctrl+d to exit or ctrl+p and ctrl+q to let it running.

docker start yolo_test

docker exec -it yolo_test /bin/bash


## TODO
In the actual project, just frames with detection are saved.
add sampling to the latency recall.


Exception wrapper to check if a point_data is None
Reduce size of yaml file saving points and annotations files separatedly. Use a key to associate each file with it's points
Put np nan to recall where there is no positive occurence