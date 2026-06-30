# UGE
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