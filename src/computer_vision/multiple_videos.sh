#!/bin/bash

# Exit immediately if some command returns a non-zero status
set -e

echo "========================================================="
echo "🚀 Starting Prediction of Multiple Videos"
echo "========================================================="

videos=("GoLive1" "GoLive2" "GoLive3" "GoLive4" "GoLive5" "GoLive6")

for video in "${videos[@]}"; do
    echo -e "\n\n Predictiong video $video: \n"

    python3 predict_video.py \
        --model_path yolo26x.pt \
        --video /app/videos/${video}.mp4 \
        --name ${video}_x \
        --save True \
        --save_txt True \
        --show False 

done

echo "========================================================="
echo "✅ All predictions completed successfully!"
echo "========================================================="