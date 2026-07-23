'''
To execute:
python3 zone_counter_w_alarm.py \
    --video /home/lucas/Documents/computer_vision/videos/marcher_180.mp4 \
    --predictions /home/lucas/Documents/computer_vision/data/annotations/yolo_marcher_180_n.txt \
    --point-d 490 840     --point-u 627 656
'''

import time
import cv2
import csv
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import threading

ROOT_DIRECTORY = Path(__file__).resolve().parents[3]
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
    
from utils.draw import write_lines, draw_bboxes_from_data
from utils.audio_handler import AudioHandler
from benchmark.stats import Stats

DEFAULT_CSV_FILE = Path(__file__).resolve().parents[0] / "results.csv"

WAVE_OUTPUT_PATH = ROOT_DIRECTORY / "calibrate" / "alarm.wav"

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


def parse_predictions(txt_path: Path) -> dict:
    """Parse the predictions txt file into a dict: frame_num -> list of (zone, bbox)."""
    predictions = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    current_frame = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 1:
            current_frame = int(parts[0])
            predictions[current_frame] = []
        elif len(parts) >= 5:
            zone = int(parts[0])
            bbox = tuple(map(float, parts[1:5]))
            if current_frame is not None:
                predictions[current_frame].append((zone, bbox))
    return predictions


def str2bool(v: str | bool) -> bool:
    """Auxiliar function for argparse to read a boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_args(arg_list = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count 'd' key presses when min zone < 3"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Video file to display." 
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the predictions txt file from lines_prediction.py"
    )
    parser.add_argument(
        "--point-d",
        type=int,
        nargs=2,
        default=[400, 1000],
        help="Lower trapezoid point (x y)"
    )
    parser.add_argument(
        "--point-u",
        type=int,
        nargs=2,
        default=[550, 600],
        help="Upper trapezoid point (x y)"
    )
    parser.add_argument(
        "--show",
        type=str2bool,
        default=True,
        help="Display the video with overlay"
    )
    parser.add_argument(
        "--draw",
        type=bool,
        default=False,
        help="If set, it shows bounding boxes, lines and zones in the image"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_FILE,
        help="Path to the csv file with all metrics obtained from this video"
    )
    return parser.parse_args(arg_list) 

def count_zones(arg_list = None):
    args = parse_args(arg_list)
    
    video_path = args.video.resolve()
    predictions_path = args.predictions.resolve()
    
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)
    if not predictions_path.exists():
        print(f"Predictions file not found: {predictions_path}")
        sys.exit(1)
    
    predictions = parse_predictions(predictions_path)
    print(f"Loaded predictions for {len(predictions)} frames")
    
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), "Error reading video file"
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    seconds = total_frames/fps

    # Audio Handler
    ah = AudioHandler(chunk=CHUNK,
                      format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE) 
    
    point_d = tuple(args.point_d)
    point_u = tuple(args.point_u)
    
    # Calculate adaptive delay based on FPS
    delay = max(1, int(1000 / (fps if fps > 0 else 30)))
    
    # Vectors for storing detections and ground truths
    ground_truth = np.zeros((total_frames), dtype=int)
    detected = np.zeros((total_frames,), dtype=int)
    
    frame_idx = 0
    
    print(f"Running video in real time (~{fps:.1f} FPS, delay={delay}ms)")
    print("Hold or press 'd' when a person is inside the trapezoids, 'q' to quit")
    
    # ------------------------------------------ Main Loop ------------------------------------------
    cv2.namedWindow("Zone Counter", cv2.WND_PROP_FULLSCREEN)

    cv2.waitKey(100)
    cv2.setWindowProperty("Zone Counter",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


    start_event = threading.Event()
    stop_event = threading.Event()

    thread_audio = threading.Thread(target=ah.record_audio, args=(seconds,))
    thread_audio.start()

    start_event.set()

    initial_time = time.perf_counter()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video")
            break
        
        # Get ground truth min_zone
        frame_data = predictions.get(frame_idx, [])
        min_zone = min((zone for zone, _ in frame_data), default=3)
        ground_truth[frame_idx] = min_zone
        
        if args.draw:
            frame = write_lines(frame, point_d, point_u, w)
            frame = draw_bboxes_from_data(frame, frame_data)
        
        # Display current metrics overlay
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Min Zone: {min_zone}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Zone Counter', frame)
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            detected[frame_idx] = 1
            
        frame_idx += 1

    stop_event.set()
    thread_audio.join()


    ah.terminate()

    end_time = time.perf_counter()

    cap.release()
    cv2.destroyAllWindows()

    # ------------------------------------- Get Audio Detection -------------------------------------
    audio_data = ah.get_audio_stream_data()

    frequency, t, dbs = ah.spectrogram(audio_data=audio_data,
                                      seconds=seconds)

    ah.save_audio(output_path=str(WAVE_OUTPUT_PATH.parent / "detection_audio.wav"))

    ah.plot_spectrogram(frequency=frequency,
                        time_stamps=t,
                        spectrogram=dbs,
                        filepath=str(WAVE_OUTPUT_PATH.parent / "spectrogram_detection.jpg"))

    alarm_frequency = 5904.290909090909 # TODO

    binary_detection = ah.get_binary_detection(audio_data=audio_data,
                                                alarm_frequency=alarm_frequency,
                                                amp_threshold=1,
                                                interval=0.05,
                                                seconds=seconds)

    final_detection = ah.morph_closing(binary_detection=binary_detection,
                                       struct_size=10)

    plt.plot(final_detection, color='red')
    plt.plot(ground_truth, color='blue')
    plt.show()


    # ------------------------------------- Generate Statistics -------------------------------------

    general_statistics = Stats(general = True)
    statistics_per_zone = {
        'red': Stats(),
        'orange': Stats(),
        'green': Stats()
    }

    for i in range(total_frames):
        statistics_per_zone["green"].update(expected = ground_truth[i] == 2, predicted = detected[i]==1)
        statistics_per_zone["orange"].update(expected = ground_truth[i] == 1, predicted = detected[i]==1)
        statistics_per_zone["red"].update(expected = ground_truth[i] == 0, predicted = detected[i]==1)
        general_statistics.update(expected=ground_truth[i]<3, predicted = detected[i]==1, weight_idx=2-ground_truth[i])
    
    # Calculate performance metrics
    processed_frames = general_statistics.total_evaluations()
    assert processed_frames == total_frames, "Problem retrieving information from the video -> processed frames are different than total frames"

    t_start, t_end = -1, -1
    for i in range(total_frames):
        if ground_truth[i] < 3 and t_start == -1:
            t_start = i

        if ground_truth[total_frames - i - 1] < 3 and t_end == -1:
            t_end = total_frames - i - 1

        if t_start!=-1 and t_end!=-1:
            break

    if t_start == -1 and t_end == -1:
        print("There is no detection of a person in the ground truth")

    minimum_weight = 0.2
    print(f"For this video the detection starts in frame {t_start} and finishes in frame {t_end}")
    
    class_la_array = general_statistics.latency_array(ground_truth = ground_truth,
                                                      predictions = detected,
                                                      t_start = t_start,
                                                      t_end = t_end)
    
    la_recall = general_statistics.calculate_latency_recall(ground_truth = ground_truth,
                                                                  predictions = detected,
                                                                  t_start = t_start,
                                                                  t_end = t_end)
    

    plt.plot(class_la_array)
    plt.title("Latency Recall Evaluation")
    plt.vlines([t_start, t_end], 0, 1 + minimum_weight, linestyles='dashed')
    plt.show()
    
    video_duration = int(total_frames//fps)
    general_statistics.calculate_rfa(video_duration)

    print("\n" + "="*30 + " EVALUATION REPORT " + "="*30)
    print(f"Elapsed Time: {end_time - initial_time}")
    print(f"Total Frames Evaluated: {processed_frames}")


    print("-"*40 + " General Statistics " + "-"*40)
    print(general_statistics)

    print("-"*40 + " Green Statistics " + "-"*40)
    print(statistics_per_zone['green'])

    print("-"*40 + " Orange Statistics " + "-"*40)
    print(statistics_per_zone['orange'])

    print("-"*40 + " Red Statistics " + "-"*40)
    print(statistics_per_zone['red'])


    plt.plot(3-ground_truth)
    plt.plot(detected)
    plt.grid()
    plt.show()

    results = {
        'Name': str(video_path).split('/')[-1],
        'Accuracy': general_statistics.calculate_accuracy(),
        'Precision': general_statistics.calculate_precision(),
        'Recall':general_statistics.calculate_recall(),
        'Latency Recall': general_statistics.la_recall,
        'Weighted Recall': general_statistics.calculate_weighted_recall(),
        'P_miss': general_statistics.calculate_pmiss(),
        'P_false_alarm': general_statistics.calculate_pfa(),
        'R_false_alarm': general_statistics.r_fa,
        'NDCR': general_statistics.calculate_ndcr(),
        'Red Recall': statistics_per_zone['red'].calculate_recall(),
        'Orange Recall': statistics_per_zone['orange'].calculate_recall(),
        'Green Recall': statistics_per_zone['green'].calculate_recall()
    }
    csv_file_path = args.csv.resolve()

    if csv_file_path.exists():
        with open(csv_file_path, "a", newline='') as csvf:
            print("Adding results to existing file.")
            writer = csv.DictWriter(csvf, fieldnames=list(results.keys()))
            writer.writerow(results)
    else:
        print("Creating new csv results file")
        with open(csv_file_path, "w", newline='') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)

if __name__ == "__main__":
    count_zones()