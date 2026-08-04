'''
To execute:
python3 zone_counter_w_alarm.py \
    --video /home/lucas/Documents/computer_vision/data/videos/GX010079_01_4.mp4 \
    --predictions /home/lucas/Documents/computer_vision/data/annotations/yolo_GX010079_01_4.txt \
    --point-d 245 2028     --point-u 1244 653 \
    --alarm blaxtair_real

    python3 zone_counter_w_alarm.py \
    --video /home/lucas/Documents/computer_vision/data/videos/distortions/GX010079_01_4_gaussian_noise.mp4 \
    --predictions /home/lucas/Documents/computer_vision/data/annotations/yolo_GX010079_01_4.txt \
    --point-d 245 2028     --point-u 1244 653 \
    --alarm blaxtair_real
'''

import time
import cv2
import csv
import yaml
import argparse
from pathlib import Path
import sys
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pyaudio
import threading

ROOT_DIRECTORY = Path(__file__).resolve().parents[3]
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
    
from utils.draw import write_lines_from_points, draw_bboxes_from_data
from core.audio_handler import AudioHandler
from utils.transformations import resize
from benchmark.stats import Stats

DEFAULT_CSV_FILE = Path(__file__).resolve().parents[0] / "results.csv"

WAVE_OUTPUT_PATH = ROOT_DIRECTORY / "calibrate"
ALARM_CONFIG_PATH = SRC_DIR / "config" / "alarm.yaml"

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

def get_alarm_frequency(alarm_name : str, filepath : Path):
    if filepath.exists():
        with open(filepath, "r") as yaml_file:
            data_alarm = yaml.safe_load(yaml_file) or {}
        if data_alarm:
            if alarm_name in list(data_alarm.keys()):
                print("Alarm found: ")
                print(data_alarm[alarm_name])
                return data_alarm[alarm_name]['frequency'], data_alarm[alarm_name]['amplitude']

    return None, None

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
    parser.add_argument(
        "--alarm",
        type=str,
        required=True,
        help="Name of the alarm. This will be used to search for the alarm detection frequency."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ALARM_CONFIG_PATH,
        help="PAth to file with alarm frequencies."
    )
    return parser.parse_args(arg_list) 

def continuous_morphological_closing(detection_array : NDArray, closing_se_size : int):
    assert closing_se_size%2==1, "closing_se_size has to be odd"

    closed_detection_array = np.copy(detection_array)
    min_value = np.min(detection_array)
    max_value = np.max(detection_array)

    pad = int(closing_se_size // 2)

    #Dilate
    detection_array_padded_min = np.pad(detection_array, (pad, pad), mode='constant', constant_values=min_value)

    for i in range(len(detection_array)):
        closed_detection_array[i] = np.max(detection_array_padded_min[i:i+closing_se_size])

    #Erode
    detection_array_padded_max = np.pad(closed_detection_array, (pad, pad), mode='constant', constant_values=max_value)
    
    for i in range(len(detection_array)):
        closed_detection_array[i] = np.min(detection_array_padded_max[i:i+closing_se_size])

    return closed_detection_array

def count_zones(arg_list = None):
    args = parse_args(arg_list)

    close_detection_gaps = 20 # Corresponds to how many frames the system can ignore to consider a single detection extract
    window = 60
    
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

    video_seconds = total_frames/fps

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
    ground_truth = np.full((total_frames), 3, dtype=int)
    detected = np.zeros((total_frames,), dtype=int)
    frame_time = np.zeros((total_frames,), dtype=float)
    
    frame_idx = 0
    
    print(f"Running video in real time (~{fps:.1f} FPS, delay={delay}ms)")
    print("Hold or press 'd' when a person is inside the trapezoids, 'q' to quit")
    
    # ------------------------------------------ Main Loop ------------------------------------------
    cv2.namedWindow("Zone Counter", cv2.WND_PROP_FULLSCREEN)

    cv2.waitKey(100)
    cv2.setWindowProperty("Zone Counter",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


    start_event = threading.Event()
    stop_event = threading.Event()

    thread_audio = threading.Thread(target=ah.record_audio_async, args=(start_event, stop_event))
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
            frame = write_lines_from_points(frame, point_d, point_u, w)
            frame = draw_bboxes_from_data(frame, frame_data)
        
        # Display current metrics overlay
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Min Zone: {min_zone}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        resized_frame = resize(frame, width=5120, height=2160)
        cv2.imshow('Zone Counter', resized_frame)
        frame_time[frame_idx] = time.perf_counter()
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            detected[frame_idx] = 1
            
        frame_idx += 1

    stop_event.set()
    thread_audio.join()

    end_time = time.perf_counter()


    ah.terminate()


    cap.release()
    cv2.destroyAllWindows()

    # ------------------------------------- Closing Ground Truth Holes -------------------------------------

    closed_ground_truth = continuous_morphological_closing(3-ground_truth, 21)
    closed_ground_truth = (3-closed_ground_truth).astype(int)

    plt.subplot(1,2,1)
    plt.plot(3-ground_truth, color='red')
    plt.title("Original Grount Truth")
    
    plt.subplot(1,2,2)
    plt.plot(3-closed_ground_truth, color='blue')
    plt.title("Grount Truth After Morphological Closing")
    plt.show()
    

    # ------------------------------------- Get Audio Detection -------------------------------------
    audio_data = ah.get_audio_stream_data()

    audio_seconds = len(audio_data) / RATE

    frequency, t, dbs = ah.spectrogram(audio_data=audio_data,
                                      seconds=audio_seconds)

    ah.save_audio(output_path=str(WAVE_OUTPUT_PATH / "detection_audio.wav"))

    ah.plot_spectrogram(frequency=frequency,
                        time_stamps=t,
                        spectrogram=dbs,
                        filepath=str(WAVE_OUTPUT_PATH / "spectrogram_detection.jpg"))

    alarm_frequency, amplitude_threshold = get_alarm_frequency(args.alarm, args.config)

    if alarm_frequency is None:
        raise RuntimeError("Could not get alarm frequency from yaml file.")
    if amplitude_threshold is None:
        raise RuntimeError("Could not get alarm amplitude threshold from yaml file.")

    binary_detection = ah.get_binary_detection(audio_data=audio_data,
                                                alarm_frequency=alarm_frequency,
                                                amp_threshold=amplitude_threshold,
                                                interval=0.05,
                                                seconds=audio_seconds)

    final_detection = ah.morph_closing(binary_detection=binary_detection,
                                       struct_size=close_detection_gaps)

    detected_audio_video = ah.resample_detection(frame_time, final_detection)

    plt.subplot(1,2,1)
    plt.title("Audio detection")
    plt.plot(final_detection, color='red')

    plt.subplot(1,2,2)
    plt.title("Audio detection after resampling")
    plt.plot(detected_audio_video, color='red')
    plt.plot(3-closed_ground_truth, color='blue')
    plt.show()


    # ------------------------------------- Generate Statistics -------------------------------------
    
    detected = detected_audio_video
    general_statistics = Stats(general = True)
    statistics_per_zone = {
        'red': Stats(),
        'orange': Stats(),
        'green': Stats()
    }

    for i in range(total_frames):
        statistics_per_zone["green"].update(expected = closed_ground_truth[i] == 2, predicted = detected[i]==1)
        statistics_per_zone["orange"].update(expected = closed_ground_truth[i] == 1, predicted = detected[i]==1)
        statistics_per_zone["red"].update(expected = closed_ground_truth[i] == 0, predicted = detected[i]==1)
        general_statistics.update(expected=closed_ground_truth[i]<3, predicted = detected[i]==1, weight_idx=2-closed_ground_truth[i])
    
    # Calculate performance metrics
    processed_frames = general_statistics.total_evaluations()
    assert processed_frames == total_frames, "Problem retrieving information from the video -> processed frames are different than total frames"

    

    minimum_weight = general_statistics.minimum_weight
    binary_ground_truth = (closed_ground_truth < 3).astype(int)

    time_stamps = general_statistics.get_detection_duration_in_frames(ground_truth=binary_ground_truth,
                                                                      closing_se_size=close_detection_gaps)
                                                            
    class_la_array = general_statistics.latency_array(ground_truth = binary_ground_truth,
                                                      predictions = detected,
                                                      time_stamps = time_stamps)
    
    la_recall = general_statistics.calculate_latency_recall(ground_truth = binary_ground_truth,
                                                            predictions = detected,
                                                            closing_structure_size=close_detection_gaps)
    

    plt.plot(class_la_array)
    plt.title("Latency Recall Evaluation")
    plt.vlines(time_stamps, 0, 1 + minimum_weight, linestyles='dashed')
    plt.show()

    frame_delay = general_statistics.get_frame_delay(b_video_detection=binary_ground_truth,
                                                     b_audio_detection=detected_audio_video,
                                                     window = window,
                                                     closing_se_size=close_detection_gaps)

    second_delay = general_statistics.get_seconds_delay(b_video_detection=binary_ground_truth,
                                                        frame_seconds=frame_time,
                                                        audio_detection=final_detection,
                                                        audio_duration=audio_seconds,
                                                        audio_start_timestamp=ah.record_start_time,
                                                        window=window,
                                                        closing_se_size=close_detection_gaps)

    print("Frame Delay: ", frame_delay)
    print("Second Delay: ", second_delay)
    if len(frame_delay) != len(second_delay):
        raise RuntimeError("Quantity of frame delays and seconds delays are different, what shouldn't happen")
    
    video_duration = total_frames/fps
    general_statistics.calculate_rfa(video_duration)

    print("\n" + "="*30 + " EVALUATION REPORT " + "="*30)
    print(f"Original Video Duration (seconds): {video_seconds}")
    print(f"Elapsed Time (seconds): {end_time - initial_time}")
    print(f"Video started at {initial_time}")
    print(f"Video finishes at {end_time}")
    print(f"Total Frames Evaluated: {processed_frames}")


    print("-"*40 + " General Statistics " + "-"*40)
    print(general_statistics)

    print("-"*40 + " Green Statistics " + "-"*40)
    print(statistics_per_zone['green'])

    print("-"*40 + " Orange Statistics " + "-"*40)
    print(statistics_per_zone['orange'])

    print("-"*40 + " Red Statistics " + "-"*40)
    print(statistics_per_zone['red'])


    plt.plot(3-closed_ground_truth)
    plt.plot(detected)
    plt.grid()
    plt.show()

    results = {
        'Name': str(video_path).split('/')[-1],
        'Accuracy': general_statistics.calculate_accuracy(),
        'Precision': general_statistics.calculate_precision(),
        'Recall': general_statistics.calculate_recall(),
        'F1 Score': general_statistics.calculate_f1_score(),
        'Latency Recall': general_statistics.la_recall,
        'Weighted Recall': general_statistics.calculate_weighted_recall(),
        'P_miss': general_statistics.calculate_pmiss(),
        'P_false_alarm': general_statistics.calculate_pfa(),
        'R_false_alarm': general_statistics.r_fa,
        'NDCR': general_statistics.calculate_ndcr(),
        'Frame_Delay': general_statistics.frame_delay,
        "Seconds_Delay": general_statistics.seconds_delay,
        'Red Recall': statistics_per_zone['red'].calculate_recall(),
        'Orange Recall': statistics_per_zone['orange'].calculate_recall(),
        'Green Recall': statistics_per_zone['green'].calculate_recall()
    }
    csv_file_path = args.csv.resolve()
    delay_csv_file_path = csv_file_path.with_stem(f"{csv_file_path.stem}_delay")
    delay_csv_file_path = delay_csv_file_path.resolve()


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

    # ------------------------------------ DELAY ------------------------------------

    delay_stats = []
    print("frame delay", frame_delay)
    for i in range(len(frame_delay)):
        delay_dict = {
            'name': video_path.stem,
            'frame': frame_delay[i],
            'second': second_delay[i]
        }
        delay_stats.append(delay_dict)

    if delay_csv_file_path.exists():
        with open(delay_csv_file_path, "a", newline='') as csvfile:
            print("Adding delays to existing file...")
            writer = csv.DictWriter(csvfile, fieldnames=list(delay_stats[0].keys()))
            writer.writerows(delay_stats)
    else:
        print("Creating new csv file for storing delay")
        with open(delay_csv_file_path, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(delay_stats[0].keys()))
            writer.writeheader()
            writer.writerows(delay_stats)
        

if __name__ == "__main__":
    count_zones()