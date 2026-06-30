'''
To execute:
python3 zone_counter.py \
    --video /home/lucas/Documents/computer_vision/videos/marcher_180.mp4 \
    --predictions yolo_marcher_180_n.txt \
    --point-d 400 1000     --point-u 550 600
'''

import time
import cv2
import argparse
from pathlib import Path
import sys
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

ROOT_DIRECTORY = Path(__file__).resolve().parents[2]
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

from computer_vision.lines_prediction import generate_trapezes, write_lines


ZONE_COLORS = [(0, 0, 255), (0, 150, 255), (0, 255, 0), (255, 0, 0)]


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


def draw_bboxes(image, frame_data):
    """Draw bounding boxes on the image colored by zone."""
    for zone, bbox in frame_data:
        x1, y1, x2, y2 = map(int, bbox)
        color = ZONE_COLORS[min(zone, 3)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=3)
        label = f"Zone: {zone}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()

def calculate_la_recall(ground_truth : NDArray, predictions : NDArray, 
              t_start : int, t_end : int, 
              alpha : float, beta: float, 
              minimum_weight: float, sampling : int = 1):
    
    if sampling != 1:
        raise NotImplemented("Sampling different from one was not yet implemented")
    
    
    t = np.arange(len(ground_truth), step=sampling)

    if t_end == t_start:
        delta = np.zeros_like(t, dtype=float)
    else:
        delta = (t - t_start) / (t_end - t_start)

    
    gt_mask = (ground_truth < 3).astype(int) # Only the true detection can be counted
    s_delta = 1 - 1 / (1 + np.exp(-beta * (2*delta - 1)))*(1-minimum_weight)

    larec = gt_mask * predictions * s_delta #* np.pow(alpha, -t)

    return larec

class Stats:
    def __init__(self, general: bool = False) -> None:
        """
        Initializes the Stats object to track evaluation metrics.

        Sets up the base confusion matrix variables (TP, TN, FP, FN). 
        If configured for general purpose, it also initializes parameters 
        required for advanced metrics such as Weighted Recall (zone-based penalties) 
        and Latency Recall (time-decaying scoring).

        Args:
            general (bool, optional): If True, initializes variables and hyperparameters 
                for weighted and latency-aware metrics. Defaults to False.

        Attributes:
            tp (int): True Positives count.
            tn (int): True Negatives count.
            fp (int): False Positives count.
            fn (int): False Negatives count.
            general_purpose (bool): Flag indicating if advanced metrics are enabled.
            
            Attributes initialized only if general is True:
                weights (list[int]): Importance weights for different detection zones.
                weighted_tp (int): Accumulated weighted True Positives.
                weighted_fn (int): Accumulated weighted False Negatives.
                
                alpha (int): Exponential decay parameter for latency recall.
                beta (int): Steepness parameter for the Sigmoid time-decay function.
                minimum_weight (float): The minimum baseline score for a delayed detection.
                sampling (int): Frame sampling interval.
        """
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        

        self.general_purpose = general
        if general:
            self.weights = [1, 5, 10]
            self.weighted_tp = 0
            self.weighted_fn = 0

            self.alpha = 1
            self.beta = 4
            self.minimum_weight = 0.2
            self.sampling = 1
            self.la_recall = 0.
    
    def set_latency_parameters(self, alpha : float, beta: float, minimum_weight : float, sampling : int):
        self.alpha = alpha
        self.beta = beta
        self.minimum_weight = minimum_weight
        self.sampling = sampling

    def update(self, predicted : bool, expected : bool, weight_idx : int | None = None) -> None:
        if predicted and expected:
            self.tp += 1
        if predicted and not expected:
            self.fp += 1
        if not predicted and expected:
            self.fn += 1
        if not predicted and not expected:
            self.tn += 1

        if weight_idx is not None:
            if not self.general_purpose:
                print("WARNING: Since this stats class is not for general purposes, weights will be disconsidered!")
                self.weights = [1]
                self.update_with_weights(predicted = predicted, expected = expected, weight_idx = 0)
            else:
                self.update_with_weights(predicted = predicted, expected = expected, weight_idx = weight_idx)

    def update_with_weights(self, predicted : bool, expected: bool, weight_idx : int):
        if predicted and expected:
            self.weighted_tp += self.weights[weight_idx]
        if not predicted and expected:
            self.weighted_fn += self.weights[weight_idx]


    def calculate_accuracy(self):
        if self.tp + self.fn + self.fp + self.tn > 0:
            return (self.tp + self.tn) / (self.tp + self.fn + self.fp + self.tn)
        else:
            return 0
        
    def calculate_precision(self):
        if self.tp + self.fp > 0:
            return self.tp / (self.tp + self.fp)
        else:
            return 0
        
    def calculate_recall(self):
        if self.tp + self.fn > 0:
            return self.tp / (self.tp + self.fn)
        else:
            return 0
        
    def total_evaluations(self) -> int:
        return self.tp + self.tn + self.fp + self.fn
    
    def calculate_weighted_recall(self):
        if self.weighted_fn + self.weighted_tp > 0:
            return self.weighted_tp / (self.weighted_tp + self.weighted_fn)
        else:
            return 0
        
    
    def latency_array(self, ground_truth : NDArray, predictions : NDArray, 
                            t_start : int, t_end : int):
    
        if self.sampling != 1:
            raise NotImplemented("Sampling different from one was not yet implemented")
        
        
        t = np.arange(len(ground_truth), step=self.sampling)

        if t_end == t_start:
            delta = np.zeros_like(t, dtype=float)
        else:
            delta = (t - t_start) / (t_end - t_start)

        
        gt_mask = (ground_truth < 3).astype(int) # Only the true detection can be counted
        s_delta = 1 - 1 / (1 + np.exp(-self.beta * (2*delta - 1)))*(1-self.minimum_weight)

        larec = gt_mask * predictions * s_delta #* np.pow(self.alpha, -t)

        return larec
    
    def calculate_latency_recall(self, ground_truth : NDArray, predictions : NDArray, 
                                 t_start : int, t_end : int) -> float:
        
        la_rec = self.latency_array(ground_truth=ground_truth,
                                   predictions = predictions,
                                   t_start=t_start,
                                   t_end=t_end)
        
        positive_la_recall = la_rec[la_rec>0]
        avg_la_recall = np.mean(positive_la_recall).astype(float)
        self.la_recall = avg_la_recall

        return avg_la_recall
            
    def __str__(self) -> str:
        absolute_variables = f"Hits (True Positive):        {self.tp}\n" + \
                             f"Misses (False Negatives):       {self.fn}\n" + \
                             f"False Alarms (False Positives): {self.fp}\n" + \
                             f"Correct Rejections (True Negatives): {self.tn}\n"
            
        separation = "-"*79 + "\n"
        metrics = f"Accuracy:  {self.calculate_accuracy()*100:.2f}%\n" + \
                  f"Precision: {self.calculate_precision()*100:.2f}% (How reliable the detections were)\n" + \
                  f"Recall:    {self.calculate_recall()*100:.2f}% (How many actual pedestrians were caught)\n"
    
        if self.general_purpose:
            absolute_variables += f"Weighted True Positives: {self.weighted_tp}\n" + \
                                  f"Weighted False Negatives: {self.weighted_fn}\n"
            
            metrics += f"WRecall:   {self.calculate_weighted_recall()*100:.2f}% (How many actual pedestrian were caught with a bigger weight to closer detections)\n" + \
                       f"LaRecall:  {self.la_recall*100:.2f}% (How many actual pedestrian were caught with a bigger weight to early detections)\n"
            
        separation_2 = "="*79 + "\n"

        return absolute_variables + separation + metrics + separation_2


def main():
    args = parse_args()
    
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
    
    point_d = tuple(args.point_d)
    point_u = tuple(args.point_u)
    trapezes = generate_trapezes(point_d, point_u, w)
    
    # Calculate adaptive delay based on FPS
    delay = max(1, int(1000 / (fps if fps > 0 else 30)))
    
    # Vectors for storing detections and ground truths
    ground_truth = np.zeros((total_frames), dtype=int)
    detected = np.zeros((total_frames,), dtype=int)
    
    frame_idx = 0
    
    print(f"Running video in real time (~{fps:.1f} FPS, delay={delay}ms)")
    print("Hold or press 'd' when a person is inside the trapezoids, 'q' to quit")
    
    # ------------------------------------------ Main Loop ------------------------------------------
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
        
        # Draw visual elements on frame
        frame = write_lines(frame, point_d, point_u, w)
        frame = draw_bboxes(frame, frame_data)
        
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

    end_time = time.perf_counter()

    cap.release()
    cv2.destroyAllWindows()

    # ------------------------------------- Generate Statistics -------------------------------------

    tp = 0
    fn = 0
    fp = 0
    tn = 0
    fn_weighted = 0
    tp_weighted = 0
    w_recall_weights = [1,5,10]

    general_statistics = Stats(general = True)
    statistics_per_zone = {
        'red': Stats(),
        'orange': Stats(),
        'green': Stats()
    }

    zone_names = ['green', 'orange', 'red']

    for i in range(total_frames):
        statistics_per_zone["green"].update(expected = ground_truth[i] == 2, predicted = detected[i]==1)
        statistics_per_zone["orange"].update(expected = ground_truth[i] == 1, predicted = detected[i]==1)
        statistics_per_zone["red"].update(expected = ground_truth[i] == 0, predicted = detected[i]==1)
        general_statistics.update(expected=ground_truth[i]<3, predicted = detected[i]==1, weight_idx=2-ground_truth[i])

        if ground_truth[i] < 3 and detected[i]==1:
            tp += 1
            tp_weighted += w_recall_weights[2 - ground_truth[i]]
        elif ground_truth[i] < 3 and not detected[i]==1:
            fn += 1
            fn_weighted += w_recall_weights[2 - ground_truth[i]]
        elif not ground_truth[i] < 3 and detected[i]==1:
            fp += 1
        elif not ground_truth[i] < 3 and not detected[i]==1:
            tn += 1
    
    # Calculate performance metrics
    processed_frames = tp + fn + fp + tn
    assert processed_frames == total_frames, "Problem retrieving information from the video -> processed frames are different than total frames"

    accuracy = (tp + tn) / processed_frames if processed_frames > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    w_recall = tp_weighted / (tp_weighted + fn_weighted) if (tp_weighted + fn_weighted) > 0 else 0.0

    t_start, t_end = -1, -1
    for i in range(total_frames):
        if ground_truth[i] < 3 and t_start == -1:
            t_start = i

        if ground_truth[total_frames - i - 1] < 3 and t_end == -1:
            t_end = total_frames - i - 1

        if t_start!=-1 and t_end!=-1:
            break

    alpha = 1
    beta = 4
    minimum_weight = 0.2
    sampling = 1
    print(f"For this video the detection starts in frame {t_start} and finishes in frame {t_end}")

    la_recall = calculate_la_recall(ground_truth=ground_truth,
                                    predictions=detected,
                                    t_start=t_start,
                                    t_end=t_end,
                                    alpha=alpha,
                                    beta=beta,
                                    minimum_weight=minimum_weight,
                                    sampling=sampling)
    
    class_la_array = general_statistics.latency_array(ground_truth = ground_truth,
                                                      predictions = detected,
                                                      t_start = t_start,
                                                      t_end = t_end)
    class_la_recall = general_statistics.calculate_latency_recall(ground_truth = ground_truth,
                                                                  predictions = detected,
                                                                  t_start = t_start,
                                                                  t_end = t_end)
    
    
    plt.plot(la_recall, linewidth=7.0)
    plt.plot(class_la_array)
    plt.title("Latency Recall Evaluation")
    plt.vlines([t_start, t_end], 0, 1 + minimum_weight, linestyles='dashed')
    plt.show()
    
    positive_la_recall = la_recall[la_recall>0]
    avg_la_recall = np.mean(positive_la_recall)

    print("\n" + "="*30 + " EVALUATION REPORT " + "="*30)
    print(f"Elapsed Time: {end_time - initial_time}")
    print(f"Total Frames Evaluated: {processed_frames}")
    print(f"Hits (True Positive):        {tp}")
    print(f"Misses (False Negatives):       {fn}")
    print(f"False Alarms (False Positives): {fp}")
    print(f"Correct Rejections (True Negatives): {tn}")
    print(f"Weighted True Positives: {tp_weighted}")
    print(f"Weighted False Negatives: {fn_weighted}")
    print("-"*79)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}% (How reliable the detections were)")
    print(f"Recall:    {recall*100:.2f}% (How many actual pedestrians were caught)")
    print(f"WRecall:   {w_recall*100:.2f}% (How many actual pedestrian were caught with a bigger weight to closer detections)")
    print(f"LaRecall:  {avg_la_recall*100:.2f}% (How many actual pedestrian were caught with a bigger weight to early detections)")
    print("="*79)


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

if __name__ == "__main__":
    main()