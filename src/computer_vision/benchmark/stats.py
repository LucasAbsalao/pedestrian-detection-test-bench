import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_closing
import math
from typing import Optional

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
            r_fa (int): false alarm rate.
            general_purpose (bool): Flag indicating if advanced metrics are enabled.
            
            Attributes initialized only if general is True:
                weights (list[int]): Importance weights for different detection zones.
                weighted_tp (int): Accumulated weighted True Positives.
                weighted_fn (int): Accumulated weighted False Negatives.
                
                alpha (int): Exponential decay parameter for latency recall.
                beta (int): Steepness parameter for the Sigmoid time-decay function.
                minimum_weight (float): The minimum baseline score for a delayed detection.
                sampling (int): Frame sampling interval.
                la_recall (float): Value of latency recall
        """
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        
        self.r_fa = 0

        self.general_purpose = general
        if general:
            self.weights = [1, 3, 6]
            self.weighted_tp = 0
            self.weighted_fn = 0

            self.alpha = 1
            self.beta = 4
            self.minimum_weight = 0.2
            self.sampling_la_recall = 1
            self.la_recall = 0.

            self.frame_delay = []
            self.seconds_delay = []
    
    def set_latency_parameters(self, alpha : float, beta: float, minimum_weight : float, sampling : int):
        self.alpha = alpha
        self.beta = beta
        self.minimum_weight = minimum_weight
        self.sampling_la_recall = sampling

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


    def calculate_accuracy(self) -> float:
        if self.tp + self.fn + self.fp + self.tn > 0:
            return (self.tp + self.tn) / (self.tp + self.fn + self.fp + self.tn)
        else:
            return 0.0
        
    def calculate_precision(self) -> Optional[float]:
        if self.tp + self.fp > 0:
            return self.tp / (self.tp + self.fp)
        else:
            return None
        
    def calculate_recall(self) -> Optional[float]:
        if self.tp + self.fn > 0:
            return self.tp / (self.tp + self.fn)
        else:
            return None

    def calculate_f1_score(self) -> Optional[float]:
        recall = self.calculate_recall()
        precision = self.calculate_precision()
        if precision is not None and recall is not None and (precision+recall)>0:
            return (2*precision*recall) / (precision + recall)
        else:
            return None
        
    
    def calculate_weighted_recall(self) -> Optional[float]:
        if self.weighted_fn + self.weighted_tp > 0 and self.general_purpose:
            return self.weighted_tp / (self.weighted_tp + self.weighted_fn)
        else:
            return None
        
    def total_evaluations(self) -> int:
        return self.tp + self.tn + self.fp + self.fn
    
    def calculate_pmiss(self) -> float:
        n_miss = self.fn
        n_target = self.tp + self.fn

        if n_target > 0:
            return n_miss / n_target
        else:
            return 0

    def calculate_pfa(self) -> Optional[float]:
        n_fa = self.fp
        n_source = self.fp + self.tn

        if n_source > 0:
            return n_fa / n_source
        else:
            return None
    
    def calculate_rfa(self, time:float) -> float:
        n_fa = self.fp
        hours = time/3600

        self.r_fa = n_fa / hours if hours != 0 else 0

        return self.r_fa
    
    def calculate_ndcr(self) -> Optional[float]:
        fa_term = self.calculate_pfa() if self.r_fa == 0 else self.r_fa

        p_miss = self.calculate_pmiss()
        if p_miss is None or fa_term is None:
            return None
        else:
            return p_miss + 0.005 * fa_term
    

    def get_detection_duration_in_frames(self, ground_truth : NDArray, closing_se_size:int):
        timestamps = []
        
        structure = np.ones((closing_se_size,), dtype=int)

        closed_mask = binary_closing(ground_truth, structure=structure, border_value=0).astype(int)

        padded = np.pad(closed_mask, (1, 1), mode='constant', constant_values=0)

        diffs = np.diff(padded)

        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0] - 1

        timestamps = np.column_stack((starts, ends)).tolist()

        return timestamps

    
    def get_frame_delay(self, b_video_detection : NDArray, b_audio_detection : NDArray, window : int, closing_se_size : int) -> list[int]:

        if len(b_video_detection) != len(b_audio_detection):
            raise RuntimeError("Both vectors need to have the same size!")

        time_stamps = self.get_detection_duration_in_frames(ground_truth=b_video_detection, closing_se_size=closing_se_size)
        if not time_stamps:
            return []

        frame_delay = []
        for start_frame, end_frame in time_stamps:

            limit_sup = min(start_frame+window, len(b_video_detection))

            detections = np.flatnonzero(b_audio_detection[start_frame:limit_sup])
            if detections.size>0:
                first_audio_detection = int(detections[0])
                frame_delay.append(first_audio_detection)
            else:
                frame_delay.append(window)

            # detected = False
            # for i in range(start_frame, limit_sup):
            #     if b_audio_detection[i] == 1:
            #         frame_delay.append(i-start_frame)
            #         detected = True
            #         break

            # if not detected:
            #     frame_delay.append(window)
                    
        self.frame_delay = frame_delay
        return frame_delay

    def get_first_idx_after_time(self, time:float, sampling_period:float):
        idx = math.ceil(round(time / sampling_period, 7))
        return idx
        

    def get_seconds_delay(self, b_video_detection : NDArray, frame_seconds : NDArray, audio_detection : NDArray, audio_start_timestamp : float, audio_duration : float, window : int, closing_se_size : int) -> list[float]:

        if len(b_video_detection) != len(frame_seconds):
                    raise RuntimeError("Both vectors of detection and timestamps need to have the same size!")
        
        time_stamps = self.get_detection_duration_in_frames(ground_truth=b_video_detection, closing_se_size=closing_se_size)
        if not time_stamps:
            return []

        sampling_period_in_seconds = audio_duration / len(audio_detection)

        second_delay = []

        if len(frame_seconds) > 1:
            max_seconds_delay = (frame_seconds[-1] - frame_seconds[0])/(len(frame_seconds)-1) * window
        else:
            raise NotImplementedError("Can't evaluate videos with just one frame")
            
        for start_frame, end_frame in time_stamps:

            limit_sup_frame = min(start_frame+window, len(frame_seconds)-1)

            limit_inf_seconds = float(frame_seconds[start_frame])
            limit_inf_audio_idx = self.get_first_idx_after_time(limit_inf_seconds - audio_start_timestamp, sampling_period_in_seconds) # Indice of first audio chunk after frame was showed

            limit_sup_seconds = float(frame_seconds[limit_sup_frame])
            limit_sup_audio_idx = self.get_first_idx_after_time(limit_sup_seconds - audio_start_timestamp, sampling_period_in_seconds) # Indice of first audio chunk after detection stopped or after last frame of the video was showed

            # Check if it can get bigger than the length of the audio detection
            limit_sup_audio_idx = min(limit_sup_audio_idx, len(audio_detection))

            detections = np.flatnonzero(audio_detection[limit_inf_audio_idx:limit_sup_audio_idx])
            if detections.size > 0:
                first_audio_idx = limit_inf_audio_idx + detections[0]
                second_delay.append(float(audio_start_timestamp + first_audio_idx*sampling_period_in_seconds - limit_inf_seconds))
            else:
                second_delay.append(max_seconds_delay)

        self.seconds_delay = second_delay

        return second_delay


    def get_seconds_delay_vectorized(self, b_video_detection : NDArray, frame_seconds : NDArray, audio_detection : NDArray, audio_start_timestamp : float, audio_duration : float, window : int, closing_se_size : int, measure_from_end : bool = False) -> list[float]:
        """
        Vectorized, more robust version of get_seconds_delay.

        Differences from get_seconds_delay:
          - Uses np.searchsorted on the audio timestamp axis instead of math.ceil,
            which avoids float-rounding artifacts and negative-index wrap-around.
          - Clamps audio indices to the valid range [0, len(audio_detection)].
          - Fixed frame-period divisor (N frames -> N-1 intervals) for the sentinel.
          - Optional measure_from_end: search the window starting at the END of the
            detection segment instead of the start (useful when the alarm fires
            after the person has left the zone).

        Not being used.
        """
        b_video_detection = np.asarray(b_video_detection)
        frame_seconds = np.asarray(frame_seconds, dtype=float)
        audio_detection = np.asarray(audio_detection)

        if b_video_detection.shape[0] != frame_seconds.shape[0]:
            raise RuntimeError("b_video_detection and frame_seconds must have the same size!")
        if frame_seconds.shape[0] < 1:
            raise RuntimeError("frame_seconds must not be empty!")

        sampling_period = audio_duration / audio_detection.shape[0]

        # Absolute timestamp of every audio sample
        audio_times = audio_start_timestamp + sampling_period * np.arange(audio_detection.shape[0])

        # Average frame period: N frames span N-1 intervals
        if frame_seconds.shape[0] > 1:
            frame_period = (frame_seconds[-1] - frame_seconds[0]) / (frame_seconds.shape[0] - 1)
        else:
            frame_period = 0.0
        max_seconds_delay = frame_period * window

        time_stamps = self.get_detection_duration_in_frames(ground_truth=b_video_detection, closing_se_size=closing_se_size)
        if not time_stamps:
            return []

        second_delay = []
        for start_frame, end_frame in time_stamps:
            if measure_from_end:
                anchor_frame = end_frame
            else:
                anchor_frame = start_frame

            limit_sup_frame = min(anchor_frame + window, frame_seconds.shape[0] - 1)

            t_inf = frame_seconds[anchor_frame]
            t_sup = frame_seconds[limit_sup_frame]

            # First audio sample whose timestamp is >= t_inf (left search) and < t_sup (right, exclusive)
            lo = int(np.searchsorted(audio_times, t_inf, side='left'))
            hi = int(np.searchsorted(audio_times, t_sup, side='left'))
            hi = min(hi, audio_detection.shape[0])

            detections = np.flatnonzero(audio_detection[lo:hi])
            if detections.size > 0:
                first_audio_idx = lo + detections[0]
                second_delay.append(float(audio_times[first_audio_idx] - t_inf))
            else:
                second_delay.append(max_seconds_delay)

        self.seconds_delay = second_delay
        return second_delay


    def latency_array(self, ground_truth : NDArray, predictions : NDArray, time_stamps : list):
    
        if self.sampling_la_recall != 1:
            raise NotImplementedError("Sampling different from one was not yet implemented")

        time_stamps = [[start, end]for start, end in time_stamps if start != end]

        t = np.arange(len(ground_truth), step=self.sampling_la_recall)

        delta = np.zeros_like(t, dtype=float)
        if time_stamps:
            count = 0
            for time in t:
                if time > time_stamps[count][1] and count<len(time_stamps)-1:
                    count+=1
                delta[time] = (time - time_stamps[count][0]) / (time_stamps[count][1] - time_stamps[count][0])
        
    
        s_delta = 1 - 1 / (1 + np.exp(-self.beta * (2*delta - 1)))*(1-self.minimum_weight)

        larec = ground_truth * predictions * s_delta #* np.pow(self.alpha, -t)

        return larec
    
    def calculate_latency_recall(self, ground_truth : NDArray, predictions : NDArray, 
                                 closing_structure_size : int = 10) -> Optional[float]:
        
        if self.general_purpose:
            time_stamps = self.get_detection_duration_in_frames(ground_truth=ground_truth, closing_se_size=closing_structure_size)
            if not time_stamps:
                return None

            la_rec = self.latency_array(ground_truth=ground_truth,
                                    predictions = predictions,
                                    time_stamps=time_stamps          
            )

            la_rec_ground_truth = self.latency_array(ground_truth=ground_truth,
                                                     predictions=ground_truth,
                                                     time_stamps=time_stamps)
            
            sum_gt = np.sum(la_rec_ground_truth)
            if sum_gt == 0:
                return None

            sum_la_rec = np.sum(la_rec)
            final_la_recall = (sum_la_rec / sum_gt).astype(float)

            self.la_recall = final_la_recall

            return final_la_recall
        else:
            return None

    def calculate_latency_recall_mean(self, ground_truth : NDArray, predictions : NDArray, 
                                     closing_structure_size : int = 10) -> float:
            
            if self.general_purpose:
                time_stamps = self.get_detection_duration_in_frames(ground_truth=ground_truth, closing_se_size=closing_structure_size)
    
                la_rec = self.latency_array(ground_truth=ground_truth,
                                        predictions = predictions,
                                        time_stamps=time_stamps          
                )
                
                positive_la_recall = la_rec[la_rec>0]
                avg_la_recall = np.mean(positive_la_recall).astype(float) if positive_la_recall.size>0 else 0
                self.la_recall = avg_la_recall
    
                return avg_la_recall
            else:
                return 0.0

    def calculate_is_detected(self, ground_truth : NDArray, predictions : NDArray, closing_structure_size : int = 10) -> tuple[int, int] | tuple[None, None]:
        '''Check if a person was detected at least onde for every continuous detection interval'''
        self.gt_intervals = None
        self.detected_interval = None
        if self.general_purpose:
            time_stamps = self.get_detection_duration_in_frames(ground_truth=ground_truth,
                                                                closing_se_size=closing_structure_size)
            if len(time_stamps) == 0:

                return None, None
            total_existence = len(time_stamps)
            detections = 0
            for t_start, t_end in time_stamps:
                detected = np.flatnonzero(predictions[t_start:t_end+1])
                if detected.size > 0:
                    detections+=1

            self.gt_intervals = total_existence
            self.detected_interval = detections

            return detections, total_existence
        else:
            return None, None
        
    def __str__(self) -> str:
        absolute_variables = f"Hits (True Positive):        {self.tp}\n" + \
                             f"Misses (False Negatives):       {self.fn}\n" + \
                             f"False Alarms (False Positives): {self.fp}\n" + \
                             f"Correct Rejections (True Negatives): {self.tn}\n"

        acc = self.calculate_accuracy()
        prec = self.calculate_precision()
        rec = self.calculate_recall()
        f1 = self.calculate_f1_score()
        pmiss = self.calculate_pmiss()
            
        separation = "-"*79 + "\n"

        metrics = f"Accuracy:                               {acc*100:.2f}% (How many detections were right)\n"

        if prec is not None:
            metrics += f"Precision:                              {prec*100:.2f}% (How reliable the detections were)\n"
        else:
            metrics += f"Precision:                              N/A (No actual pedestrians in video)\n"

        if rec is not None:
            metrics += f"Recall:                                 {rec*100:.2f}% (How many actual pedestrians were caught)\n"
        else:
            metrics += f"Recall:                                 N/A (No actual pedestrians in video)\n"

        if f1 is not None:
            metrics += f"F1 Score:                               {f1*100:.2f}% (Overall balance between detecting pedestrians and avoiding false alarms)\n"
        else:
            metrics += f"F1 Score:                               N/A (No actual pedestrians in video)\n"

        
        metrics += f"Miss Probability:                       {pmiss*100:.2f}% (Missed Detection probability)\n"

    
        if self.general_purpose:
            absolute_variables += f"Weighted True Positives: {self.weighted_tp}\n" + \
                                  f"Weighted False Negatives: {self.weighted_fn}\n" + \
                                  f"Frame Delays in Seconds: {self.seconds_delay}\n" + \
                                  f"Frame Delays in frames: {self.frame_delay}\n"

            w_rec = self.calculate_weighted_recall()
            p_fa = self.calculate_pfa()

            if p_fa is not None:    
                metrics += f"False Alarm Probability:                {p_fa*100:.2f}% (False Alarm probability)\n"
            else:
                metrics += f"False Alarm Probability:                N/A (No actual pedestrians in video)\n"

            metrics += f"False Alarm Rate:                       {self.r_fa*100:.2f} (False Alarm Rate in occurence per hour)\n" + \
                       f"NDCR:                                   {self.calculate_ndcr():.2f} (Normalized Detection Cost Rate, a weighted combination)\n"

            if w_rec is not None:
                metrics += f"WRecall:                                {w_rec*100:.2f}% (How many actual pedestrian were caught with a bigger weight to closer detections)\n"
            else:
                metrics += f"WRecall:                                N/A (No actual pedestrians in video)\n"

            metrics +=  f"LaRecall:                               {self.la_recall*100:.2f}% (How many actual pedestrian were caught with a bigger weight to early detections)\n"
            
        separation_2 = "="*79 + "\n"

        return absolute_variables + separation + metrics + separation_2

