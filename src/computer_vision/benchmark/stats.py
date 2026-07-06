import numpy as np
from numpy.typing import NDArray


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


    def calculate_accuracy(self) -> float:
        if self.tp + self.fn + self.fp + self.tn > 0:
            return (self.tp + self.tn) / (self.tp + self.fn + self.fp + self.tn)
        else:
            return 0.0
        
    def calculate_precision(self) -> float:
        if self.tp + self.fp > 0:
            return self.tp / (self.tp + self.fp)
        else:
            return 0.0
        
    def calculate_recall(self) -> float:
        if self.tp + self.fn > 0:
            return self.tp / (self.tp + self.fn)
        else:
            return 0.0
        
    
    def calculate_weighted_recall(self) -> float:
        if self.weighted_fn + self.weighted_tp > 0 and self.general_purpose:
            return self.weighted_tp / (self.weighted_tp + self.weighted_fn)
        else:
            return 0.0
        
    def total_evaluations(self) -> int:
        return self.tp + self.tn + self.fp + self.fn
    
    def calculate_pmiss(self) -> float:
        n_miss = self.fn
        n_target = self.tp + self.fn

        p_miss = n_miss/n_target if n_target>0 else 0

        return p_miss

    def calculate_pfa(self) -> float:
        n_fa = self.fp
        n_source = self.fp + self.tn

        p_fa = n_fa/n_source if n_source>0 else 0
    
        return p_fa
    
    def calculate_rfa(self, time:int) -> float:
        n_fa = self.fp
        hours = time/3600

        self.r_fa = n_fa / hours if hours != 0 else 0

        return self.r_fa
    
    def calculate_ndcr(self) -> float:
        fa_term = self.calculate_pfa() if self.r_fa == 0 else self.r_fa

        return self.calculate_pmiss() + 0.005 * fa_term
    
    def latency_array(self, ground_truth : NDArray, predictions : NDArray, 
                            t_start : int, t_end : int):
    
        if self.sampling != 1:
            raise NotImplemented("Sampling different from one was not yet implemented")
        
        
        t = np.arange(len(ground_truth), step=self.sampling)

        if t_end <= t_start:
            delta = np.zeros_like(t, dtype=float)
        else:
            delta = (t - t_start) / (t_end - t_start)

        
        gt_mask = (ground_truth < 3).astype(int) # Only the true detection can be counted
        s_delta = 1 - 1 / (1 + np.exp(-self.beta * (2*delta - 1)))*(1-self.minimum_weight)

        larec = gt_mask * predictions * s_delta #* np.pow(self.alpha, -t)

        return larec
    
    def calculate_latency_recall(self, ground_truth : NDArray, predictions : NDArray, 
                                 t_start : int, t_end : int) -> float:
        
        if self.general_purpose:
            la_rec = self.latency_array(ground_truth=ground_truth,
                                    predictions = predictions,
                                    t_start=t_start,
                                    t_end=t_end)
            
            positive_la_recall = la_rec[la_rec>0]
            avg_la_recall = np.mean(positive_la_recall).astype(float) if positive_la_recall.size>0 else 0
            self.la_recall = avg_la_recall

            return avg_la_recall
        else:
            return 0.0
            
    def __str__(self) -> str:
        absolute_variables = f"Hits (True Positive):        {self.tp}\n" + \
                             f"Misses (False Negatives):       {self.fn}\n" + \
                             f"False Alarms (False Positives): {self.fp}\n" + \
                             f"Correct Rejections (True Negatives): {self.tn}\n"
            
        separation = "-"*79 + "\n"
        metrics = f"Accuracy:                               {self.calculate_accuracy()*100:.2f}% (How many detections were right)\n" + \
                  f"Precision:                              {self.calculate_precision()*100:.2f}% (How reliable the detections were)\n" + \
                  f"Recall:                                 {self.calculate_recall()*100:.2f}% (How many actual pedestrians were caught)\n" + \
                  f"Miss Probability:                       {self.calculate_pmiss()*100:.2f}% (Missed Detection probability)\n"
    
        if self.general_purpose:
            absolute_variables += f"Weighted True Positives: {self.weighted_tp}\n" + \
                                  f"Weighted False Negatives: {self.weighted_fn}\n"
            
            metrics += f"False Alarm Probability:                {self.calculate_pfa()*100:.2f}% (False Alarm probability)\n" + \
                       f"False Alarm Rate:                       {self.r_fa*100:.2f} (False Alarm Rate in occurence per hour)\n" + \
                       f"NDCR:                                   {self.calculate_ndcr():.2f} (Normalized Detection Cost Rate, a weighted combination)\n" + \
                       f"WRecall:                                {self.calculate_weighted_recall()*100:.2f}% (How many actual pedestrian were caught with a bigger weight to closer detections)\n" + \
                       f"LaRecall:                               {self.la_recall*100:.2f}% (How many actual pedestrian were caught with a bigger weight to early detections)\n"
            
        separation_2 = "="*79 + "\n"

        return absolute_variables + separation + metrics + separation_2

