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
    
from computer_vision.utils.draw import write_lines, draw_bboxes_from_data
from computer_vision.core.audio_handler import AudioHandler
from computer_vision.utils.transformations import resize,  continuous_morphological_closing
from computer_vision.utils.video import get_video_parameters
from computer_vision.benchmark.stats import Stats

from.config import ALARM_CONFIG

# Audio Settings
from.config import CHUNK, CHANNELS, RATE, FORMAT



class ZoneDetector:
    def __init__(self,
                 alarm : str,
                 show : bool = False,
                 csv : Path = Path("results.csv"),
                 alarm_config : Path = ALARM_CONFIG,
                 close_detection_gaps : int = 20,
                 delay_window : int = 60,
                 chunk : int = CHUNK,
                 format : int = FORMAT,
                 channels : int = CHANNELS,
                 rate : int = RATE,
                 save_audio : bool = False
                 ):
        
        self.show = show

        self.csv_path = csv.resolve()

        self.alarm_name = alarm
        self.alarm_path = alarm_config.resolve()

        print(self.alarm_path)
        self.alarm_frequency, self.alarm_amplitude = self._get_alarm_frequency(self.alarm_name, self.alarm_path)

        if self.alarm_frequency is None:
            raise RuntimeError("Could not get alarm frequency from yaml file.")
        if self.alarm_amplitude is None:
            raise RuntimeError("Could not get alarm amplitude threshold from yaml file.")

        # Corresponds to how many frames the system can ignore to consider a single detection extract
        self.close_detection_gaps = close_detection_gaps
        self.window = delay_window 

        self.save_audio = save_audio

        self.chunk = chunk
        self.rate = rate
        self.format = format
        self.channels = channels
    
    def _parse_predictions(self, txt_path: Path) -> dict:
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

    def _get_alarm_frequency(self, alarm_name : str, filepath : Path) -> tuple[None, None] | tuple[float, float]:
        alarm_name = alarm_name.strip() 

        if filepath.exists():
            with open(filepath, "r") as yaml_file:
                data_alarm = yaml.safe_load(yaml_file) or {}
            if data_alarm:
                if alarm_name in data_alarm:
                    print("Alarm found: ")
                    print(data_alarm[alarm_name])
                    return data_alarm[alarm_name]['frequency'], data_alarm[alarm_name]['amplitude']

        return None, None


    def _init_video(self, 
                    video:Path,
                    predictions : Path,
                    point_d : tuple[int, int],
                    point_u : tuple[int, int],
                    ):
        self.video = video.resolve()
        if not self.video.exists():
            raise FileExistsError(f"Video not found: {self.video}")

        w, h, fps = get_video_parameters(self.video)

        self.width = w
        self.height = h
        self.fps = fps

        self.predictions_path = predictions.resolve()
        
        self.point_d = point_d
        self.point_u = point_u

        if not self.predictions_path.exists():
            raise FileExistsError(f"Predictions file not found: {self.predictions_path}")

        self.predictions = self._parse_predictions(self.predictions_path)

        self.log_folder = self.csv_path.parent / self.video.stem
        self.log_folder.mkdir(exist_ok=True, parents=True)

        # Audio Handler
        self.ah = AudioHandler(chunk=self.chunk,
                                format=self.format,
                                channels=self.channels,
                                rate=self.rate)



    # ------------------------------------- Generate Statistics -------------------------------------
    def extract_stats(self, 
                        ground_truth : NDArray, 
                        detected : NDArray, 
                        detected_without_resampling : NDArray, 
                        total_frames : int, 
                        initial_video_time : float, 
                        final_video_time : float, 
                        original_video_duration : float,
                        audio_duration : float,
                        frame_seconds : NDArray):
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

        

        minimum_weight = general_statistics.minimum_weight
        binary_ground_truth = (ground_truth < 3).astype(int)

        time_stamps = general_statistics.get_detection_duration_in_frames(ground_truth=binary_ground_truth,
                                                                        closing_se_size=self.close_detection_gaps)
                                                                
        class_la_array = general_statistics.latency_array(ground_truth = binary_ground_truth,
                                                        predictions = detected,
                                                        time_stamps = time_stamps)
        
        la_recall = general_statistics.calculate_latency_recall(ground_truth = binary_ground_truth,
                                                                predictions = detected,
                                                                closing_structure_size=self.close_detection_gaps)

        plt.figure()
        plt.plot(class_la_array)
        plt.title("Latency Recall Evaluation")
        plt.vlines(time_stamps, 0, 1 + minimum_weight, linestyles='dashed')
        plt.savefig(self.log_folder / "Latency_Recall.jpg")
        plt.close()

        frame_delay = general_statistics.get_frame_delay(b_video_detection=binary_ground_truth,
                                                        b_audio_detection=detected,
                                                        window = self.window,
                                                        closing_se_size=self.close_detection_gaps)

        second_delay = general_statistics.get_seconds_delay(b_video_detection=binary_ground_truth,
                                                            frame_seconds=frame_seconds,
                                                            audio_detection=detected_without_resampling,
                                                            audio_duration=audio_duration,
                                                            audio_start_timestamp=self.ah.record_start_time,
                                                            window=self.window,
                                                            closing_se_size=self.close_detection_gaps)

        print("Frame Delay: ", frame_delay)
        print("Second Delay: ", second_delay)
        if len(frame_delay) != len(second_delay):
            raise RuntimeError("Quantity of frame delays and seconds delays are different, what shouldn't happen")
        
        video_duration = total_frames/self.fps
        general_statistics.calculate_rfa(video_duration)

        print("\n" + "="*30 + " EVALUATION REPORT " + "="*30)
        print(f"Original Video Duration (seconds): {original_video_duration}")
        print(f"Elapsed Time (seconds): {final_video_time - initial_video_time}")
        print(f"Video started at {initial_video_time}")
        print(f"Video finishes at {final_video_time}")
        print(f"Total Frames Evaluated: {processed_frames}")


        print("-"*40 + " General Statistics " + "-"*40)
        print(general_statistics)

        print("-"*40 + " Green Statistics " + "-"*40)
        print(statistics_per_zone['green'])

        print("-"*40 + " Orange Statistics " + "-"*40)
        print(statistics_per_zone['orange'])

        print("-"*40 + " Red Statistics " + "-"*40)
        print(statistics_per_zone['red'])

        plt.figure()
        plt.plot(3-ground_truth)
        plt.plot(detected)
        plt.grid()
        plt.savefig(self.log_folder / "Detected_X_Ground_Truth.jpg")
        plt.close()

        results = {
            'Name': str(self.video).split('/')[-1],
            'Has_Pedestrians': (general_statistics.tp + general_statistics.fn) > 0,
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


        return results

    def _extract_delays(self, frame_delay, seconds_delay):
        delay_stats = []
        print("frame delay", frame_delay)
        for i in range(len(frame_delay)):
            delay_dict = {
                'name': self.video.stem,
                'frame': frame_delay[i],
                'second': seconds_delay[i]
            }
            delay_stats.append(delay_dict)

        return delay_stats

    def save_csv(self, results):

        csv_file_path = self.csv_path.resolve()
        delay_csv_file_path = csv_file_path.with_stem(f"{csv_file_path.stem}_delay")
        delay_csv_file_path = delay_csv_file_path.resolve()

        frame_delay = results['Frame_Delay']
        seconds_delay = results['Seconds_Delay']

        del results['Frame_Delay']
        del results['Seconds_Delay']


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

        delay_stats = self._extract_delays(frame_delay=frame_delay, seconds_delay=seconds_delay)

        if delay_stats:
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
                

    def detect_zones(self, 
                    video:Path,
                    predictions : Path,
                    point_d : tuple[int, int],
                    point_u : tuple[int, int],
                    ):
        
        self._init_video(video, predictions=predictions, point_d=point_d, point_u=point_u)

        print(f"Loaded predictions for {len(self.predictions)} frames")
        
        cap = cv2.VideoCapture(str(self.video))
        assert cap.isOpened(), "Error reading video file"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_seconds = total_frames/self.fps
        
        # Calculate adaptive delay based on FPS
        delay = max(1, int(1000 / (self.fps if self.fps > 0 else 30)))
        
        # Vectors for storing detections and ground truths
        ground_truth = np.full((total_frames), 3, dtype=int)
        frame_time = np.zeros((total_frames,), dtype=float)
        
        frame_idx = 0
        
        print(f"Running video in real time (~{self.fps:.1f} FPS, delay={delay}ms)")
        print("Hold or press 'd' when a person is inside the trapezoids, 'q' to quit")
        
        # ------------------------------------------ Main Loop ------------------------------------------
        cv2.namedWindow("Zone Counter", cv2.WND_PROP_FULLSCREEN)

        cv2.waitKey(100)
        cv2.setWindowProperty("Zone Counter",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


        start_event = threading.Event()
        stop_event = threading.Event()

        thread_audio = threading.Thread(target=self.ah.record_audio_async, args=(start_event, stop_event))
        thread_audio.start()

        start_event.set()

        initial_time = time.perf_counter()
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("End of video")
                break
            
            # Get ground truth min_zone
            frame_data = self.predictions.get(frame_idx, [])
            min_zone = min((zone for zone, _ in frame_data), default=3)
            ground_truth[frame_idx] = min_zone
            
            if self.show:
                frame = write_lines(frame, self.point_d)
                frame = draw_bboxes_from_data(frame, frame_data)
            
                # Display current metrics overlay
                cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Min Zone: {min_zone}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        
            cv2.imshow('Zone Counter', frame)
            frame_time[frame_idx] = time.perf_counter()
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q'):
                break
            frame_idx += 1

        stop_event.set()
        thread_audio.join()

        end_time = time.perf_counter()


        self.ah.terminate()
        cap.release()
        cv2.destroyAllWindows()

        execution_fps = (frame_idx+1) / (end_time - initial_time)
        with open(self.log_folder / "fps.txt", "a") as fps_txt:
            fps_txt.write(f"{self.video.stem} fps = {execution_fps}")

        # ------------------------------------- Closing Ground Truth Holes -------------------------------------

        closed_ground_truth = continuous_morphological_closing(3-ground_truth, 21)
        closed_ground_truth = (3-closed_ground_truth).astype(int)

        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(3-ground_truth, color='red')
        plt.title("Original Grount Truth")
        
        plt.subplot(1,2,2)
        plt.plot(3-closed_ground_truth, color='blue')
        plt.title("Ground Truth After Morphological Closing")
        plt.savefig(self.log_folder / "Ground_Truth_After_Closing.jpg")

        plt.close()
        

        # ------------------------------------- Get Audio Detection -------------------------------------
        audio_data = self.ah.get_audio_stream_data()

        audio_seconds = len(audio_data) / self.ah.rate

        frequency, t, dbs = self.ah.spectrogram(audio_data=audio_data)

        if self.save_audio:
            self.ah.save_audio(output_path=str(self.csv_path.with_name("Detection_Audio.wav")))

        self.ah.plot_spectrogram(frequency=frequency,
                            time_stamps=t,
                            spectrogram=dbs,
                            show=False,
                            filepath=str(self.log_folder / "Spectrogram_Detection.jpg"))

        if isinstance(self.alarm_frequency, float) and isinstance(self.alarm_amplitude, float):
            binary_detection = self.ah.get_binary_detection(audio_data=audio_data,
                                                        alarm_frequency=self.alarm_frequency,
                                                        amp_threshold=self.alarm_amplitude,
                                                        interval=0.05,
                                                        seconds=audio_seconds,
                                                        save_plot=self.log_folder)
        else:
            raise ValueError("Alarm Frequency and Alarm Amplitude should be floating points numbers")

        final_detection = self.ah.morph_closing(binary_detection=binary_detection,
                                        struct_size=self.close_detection_gaps)

        detected_audio_video = self.ah.resample_detection(frame_time, final_detection)

        plt.figure()
        plt.subplot(1,2,1)
        plt.title("Audio detection")
        plt.plot(final_detection, color='red')

        plt.subplot(1,2,2)
        plt.title("Audio detection after resampling")
        plt.plot(detected_audio_video, color='red')
        plt.plot(3-closed_ground_truth, color='blue')
        plt.savefig(self.log_folder / "Audio_Resampling.jpg")

        plt.close()

        results = self.extract_stats(ground_truth=closed_ground_truth, 
                                     detected=detected_audio_video, 
                                     detected_without_resampling=final_detection,
                                     total_frames=total_frames,
                                     initial_video_time=initial_time,
                                     final_video_time=end_time,
                                     original_video_duration=video_seconds,
                                     audio_duration=audio_seconds,
                                     frame_seconds=frame_time
                                     )
        
        self.save_csv(results=results)

    
            
