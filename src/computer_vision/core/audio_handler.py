import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import time
from pathlib import Path
import math
from scipy import signal
from scipy import stats
from scipy.ndimage import binary_closing

from .config import CHUNK, FORMAT, CHANNELS, RATE

class AudioHandler:
    def __init__(self, 
                 chunk : int = CHUNK,
                 format : int = FORMAT,
                 channels : int = CHANNELS,
                 rate : int = RATE,
                 nperseg_factor : int = 114,
                 dbs : bool = True) -> None:
        
        self.pyaudio = pyaudio.PyAudio()

        self.stream = self.pyaudio.open(format=format,
                                   channels=channels,
                                   rate=rate,
                                   input=True,
                                   frames_per_buffer=chunk)
        self.chunk = chunk
        self.format = format
        self.channels = channels  
        self.rate = rate      
        self.dbs = dbs
        self.time_samples_per_seg = nperseg_factor
        

    def record_audio(self, seconds):

        print("[AUDIO] * recording")

        self.frames = []

        for i in range(0, int(self.rate / self.chunk * seconds)):
            data = self.stream.read(self.chunk)
            self.frames.append(data)

        print("[AUDIO] * done recording")

    def record_audio_async(self, start_event, stop_event):
        start_event.wait()


        print("[AUDIO] * recording")
        self.frames = []

        start_time = time.perf_counter()
        while not stop_event.is_set():
            data = self.stream.read(self.chunk)
            self.frames.append(data)

        end_time = time.perf_counter()
        print("[AUDIO] * done recording")

        print("[AUDIO] Start Time: ", start_time)
        print("[AUDIO] End Time: ", end_time)
        print("[AUDIO] Audio Duration: ", end_time - start_time)

        self.record_start_time = start_time
        self.record_end_time = end_time
        self.record_duration = end_time - start_time

    def read_buffer(self):
        return self.stream.read(self.chunk)

    def terminate(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

    def save_audio(self, output_path):

        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))

        print("[AUDIO] Quantity of frames: ", len(self.frames))

    def get_audio_stream_data(self) -> NDArray:
        # Join all byte chunks together and convert them to 16-bit integers
        audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        print("[AUDIO] Size of audio_data:", audio_data.shape)
        return audio_data

    def plot_audio_stream(self, audio_data, seconds : float | None = None):

        # Create a time axis (in seconds) for the X-axis
        if seconds is not None:
            time_axis = np.linspace(0, seconds, num=len(audio_data))

        print("[AUDIO] Size of audio detection: ", audio_data.shape)
        print("[AUDIO]* plotting...")
        plt.figure(figsize=(10, 6))

        if seconds is not None:
            plt.plot(time_axis, audio_data, color='red', alpha=0.7)
        else:
            plt.plot(audio_data, color='red', alpha=0.7)

        plt.title("Right Channel Amplitude")
        plt.ylabel("Amplitude (16-bit)")
        plt.xlabel("Time (seconds)" if seconds is not None else "Time (Unit)")
        plt.grid(True)
        plt.ylim(-32768, 32767)

        plt.tight_layout()
        plt.show()

        plt.close()

    def spectrogram(self, audio_data, dbs = None):
        nperseg = int((8*self.rate) / (7*self.time_samples_per_seg + 1))

        f, t, Sxx = signal.spectrogram(audio_data, self.rate, nperseg=nperseg) # Standard value 255

        if dbs or (dbs is None and self.dbs):
            Sxx = 10 * np.log10(Sxx + 1e-10)

        print("Frequency size: ", f.shape)
        print("Time quantization: ", t.shape)
        print("Shape of Spectrogram: ",Sxx.shape)

        return f, t, Sxx

    def plot_spectrogram(self, frequency, time_stamps, spectrogram, filepath:str, show=True):
        plt.figure()
        plt.pcolormesh(time_stamps, frequency, spectrogram)
        plt.ylabel('Frequency [dBS]')
        plt.xlabel('Time [sec]')
        plt.savefig(filepath)
        if show:
            plt.show()
        plt.close()

    def detection_frequency(self, spectrogram, frequency, frames_offset:int=20, save_plot : Path = Path("amplitude_threshold.jpg")):

        frequence_with_max_amp = np.argmax(spectrogram, axis=0)

        f_idx = stats.mode(frequence_with_max_amp)[0]

        alarm_detection = frequence_with_max_amp == f_idx
        alarm_detection = alarm_detection[frames_offset:] # Small delay to initialize audio sensor

        alarm_frequency_amplitudes = spectrogram[f_idx,frames_offset:]

        amplitude = np.quantile(alarm_frequency_amplitudes[alarm_detection], q=0.05)

        time_axis = np.arange(frames_offset, spectrogram.shape[1])

        plt.figure()
        plt.scatter(time_axis[alarm_detection], 
                    alarm_frequency_amplitudes[alarm_detection], 
                    color='red', label='Alarm', s=15)
    
        plt.scatter(time_axis[~alarm_detection], 
                    alarm_frequency_amplitudes[~alarm_detection], 
                    color='blue', label='Noise', s=15)

        plt.hlines(amplitude, 0, len(time_axis), linestyle='dashed', color='black', label=f'Min Amp Threshold')
        plt.title("Detecting amplitude threshold")
        plt.savefig(str(save_plot))
        plt.show()
        plt.close()

        return frequency[f_idx], amplitude

    def convolution_1d(self, data, kernel):
        kernel_size = kernel.shape[1]
        padded_data = np.pad(data, pad_width=((0,0),(int(kernel_size/2), int(kernel_size/2))), mode='edge')
        convolved_1d = np.zeros(data.shape[1], dtype=padded_data.dtype)

        for i in range(data.shape[1]):
            convolved_1d[i] = np.sum(padded_data[:,i:i+kernel_size] * kernel)

        return convolved_1d

    def normalized_cross_correlation(self, data, kernel):
        kernel_size = kernel.shape[1]
        padded_data = np.pad(data, pad_width=((0,0),(int(kernel_size/2), int(kernel_size/2))), mode='edge')
        correlated_1d = np.zeros(data.shape[1], dtype=padded_data.dtype)

        centered_kernel = kernel - np.mean(kernel)

        for i in range(data.shape[1]):
            centered_patch = padded_data[:,i:i+kernel_size] - np.mean(padded_data[:,i:i+kernel_size])
            norm_product = np.linalg.norm(centered_patch) * np.linalg.norm(centered_kernel)

            if norm_product != 0:
                correlated_1d[i] = np.sum(centered_patch * centered_kernel) / norm_product
            else:
                correlated_1d[i] = 0.0

        return correlated_1d

    def detection_frequency_correlation(self, spectrogram, frequency, frames_offset:int=20, save_plot : Path = Path("amplitude_threshold.jpg")):
        frequence_with_max_amp = np.argmax(spectrogram, axis=0)
        
        f_idx = stats.mode(frequence_with_max_amp)[0]

        alarm_detection = frequence_with_max_amp == f_idx
        alarm_detection = alarm_detection[frames_offset:] # Small delay to initialize audio sensor
        spectrogram_with_delay = spectrogram[:,frames_offset:]

        alarm_frequency_amplitudes = spectrogram_with_delay[f_idx,:]

        amplitude_for_conv_kernel = np.quantile(alarm_frequency_amplitudes[alarm_detection], q=0.90, method='nearest')
        print("Amplitude for convolutional kernel: ", amplitude_for_conv_kernel)

        # Defining pattern matching kernel
        kernel_size = int(self.time_samples_per_seg/10)
        start_time_idx = np.where(alarm_frequency_amplitudes == amplitude_for_conv_kernel)[0][0]
        end_time_idx = start_time_idx + kernel_size

        pattern_matching_kernel = spectrogram_with_delay[:, start_time_idx:end_time_idx]

        self.plot_spectrogram(frequency=frequency, 
                                      time_stamps=np.arange(0, pattern_matching_kernel.shape[1]), 
                                      spectrogram=pattern_matching_kernel, 
                                      filepath = str(save_plot.parent / "pattern_convolution.jpg"))
        print("Max: ", np.max(pattern_matching_kernel), " Min: ", np.min(pattern_matching_kernel))

        # Cross correlation
        correlated_spec_1d = self.normalized_cross_correlation(spectrogram_with_delay, pattern_matching_kernel)

        # Relationship between convolution result and the time stamps where the frequency with maximal amplitude was the alarm frequency
        correlation_score_threshold = np.quantile(correlated_spec_1d[alarm_detection], q=0.08, method='nearest')
        print("Correlation Score Threshold: ", correlation_score_threshold)

        # Plot 
        time_axis = np.arange(frames_offset, spectrogram_with_delay.shape[1] + frames_offset)

        plt.figure()
        plt.scatter(time_axis[alarm_detection], 
                    alarm_frequency_amplitudes[alarm_detection], 
                    color='red', label='Alarm', s=15)
    
        plt.scatter(time_axis[~alarm_detection], 
                    alarm_frequency_amplitudes[~alarm_detection], 
                    color='blue', label='Noise', s=15)

        plt.hlines(amplitude_for_conv_kernel, 0, len(time_axis), linestyle='dashed', color='black', label=f'Min Amp Threshold')
        plt.title("Detecting amplitude threshold")
        plt.savefig(str(save_plot))
        plt.show()
        plt.close()

        plt.figure()
        plt.scatter(time_axis[alarm_detection], 
                    correlated_spec_1d[alarm_detection], 
                    color='red', label='Alarm', s=15)
            
        plt.scatter(time_axis[~alarm_detection], 
                    correlated_spec_1d[~alarm_detection], 
                    color='blue', label='Noise', s=15)
        plt.hlines(correlation_score_threshold, 0, len(time_axis), linestyle='dashed', color='black', label=f'Min Threshold')
        plt.title("Detecting convolution threshold")
        plt.savefig(str(save_plot.parent / "convolution_threshold.jpg"))
        plt.show()
        plt.close()

        return frequency[f_idx], float(correlation_score_threshold), pattern_matching_kernel

    
    def get_binary_detection(self, audio_data, alarm_frequency:float, amp_threshold:float, interval:float, save_plot : Path | None = None):

        f, t, Sxx = self.spectrogram(audio_data=audio_data)

        inf_detection = (1.0-interval)*alarm_frequency
        sup_detection = (1.0+interval)*alarm_frequency


        frequencies_idx_sup = f >= inf_detection 
        frequencies_idx_inf = f <= sup_detection
        frequencies_idx = frequencies_idx_inf & frequencies_idx_sup

        alarm_spectrogram = Sxx[frequencies_idx, :]

        plt.figure()
        plt.subplot(2,1,1)
        max_amplitude = np.max(alarm_spectrogram, axis=0)
        plt.title("max_amplitude_at_alarm_frequency")
        plt.plot(max_amplitude)
    
        plt.subplot(2,1,2)
        mean_amplitude = np.mean(alarm_spectrogram, axis=0)
        plt.plot(mean_amplitude)
        plt.title("mean_amplitude_at_alarm_frequency")

        if save_plot is not None:
            plt.savefig(str(save_plot / "Mean_Max_Amplitude_at_Alarm_Frequency.jpg"))

        plt.close()

        max_amplitude_alarm_sxx = np.max(alarm_spectrogram, axis=0)

        binary_detection = max_amplitude_alarm_sxx > amp_threshold

        return binary_detection.astype(int)

    def get_binary_detection_correlation(self, audio_data, conv_threshold:float, convolutional_kernel:NDArray, save_plot : Path | None = None):

        f, t, Sxx = self.spectrogram(audio_data=audio_data)

        correlated_spec_1d = self.normalized_cross_correlation(Sxx, convolutional_kernel)

        plt.figure()
        plt.plot(correlated_spec_1d, color = 'purple')
        plt.hlines(conv_threshold, 0, correlated_spec_1d.shape[0], linestyle='dashed', color='black', label=f'Min Threshold')
        if save_plot is not None:
            plt.savefig(str(save_plot / "Audio_Detection_Convoluted.jpg"))
        plt.close()


        binary_detection = correlated_spec_1d > conv_threshold

        return binary_detection.astype(int)

    def morph_closing(self, binary_detection, struct_size:int):
        structuring_element = np.ones(struct_size, dtype=int) # t = 10 * struct_size ms
        closing_detection = binary_closing(binary_detection, structure=structuring_element, border_value=1).astype(int)

        return closing_detection

    def valid_video_frames(self, video_time_array):        
        frames_after_audio_begin = video_time_array > self.record_start_time
        frames_before_audio_finish = video_time_array < self.record_end_time
        valid_frames = frames_after_audio_begin &  frames_before_audio_finish

        return np.arange(len(video_time_array))[valid_frames]

    def first_audio_data_after_time(self, audio_data:NDArray, time:float, period_in_seconds:float):
        idx = math.ceil(round(time / period_in_seconds, 7))
        return audio_data[idx]


    def resample_detection(self, video_time_array : NDArray, binary_detection : NDArray , audio_start_timestamp : float):
        video_audio_b_detection = np.zeros(video_time_array.shape, dtype=int)
        
        valid_frames = self.valid_video_frames(video_time_array=video_time_array)

        period = self.record_duration / len(binary_detection)

        for frame in valid_frames:
            frame_time = float(video_time_array[frame])
            video_audio_b_detection[frame] = self.first_audio_data_after_time(audio_data=binary_detection, time=frame_time - audio_start_timestamp, period_in_seconds=period)

        return video_audio_b_detection