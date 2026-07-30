import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import time
from pathlib import Path
from scipy import signal
from scipy import stats
from scipy.ndimage import binary_closing


class AudioHandler:
    def __init__(self, 
                 chunk : int = 1024,
                 format : int = pyaudio.paInt16,
                 channels : int = 1,
                 rate : int = 44100,
                 nperseg_factor : int = 100,
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
        self.nperseg = nperseg_factor
        

    def record_audio(self, seconds):

        print("[AUDIO] * recording")

        self.frames = []

        for i in range(0, int(self.rate / self.chunk * seconds)):
            data = self.stream.read(self.chunk)
            self.frames.append(data)

        print("[AUDIO] * done recording")

    def record_audio_async(self, start_event, stop_event):
        start_event.wait()

        start_time = time.perf_counter()

        print("[AUDIO] * recording")
        self.frames = []

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

    def spectrogram(self, audio_data, seconds : float, dbs = None):
        f, t, Sxx = signal.spectrogram(audio_data, int(len(audio_data)/seconds), nperseg=int(len(audio_data)/seconds/self.nperseg)) #Standard 255

        if dbs or (dbs is None and self.dbs):
            Sxx = 10 * np.log10(Sxx)

        print("Frequency size: ", f.shape)
        print("Time quantization: ", t.shape)
        print("Shape of Spectrogram: ",Sxx.shape)

        return f, t, Sxx

    def plot_spectrogram(self, frequency, time_stamps, spectrogram, filepath:str, show=True):
        plt.pcolormesh(time_stamps, frequency, spectrogram)
        plt.ylabel('Frequency [dBS]')
        plt.xlabel('Time [sec]')
        plt.savefig(filepath)
        if show:
            plt.show()

    def detection_frequency(self, spectrogram, frequency, frames_delay:int=20):

        frequence_with_max_amp = np.argmax(spectrogram, axis=0)

        f_idx = stats.mode(frequence_with_max_amp)[0]

        alarm_detection = frequence_with_max_amp == f_idx
        alarm_detection = alarm_detection[frames_delay:] # Small delay to initialize audio sensor

        alarm_frequency_amplitudes = spectrogram[f_idx,frames_delay:]

        amplitude = np.quantile(alarm_frequency_amplitudes[alarm_detection], q=0.25)

        time_axis = np.arange(20, spectrogram.shape[1])
        plt.scatter(time_axis[alarm_detection], 
                    alarm_frequency_amplitudes[alarm_detection], 
                    color='red', label='Alarm', s=15)
    
        plt.scatter(time_axis[~alarm_detection], 
                    alarm_frequency_amplitudes[~alarm_detection], 
                    color='blue', label='Noise', s=15)

        plt.hlines(amplitude, 0, len(time_axis), linestyle='dashed', color='black', label=f'Min Amp Threshold')
        plt.title("Detecting amplitude threshold")
        plt.show()

        return frequency[f_idx], amplitude
    
    def get_binary_detection(self, audio_data, alarm_frequency, amp_threshold:float, interval:float, seconds:float):

        f, t, Sxx = self.spectrogram(audio_data=audio_data, seconds=seconds)

        inf_detection = (1.0-interval)*alarm_frequency
        sup_detection = (1.0+interval)*alarm_frequency


        frequencies_idx_sup = f >= inf_detection 
        frequencies_idx_inf = f <= sup_detection
        frequencies_idx = frequencies_idx_inf & frequencies_idx_sup

        alarm_spectrogram = Sxx[frequencies_idx, :]

        plt.subplot(2,1,1)
        max_amplitude = np.max(alarm_spectrogram, axis=0)
        plt.title("max_amplitude")
        plt.plot(max_amplitude)
    
        plt.subplot(2,1,2)
        mean_amplitude = np.mean(alarm_spectrogram, axis=0)
        plt.plot(mean_amplitude)
        plt.title("mean_amplitude")
        plt.show()

        max_amplitude_alarm_sxx = np.max(alarm_spectrogram, axis=0)

        binary_detection = max_amplitude_alarm_sxx > amp_threshold

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

    def time_interpolation(self, video_time_array, binary_detection):
        print(video_time_array)
        video_audio_b_detection = np.zeros(video_time_array.shape, dtype=int)
        
        valid_frames = self.valid_video_frames(video_time_array=video_time_array)
        print("Valid frames: ", valid_frames)

        second_per_detection_unit = self.record_duration / len(binary_detection)

        for frame in valid_frames:
            frame_time = video_time_array[frame] - video_time_array[0]
            print("Frame time: ", frame_time)
            print("second per detection unit: ", second_per_detection_unit)
            idx_detection = int(frame_time / second_per_detection_unit) + 1
            video_audio_b_detection[frame] = binary_detection[idx_detection]

        return video_audio_b_detection

            
