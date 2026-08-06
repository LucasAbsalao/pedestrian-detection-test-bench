import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .audio_handler import AudioHandler
import yaml

from.config import CALIBRATE_DIR, ALARM_CONFIG

# Audio Settings
from .config import CHANNELS, CHUNK, RATE, FORMAT

RECORD_SECONDS = 10


class AlarmCalibrator:
    def __init__(self, 
                 name : str, 
                 record_seconds : int = RECORD_SECONDS,
                 output_path : Path = ALARM_CONFIG,
                 chunk : int = CHUNK,
                 format : int = FORMAT,
                 channels : int = CHANNELS,
                 rate : int = RATE
                 ) -> None:
        
        self.alarm_name = name
        self.config_path = output_path

        self.ah = AudioHandler(chunk = chunk,
                               format = format,
                               channels = channels,
                               rate = rate)

        self.record_seconds = record_seconds

        self.save_dir = CALIBRATE_DIR / self.alarm_name
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def check_exists(self, name:str, filepath:Path):
        if not filepath.exists():
            return False

        with open(str(filepath), 'r') as yaml_file:
            detect_freq = yaml.safe_load(yaml_file)

        if detect_freq is not None:
            if name in list(detect_freq.keys()):
                return True

        return False

    def save_frequency_to_yaml(self, frequency : float, amplitude : float, name:str, filepath:Path):

        new_data = {
            "frequency" : frequency,
            "amplitude" : amplitude
        }

        new_alarm = {
            name: new_data
        }
        
        if self.check_exists(name=name, filepath=filepath):
            x = input("Do you want to change the previous alarm calibration? [y/n] ")
            if x.upper() != 'Y':
                return
            else:
                with open(str(filepath),'r') as yaml_file:
                    yaml_data = yaml.safe_load(yaml_file) or {}

                yaml_data[name] = new_data

                with open(str(filepath), "w") as yaml_file:
                    yaml.safe_dump(yaml_data, yaml_file, default_flow_style=False)

        else:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(str(filepath), "a") as yaml_file:
                yaml.safe_dump(new_alarm, yaml_file, default_flow_style=False)



    def calibrate(self):

        self.ah.record_audio(self.record_seconds)

        self.ah.terminate()

        self.ah.save_audio(self.save_dir / "alarm.wav")

        audio_data = self.ah.get_audio_stream_data()

        self.ah.plot_audio_stream(audio_data=audio_data, seconds=self.record_seconds)

        f, t, dbs = self.ah.spectrogram(audio_data=audio_data)

        self.ah.plot_spectrogram(frequency=f,
                            time_stamps=t,
                            spectrogram=dbs,
                            filepath = str(self.save_dir / "spectrogram.png"))

        plt.subplot(2,1,1)
        max_amplitude = np.max(dbs, axis=0)
        plt.title("max_amplitude")
        plt.plot(max_amplitude)

        plt.subplot(2,1,2)
        mean_amplitude = np.mean(dbs, axis=0)
        plt.plot(mean_amplitude)
        plt.title("mean_amplitude")

        plt.savefig(str(self.save_dir / "mean_max_amplitude.png"))
        plt.show()


        alarm_frequency, amplitude = self.ah.detection_frequency(spectrogram=dbs, frequency=f, 
                                                                 save_plot=self.save_dir / "amplitude_threshold.png")

        binary_detection = self.ah.get_binary_detection(audio_data=audio_data,
                                                        alarm_frequency=alarm_frequency,
                                                        amp_threshold=amplitude,
                                                        interval=0.02,
                                                        seconds=self.record_seconds,
                                                        save_plot=self.save_dir)

        if binary_detection.shape[0] == 0:
            print("Something went wrong with detection and we couldn't find a frequency in the same range " \
            "of your alarm.")
            raise ArithmeticError("Frequency resolution insufficient")
            exit(1)

        closed_b_detection = self.ah.morph_closing(binary_detection=binary_detection,
                                            struct_size=20)

        plt.subplot(3,1,1)
        frequence_with_max_amp = np.argmax(dbs, axis=0)
        plt.plot(frequence_with_max_amp)
        plt.title("frequence_with_maximum_amplitude")

        plt.subplot(3,1,2)
        plt.plot(binary_detection, color='red')
        plt.title("binary_frequence")

        plt.subplot(3,1,3)
        plt.plot(closed_b_detection, color='red')
        plt.title("after_binary_closing")
        plt.savefig(str(self.save_dir / "detected_alarm.png"))
        plt.show()

        print(f"Saving the frequency {alarm_frequency} to config folder {self.config_path} " \
            f"using name {self.alarm_name}")

        print("Type of frequency ", type(alarm_frequency))
        self.save_frequency_to_yaml(frequency=float(alarm_frequency), amplitude=float(amplitude), name=self.alarm_name, filepath=self.config_path)