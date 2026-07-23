import pyaudio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils.audio_handler import AudioHandler
import yaml
import argparse

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10

ROOT_DIR = Path(__file__).resolve().parents[2]
WAVE_OUTPUT_PATH = ROOT_DIR / "calibrate" / "alarm.wav"
YAML_OUTPUT_PATH = ROOT_DIR / "src" / "computer_vision" / "config" / "alarm.yaml"

def parse_args(arg_list = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Script to detect the most important frequency of an alarm."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the yaml instance where the detection frequency will be saved."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=YAML_OUTPUT_PATH,
        help="Path to config file where the alarm frequency will be saved."
    )

    return parser.parse_args(arg_list)

def check_exists(name:str, filepath:Path):

    if not filepath.exists():
        return False

    with open(str(filepath), 'r') as yaml_file:
        detect_freq = yaml.safe_load(yaml_file)

    if detect_freq is not None:
        if name in list(detect_freq.keys()):
            return True

    return False

def save_frequency_to_yaml(frequency, name:str, filepath:Path):

    if check_exists(name=name, filepath=filepath):
        x = input("Do you want to change the previous alarm calibration? [y/n] ")
        if x.upper() != 'Y':
            return
    new_frequency = {
        name: frequency
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(str(filepath), "a") as yaml_file:
        yaml.safe_dump(new_frequency, yaml_file, default_flow_style=False)



def calibrate_alarm(arg_list = None):
    args = parse_args(arg_list)

    alarm_name = args.name
    config_path = args.config

    ah = AudioHandler(chunk=CHUNK,
                      format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      nperseg_factor=100)

    ah.record_audio(RECORD_SECONDS)

    ah.terminate()

    ah.save_audio(WAVE_OUTPUT_PATH)

    audio_data = ah.get_audio_stream_data()

    ah.plot_audio_stream(audio_data=audio_data, seconds=RECORD_SECONDS)

    f, t, dbs = ah.spectrogram(audio_data=audio_data,
                            seconds=RECORD_SECONDS)

    ah.plot_spectrogram(frequency=f,
                        time_stamps=t,
                        spectrogram=dbs,
                        filepath = str(WAVE_OUTPUT_PATH.parent / "spectrogram.jpg"))

    plt.subplot(2,1,1)
    max_amplitude = np.max(dbs, axis=0)
    plt.title("max_amplitude")
    plt.plot(max_amplitude)

    plt.subplot(2,1,2)
    mean_amplitude = np.mean(dbs, axis=0)
    plt.plot(mean_amplitude)
    plt.title("mean_amplitude")

    plt.savefig(str(WAVE_OUTPUT_PATH.parent / "mean_max_amplitude.png"))
    plt.show()


    plt.subplot(3,1,1)
    frequence_with_max_amp = np.argmax(dbs, axis=0)
    plt.plot(frequence_with_max_amp)
    plt.title("max_freq_amplitude")


    alarm_frequency = ah.detection_frequency(spectrogram=dbs, frequency=f)

    binary_detection = ah.get_binary_detection(audio_data=audio_data,
                                            alarm_frequency=alarm_frequency,
                                            amp_threshold=1,
                                            interval=0.05,
                                            seconds=RECORD_SECONDS)

    if binary_detection.shape[0] == 0:
        print("Something went wrong with detection and we couldn't find a frequency in the same range " \
        "of your alarm.")
        raise ArithmeticError("Frequency resolution insufficient")
        exit(1)

    closed_b_detection = ah.morph_closing(binary_detection=binary_detection,
                                        struct_size=10)


    plt.subplot(3,1,2)
    plt.plot(binary_detection, color='red')
    plt.title("binary_frequence")

    plt.subplot(3,1,3)
    plt.plot(closed_b_detection, color='red')
    plt.title("after_binary_closing")
    plt.show()

    print(f"Saving the frequency {alarm_frequency} to config folder {config_path} " \
          f"using name {alarm_name}")

    print("Type of frequency ", type(alarm_frequency))
    save_frequency_to_yaml(frequency=float(alarm_frequency), name=alarm_name, filepath=config_path)

if __name__ == "__main__":
    calibrate_alarm()

    # for i in range(1,20):
    #     structuring_element = np.ones(i, dtype=int)
    #     final_detection = binary_closing(binary_detection, structure=structuring_element).astype(int)

    #     plt.plot(final_detection, color='red')
    #     plt.title(f"after_binary_closing_{i}")
    #     plt.show()