import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from scipy import signal
from scipy import stats
from scipy.ndimage import binary_closing

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10

ROOT_DIR = Path(__file__).resolve().parents[2]
WAVE_OUTPUT_PATH = ROOT_DIR / "calibrate" / "alarm.wav"


plt.subplot(2,1,1)
max_amplitude = np.max(dBS, axis=0)
plt.title("max_amplitude")
plt.plot(max_amplitude)

plt.subplot(2,1,2)
mean_amplitude = np.mean(dBS, axis=0)
plt.plot(mean_amplitude)
plt.title("mean_amplitude")
plt.savefig("mean_max_amplitude.png")
plt.show()


plt.subplot(3,1,1)
frequence_with_max_amp = np.argmax(dBS, axis=0)
plt.plot(frequence_with_max_amp)
plt.title("max_freq_amplitude")


alarm_frequence = stats.mode(frequence_with_max_amp)[0]
binary_detecion_upper = frequence_with_max_amp > 0.9*alarm_frequence
binary_detecion_lower = frequence_with_max_amp < 1.1*alarm_frequence
binary_mask = binary_detecion_lower & binary_detecion_upper

binary_detection = binary_mask.astype(int)

plt.subplot(3,1,2)
plt.plot(binary_detection, color='red')
plt.title("binary_frequence")

structuring_element = np.ones(10, dtype=int) # t = 10 * i ms
final_detection = binary_closing(binary_detection, structure=structuring_element, border_value=1).astype(int)

plt.subplot(3,1,3)
plt.plot(final_detection, color='red')
plt.title("after_binary_closing")
plt.show()

    # for i in range(1,20):
    #     structuring_element = np.ones(i, dtype=int)
    #     final_detection = binary_closing(binary_detection, structure=structuring_element).astype(int)

    #     plt.plot(final_detection, color='red')
    #     plt.title(f"after_binary_closing_{i}")
    #     plt.show()