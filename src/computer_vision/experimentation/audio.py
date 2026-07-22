import pyaudio
import wave
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy import signal
from scipy import stats
from scipy.ndimage import binary_closing

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

time_passed = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    start = time.perf_counter()
    data = stream.read(CHUNK)
    frames.append(data)
    end = time.perf_counter()
    time_passed.append(end - start)

mean_time = np.mean(np.array(time_passed))
sum_time = np.sum(np.array(time_passed))


print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print("How many frames: ", len(frames))

# 1. Join all byte chunks together and convert them to 16-bit integers
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

print("Size of audio_data:", audio_data.shape)

# 2. Since the audio is stereo (2 channels), reshape the 1D array into a 2D array
# This separates the alternating [Left, Right, Left, Right...] values
audio_data = audio_data.reshape(-1, CHANNELS)

# 3. Extract left and right channels
left_channel = audio_data[:, 0]
right_channel = audio_data[:, 1]

# 4. Create a time axis (in seconds) for the X-axis
time_axis = np.linspace(0, RECORD_SECONDS, num=len(left_channel))


print("Size of audio detection: ", right_channel.shape)
print("* plotting...")
plt.figure(figsize=(10, 6))

# Plot Left Channel
plt.subplot(2, 1, 1)
plt.plot(time_axis, left_channel, color='blue', alpha=0.7)
plt.title("Left Channel Amplitude")
plt.ylabel("Amplitude (16-bit)")
plt.xlabel("Time (seconds)")
plt.grid(True)
# 16-bit audio ranges from -32768 to 32767
plt.ylim(-32768, 32767) 

# Plot Right Channel
plt.subplot(2, 1, 2)
plt.plot(time_axis, right_channel, color='red', alpha=0.7)
plt.title("Right Channel Amplitude")
plt.ylabel("Amplitude (16-bit)")
plt.xlabel("Time (seconds)")
plt.grid(True)
plt.ylim(-32768, 32767)

plt.tight_layout()
plt.show()

f, t, Sxx = signal.spectrogram(right_channel, int(len(right_channel)/RECORD_SECONDS), nperseg=int(len(right_channel)/RECORD_SECONDS/100)) #Standard 255
dBS = 10 * np.log10(Sxx)
print(f.shape)
print(t.shape)
print(dBS.shape)
plt.pcolormesh(t, f, dBS)
plt.ylabel('Frequency [dBS]')
plt.xlabel('Time [sec]')
plt.savefig("spectrogram.png", bbox_inches='tight')
plt.show()

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