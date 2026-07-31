import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT_DIRECTORY = Path(__file__).resolve().parents[3]
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from benchmark.stats import Stats


s = Stats(general=True)

ground_truth = np.zeros(200, dtype=int)
ground_truth[50:71] = 1
ground_truth[100:101] = 1
ground_truth[130:171] = 1
ground_truth[175:200] = 1

prediction = np.ones(200, dtype=int)


time_stamps = s.get_detection_duration_in_frames(ground_truth, 1)
la_array = s.latency_array(ground_truth=ground_truth,
                           predictions=prediction,
                           time_stamps=time_stamps)

plt.plot(la_array)
plt.show()