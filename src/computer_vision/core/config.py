from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]  # project root
SRC_DIR = ROOT_DIR / "src" / "computer_vision"
DATA_DIR = ROOT_DIR / "data"
ANNOTATION_DIR = DATA_DIR / "annotations"
VIDEO_DIR = DATA_DIR / "videos"
PREDICT_DIR = DATA_DIR / "predict"
DISTORTION_DIR = VIDEO_DIR / "distortions"
MODELS_DIR = SRC_DIR / "models"
CONFIG_DIR = SRC_DIR / "config"
UTILS_DIR = SRC_DIR / "utils"
EVALUATIONS_DIR = ROOT_DIR / "evaluations"
CALIBRATE_DIR = ROOT_DIR / "calibrate"
DEPTH_ANYTHING_DIR = ROOT_DIR / "Depth-Anything-V2"
FRAMES_DIR = ROOT_DIR / "frames"

# Default model paths
DEFAULT_YOLO_MODEL = MODELS_DIR / "yolo26x.pt"
DEFAULT_DEPTH_MODEL = MODELS_DIR / "depth_anything_v2_metric_vkitti_vitl.pth"

# Default config paths
ALARM_CONFIG = CONFIG_DIR / "alarm.yaml"
PARAMETERS_CONFIG = CONFIG_DIR / "parameters.yaml"
POINTS_CONFIG = DATA_DIR / "points.yaml"

# Video settings
DEFAULT_FPS = 30
DEFAULT_CODEC = "libx265_rawvideo"

# Audio Settings
import pyaudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100