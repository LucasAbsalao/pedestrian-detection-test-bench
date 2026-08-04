from pathlib import Path
import argparse

from core.config import ALARM_CONFIG
from core.alarm_calibrator import AlarmCalibrator


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
        default=ALARM_CONFIG,
        help="Path to config file where the alarm frequency will be saved."
    )

    return parser.parse_args(arg_list)


def calibrate_alarm():
    args = parse_args()
    alarm_calibrator = AlarmCalibrator(name=args.name,
                                       record_seconds = 10,
                                       output_path = args.config)

    alarm_calibrator.calibrate()

if __name__ == "__main__":
    calibrate_alarm()