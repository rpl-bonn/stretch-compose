import sys
import time

from point_cloud_scripts import extract_point_cloud
from point_cloud_scripts import full_align
from utils.openmask_interface import get_mask_clip_features


def convert_time(nanoseconds: int) -> (int, int):
    """
    Converts nanoseconds into minutes and seconds.
    :param nanoseconds: Time in nanoseconds.
    :return: Time as tuple of minutes and seconds.
    """
    minutes = int(nanoseconds / 1e9 // 60)
    seconds = int(nanoseconds % 60)

    return minutes, seconds


def main() -> None:
    # Extract point cloud from autowalk/low_res
    try:
        point_clouds_start = time.time_ns()
        extract_point_cloud.main(sys.argv[1:])
        point_clouds_end = time.time_ns()
        minutes, seconds = convert_time(point_clouds_end - point_clouds_start)
        print(f"Successfully created point_clouds (time: {minutes}min {seconds}s).")
    except FileNotFoundError:
        print("Error: The autowalk/low_res folder does not exist.")

    # Align high_res and low_res point clouds
    try:
        align_start = time.time_ns()
        full_align.main()
        align_end = time.time_ns()
        minutes, seconds = convert_time(align_end - align_start)
        print(f"Successfully created aligned_point_clouds (time: {minutes}min {seconds}s).")
    except FileNotFoundError:
        print("Error: The prescans/high_res folder does not exist.")

    # Run OpenMask3D Segmentation
    # DON'T FORGET to run openmask docker: docker run -p 5001:5001 --gpus all -it craiden/openmask:v1.0 python3 app.py
    try:
        openmask_features_start = time.time_ns()
        get_mask_clip_features()
        openmask_features_end = time.time_ns()
        minutes, seconds = convert_time(openmask_features_end - openmask_features_start)
        print(f"Successfully created openmask_features (time: {minutes}min {seconds}s).")
    except ConnectionError:
        print("Error: Failed to establish connection to port 5001.")


if __name__ == "__main__":
    main()
