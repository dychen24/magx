import numpy as np
import os
import sys
from src.preprocess import Calibrate_Data
from src.visualize import show_track_1mag_csv_cpp, show_track_2mag_csv_cpp
from config import pSensor_large_smt
from utils.read_files import find_latest_file_with_prefix_and_suffix
if __name__ == "__main__":
    # LM_path = 'Path to the ground truth data collected by leap motion, in CSV'
    # Reading_path = 'Path to the data to be investigated, in CSV'
    calib_folder = "datasets/calib_3"
    cali_path = find_latest_file_with_prefix_and_suffix(calib_folder,"calib_3-",".csv")
    cali_path = os.path.join(calib_folder, cali_path)
    cali_data = Calibrate_Data(cali_path)
    cali_data.show_cali_result()

    # To track 2 magnets, change the function to show_track_2mag_csv_cpp
    # show_track_1mag_csv_cpp(Reading_path, cali_path,
    #                         LM_path, pSensor_large_smt, 3, False)
