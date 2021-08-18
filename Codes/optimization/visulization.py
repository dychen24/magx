from src.visualize.visualization_cpp import show_track_1mag_csv_cpp_still
import numpy as np
from src.filter import Magnet_KF
from src.solver import Solver
from src.preprocess import Calibrate_Data, Reading_Data, LM_data_2mag
from src.visualize import show_track_1mag_csv_cpp, show_track_2mag_csv_cpp
from config import pSensor_smt, pSensor_large_smt, pSensor_median_smt, pSensor_small_smt, pSensor_imu, pSensor_imu_median
import threading
import os
import matplotlib.pyplot as plt
if __name__ == "__main__":
    LM_path = 'Path to the ground truth data collected by leap motion, in CSV'
    Reading_path = 'Path to the data to be investigated, in CSV'
    cali_path = 'Path to the calibration data, in CSV'

    cali_data = Calibrate_Data(cali_path)
    cali_data.show_cali_result()

    # To track 2 magnets, change the function to show_track_2mag_csv_cpp
    show_track_1mag_csv_cpp(Reading_path, cali_path,
                            LM_path, pSensor_large_smt, 3, False)
