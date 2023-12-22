import numpy as np
from src.preprocess import Calibrate_Data
from src.visualize import show_track_1mag_csv_cpp, show_track_2mag_csv_cpp
from config import pSensor_large_smt
if __name__ == "__main__":
    LM_path = 'Path to the ground truth data collected by leap motion, in CSV'
    Reading_path = 'Path to the data to be investigated, in CSV'
    cali_path = 'live_demo-2023-11-01-13-18-24.csv'

    cali_data = Calibrate_Data(cali_path)
    cali_data.show_cali_result()

    # To track 2 magnets, change the function to show_track_2mag_csv_cpp
    # show_track_1mag_csv_cpp(Reading_path, cali_path,
                            # LM_path, pSensor_large_smt, 3, False)
