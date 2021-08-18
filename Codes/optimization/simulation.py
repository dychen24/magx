from src.simulation import Simu_Test, simulate_2mag_3type, Simu_with_magnet
import numpy as np
from config import pSensor_smt
from multiprocessing import Pool
from config import pSensor_large_smt, pSensor_small_smt, pSensor_median_smt
import os
import matplotlib.pyplot as plt
if __name__ == "__main__":

    pSensor_test = 1e-2 * np.array([
        [4.9, -4.9, -1.63],
        [-4.9, -4.9, -1.63],
        [4.9, 4.9, -1.63],
        [-4.9, 4.9, -1.63],
        [0, 4.9, 1.63],
        [4.9, 0, 1.63],
        [0, -4.9, 1.63],
        [-4.9, 0, 1.63],
    ])
    testmodel = Simu_Test(10, 30, [4, 5, 6, 7, 8], resolution=300)
