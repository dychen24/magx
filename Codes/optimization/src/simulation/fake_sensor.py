import datetime
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
import sympy as sp
from multiprocessing import Pool
import os
from tqdm import tqdm
import cppsolver as cs
from ..filter import Magnet_UKF, Magnet_KF
from ..solver import Solver, Solver_jac


class Simu_Sensor:
    def __init__(self, start, stop, scales, pSensor=None, resolution=100):
        self.scales = scales
        self.M = 2.7
        self.build_route(start, stop, resolution)
        if pSensor is None:
            self.build_psensor()
        else:
            self.pSensor = pSensor
        # self.build_expression()
        self.params = {
            'm': np.log(self.M),
            'theta': 0,
            'phy': 0,
            'gx': 50 / np.sqrt(2) * 1e-6,
            'gy': 50 / np.sqrt(2) * 1e-6,
            'gz': 0,
        }

    def build_route(self, start, stop, resolution):
        # linear route
        theta = 90 / 180.0 * np.pi
        route = np.linspace(start, stop, resolution)
        route = np.stack([route * np.cos(theta), route * np.sin(theta)]).T
        route = np.pad(route, ((0, 0), (1, 0)),
                       mode='constant',
                       constant_values=0)
        self.route = 1e-2 * route

    def test(self):

        pass
