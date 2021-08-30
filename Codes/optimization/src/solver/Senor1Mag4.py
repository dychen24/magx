import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, minimize, report_fit, Minimizer
import time
import scipy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


u0 = 4 * np.pi * 1e-7  # const permeability of vacuum


def VecM(params, j):  # calculate vector Mj
    theta, phy, M = params['theta{}'.format(
        j)], params['phy{}'.format(j)], params['m{}'.format(j)]
    # convert the M, theta and phy to its limited range
    theta = np.pi * sigmoid(theta)
    # theta = np.pi*np.tanh(theta)
    phy = np.pi * np.tanh(phy)
    M = np.exp(M)
    return 1e-7 * M * np.array([np.sin(theta) * np.cos(phy),
                                np.sin(theta) * np.sin(phy),
                                np.cos(theta)])


def VecB(params, pSensor, i=0, j=0):  # calculate Bij
    x, y, z = params['X{}'.format(j)], params['Y{}'.format(
        j)], params['Z{}'.format(j)]
    # the position of the sensor i
    xs, ys, zs = pSensor[i, 0], pSensor[i, 1], pSensor[i, 2]
    vecR = np.stack([xs - x, ys - y, zs - z]).reshape(3, 1)
    vecM = VecM(params, j).reshape(3, 1)
    dis = np.linalg.norm(vecR, 2)
    vecb = 3 * np.matmul(vecR, (np.matmul(vecM.T, vecR))) / \
        np.power(dis, 5) - vecM / np.power(dis, 3)
    return vecb

# Calculate the residual of x, y, z


def objective(params, data, pSensor, mag_count):
    """Calculate total residual for fits of Gaussians to several data sets."""
    ndata, _ = data.shape
    resid = np.zeros_like(data)
    # make residual per data set

    for i in range(ndata):
        G = np.array([params['gx'], params['gy'], params['gz']]).reshape(3, 1)
        M = G
        for j in range(mag_count):
            tmp = VecB(params, pSensor, i, j)
            M += tmp
        resid[i] = data[i] - M.flatten() * 1e6
    # now flatten this to a 1D array, as minimize() needs
    tmp = resid.flatten()
    # tmp = np.sum(np.power(tmp, 2))
    return tmp


class Solver:
    def __init__(self, mag_count=1, p1=-0.04, p2=0.04, p3=0.04):
        self.fit_params = Parameters()
        self.mag_count = mag_count
        self.fit_params.add('gx', value=0)
        self.fit_params.add('gy', value=0)
        self.fit_params.add('gz', value=0)
        for i in range(mag_count):
            self.fit_params.add('X{}'.format(i), value=p1)
            self.fit_params.add('Y{}'.format(i), value=p2)
            self.fit_params.add('Z{}'.format(i), value=p3)
            self.fit_params.add('m{}'.format(i), value=np.log(2), vary=True)
            self.fit_params.add('theta{}'.format(i), value=0.2)
            self.fit_params.add('phy{}'.format(i), value=0)

    def solve(self, data, pSensor, builtin=False):
        if builtin:
            data = np.array([
                [-102.15, 12.01, -135.52],
                [-324.49, 78.12, -377.52],
                [-727.09, 84.12, -406.56],
                [-390.59, 660.99, 174.24]
            ])
            data = np.concatenate([data, data, data], axis=0)
            # Sensor position
            pSensor = 1e-2 * np.array([
                [0, -4.4, 0],
                [0, 0, 0],
                [0, 4.4, 0],
                [-4.5, 0, 0]
            ])
            pSensor = np.concatenate([pSensor, pSensor, pSensor], axis=0)

        t0 = time.time()
        out = minimize(objective, self.fit_params,
                       args=(data, pSensor, self.mag_count), method='leastsq')
        # out = Minimizer.leastsq(objective, self.fit_params,
        #                         args=(data, pSensor))
        # self.fit_params['X'] = out.params['X']
        # self.fit_params['Y'] = out.params['Y']
        # self.fit_params['Z'] = out.params['Z']
        self.fit_params = out.params
        # self.fit_params['theta'].value = np.random.randn()
        # self.fit_params['phy'].value = np.random.randn()
        print(time.time() - t0)
        # report_fit(out.params)
        return out.params


if __name__ == "__main__":
    tmp = Solver()
    data = np.array([
        [78.11, -210.31, 24.20],
        [351.52, -423.63, -183.92],
        [859.29, -120.18, -706.64],
        [-93.14, -688.03, -445.28]
    ])
    pSensor = 1e-2 * np.array([
        [0, -4.4, 0],
        [0, 0, 0],
        [0, 4.4, 0],
        [-4.5, 0, 0]
    ])
    out = tmp.solve(data, pSensor, False)
    report_fit(out)
