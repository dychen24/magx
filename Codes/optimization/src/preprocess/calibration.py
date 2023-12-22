import numpy as np
from .data_reader import read_data


def calibrate(path):
    data = read_data(path)
    # cut = int(data.shape[0]/10)
    # data = data[cut: -cut]
    nsensor = int(data.shape[1] / 3)
    offX = np.zeros(nsensor)
    offY = np.zeros(nsensor)
    offZ = np.zeros(nsensor)
    scaleX = np.zeros(nsensor)
    scaleY = np.zeros(nsensor)
    scaleZ = np.zeros(nsensor)
    for i in range(nsensor):
        mag = data[:, i * 3:i * 3 + 3]
        H = np.array([mag[:, 0], mag[:, 1], mag[:, 2], - mag[:, 1]
                      ** 2, - mag[:, 2] ** 2, np.ones_like(mag[:, 0])]).T
        w = mag[:, 0] ** 2
        tmp = np.matmul(np.linalg.inv(np.matmul(H.T, H)), H.T)
        X = np.matmul(np.linalg.inv(np.matmul(H.T, H)), H.T).dot(w)
        # print(X.shape)
        offX[i] = X[0] / 2
        offY[i] = X[1] / (2 * X[3])
        offZ[i] = X[2] / (2 * X[4])
        temp = X[5] + offX[i] ** 2 + X[3] * offY[i]**2 + X[4] * offZ[i] ** 2
        scaleX[i] = np.sqrt(temp)
        scaleY[i] = np.sqrt(temp / X[3])
        scaleZ[i] = np.sqrt(temp / X[4])
    offset = np.stack([offX, offY, offZ], axis=0).T
    offset = offset.reshape(1, -1)
    scale = np.stack([scaleX, scaleY, scaleZ], axis=0).T
    scale = scale.reshape(1, -1)
    return [offset, scale]
