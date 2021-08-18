import numpy as np

def feature_eng(X, pSensor):
    # some feature engineering
    nsample = X.shape[0]
    newX = X.reshape(-1, 3)
    # reading norm of each sensor
    feature1 = np.linalg.norm(newX, ord=2, axis=1)
    feature2 = X/pSensor.reshape(-1)  # reading divided by the sensor position
    feature3 = (newX.T/feature1.T).T  # normalize

    feature1 = feature1.reshape(nsample, -1)
    feature3 = feature1.reshape(nsample, -1)

    # diff in B / dis
    feature4 = []
    feature5 = []
    for i in range(pSensor.shape[0]):
        for j in range(i+1, pSensor.shape[0]):
            tmp = X[:, i*3:i*3+3] - X[:, j*3:j*3+3]
            diff = np.linalg.norm(
                tmp, ord=2, axis=1).reshape(-1, 1)  # nsample x 1
            dis = np.linalg.norm(
                pSensor[i] - pSensor[j], ord=2)
            feature4.append(diff)
            feature5.append(diff/dis)
    feature4 = np.concatenate(feature4, axis=1)
    feature5 = np.concatenate(feature5, axis=1)
    feature6 = np.linalg.norm(feature4, axis=1, ord=2)
    feature7 = (feature4.T / feature6) .T
    feature8 = np.linalg.norm(feature5, axis=1, ord=2)
    feature9 = (feature5.T / feature8) .T

    feature10 = np.mean(
        X.reshape(-1, pSensor.shape[0], pSensor.shape[1]), axis=1)

    feature11 = np.max(X, axis=1).reshape(-1, 1)
    feature12 = np.min(X, axis=1).reshape(-1, 1)
    feature13 = np.argmax(X, axis=1).reshape(-1, 1)
    feature14 = np.argmin(X, axis=1).reshape(-1, 1)

    sensor_dis = np.linalg.norm(pSensor, axis=1).reshape(-1, 1)
    feature15 = (X.reshape(-1, pSensor.shape[0],
                           pSensor.shape[1])/sensor_dis).reshape(nsample, -1)
    # print(feature1.shape, feature2.shape,
    #       feature3.shape, feature4.shape, feature5.shape)

    XX = np.concatenate([X, feature1, feature2, feature3,
                         feature4, feature5, feature6.reshape(-1, 1), feature7, feature8.reshape(-1, 1), feature9, feature10, feature11, feature12, feature13, feature14, feature15], axis=1)
    # print(XX.shape)
    return XX
