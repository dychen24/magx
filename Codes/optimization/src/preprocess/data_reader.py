import numpy as np
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt
from datetime import datetime
from parse import parse
from ..filter import mean_filter
from datetime import datetime

# New method


def data_mapping(data):
    readingi = np.fromstring(data[1:-1], dtype=float, sep=', ')
    return readingi


def time_mapping(t):
    tdata = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
    origin_day = datetime(year=tdata.year,
                          month=tdata.month,
                          day=tdata.day)
    tstamp = (tdata - origin_day).total_seconds()
    return tstamp


def read_data(file_path):
    data = []
    with open(file_path, 'rb') as f:
        i = -1
        flag = False
        nsensor = 0
        for line in f.readlines():
            i += 1
            if (i == 0):
                continue
            line_decoded = line.decode('utf-8')
            if (line_decoded.startswith('#')):
                if not flag and i > 1:
                    nsensor = i - 3
                    flag = True
                continue
            elif ('Hz' in line_decoded):
                continue
            tmp = line_decoded.split(':')[1]
            # print(tmp)
            data.append(float(tmp.split(',')[0]))
            data.append(float(tmp.split(',')[1]))
            data.append(float(tmp.split(',')[2]))

    # frame x (n*3)  n is the number of sensor
    data = np.array(data).reshape(-1, 3 * nsensor)
    print(data.shape)
    return data


def read_calibrate_data(file_path):
    data = []
    with open(file_path, 'rb') as f:
        i = -1
        flag = False
        nsensor = 0
        for line in f.readlines():
            i += 1
            if (i == 0):
                continue
            line_decoded = line.decode('utf-8')
            if (line_decoded.startswith('#')):
                if not flag and i > 1:
                    nsensor = i - 1
                    flag = True
                continue
            elif ('Hz' in line_decoded):
                continue

            result = parse('X: {} 	Y: {} 	Z: {}', line_decoded)
            # print(result[0])
            data.append(float(result[0]))
            data.append(float(result[1]))
            data.append(float(result[2]))

    # frame x (n*3)  n is the number of sensor
    data = np.array(data).reshape(-1, 3 * nsensor)
    print(data.shape)
    return data


class Reading_Data(object):
    def __init__(self, data_path=None, cali_path=None):
        super().__init__()

        if data_path is not None:
            self.df = pd.read_csv(data_path)
            if "IMU" in self.df.columns:
                self.IMU = np.stack(self.df['IMU'].map(data_mapping))
                self.df = self.df.drop(['IMU'], axis=1)
            self.build_dict()
            print("Finish Loading Sensor Reading")

        if cali_path is not None:
            self.cali_data = Calibrate_Data(cali_path)
            self.calibrate()
            print("Finish Calibration")

    def build_dict(self):
        self.nSensor = len(self.df.keys()) - 2
        # old method
        old = False
        if old:
            tstamps = []
            readings = []

            results = []
            pool = Pool()

            for i in range(self.df.shape[0]):
                # self.cal_thread(i)
                results.append(pool.apply_async(self.cal_thread, args=(i, )))
            pool.close()
            pool.join()

            for result in results:
                [tstamp, readingsi] = result.get()
                tstamps.append(tstamp)
                readings.append(readingsi)

            self.tstamps = np.array(tstamps)
            self.readings = np.array(readings)

            # sort the data according the time stamp
            index = np.argsort(self.tstamps)
            self.tstamps = self.tstamps[index]
            self.readings = self.readings[index]
            self.raw_readings = self.readings.copy()
        else:
            self.tstamps = self.df['Time Stamp'].map(time_mapping).to_numpy()

            readings = []
            for i in range(self.nSensor):
                readings.append(
                    np.stack(self.df['Sensor {}'.format(i+1)].map(data_mapping).to_numpy()))
            self.raw_readings = np.concatenate(readings, axis=1)
            self.readings = self.raw_readings.copy()

    def __len__(self):
        return self.readings.shape[0]

    def __getitem__(self, i):
        return [self.tstamps[i], self.readings[i]]

    def shape(self):
        return self.readings.shape

    def cal_thread(self, i):

        str_time = self.df['Time Stamp'][i]
        # ex. 2021-01-11 16:11:48.255801
        tdata = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S.%f')
        origin_day = datetime(year=tdata.year,
                              month=tdata.month,
                              day=tdata.day)
        tstamp = (tdata - origin_day).total_seconds()

        readings = []
        datai = self.df[self.df.columns[2:]].iloc[i]
        for iSensor in range(self.nSensor):
            readingi = datai['Sensor {}'.format(iSensor + 1)]
            readingi = np.fromstring(readingi[1:-1], dtype=float, sep=', ')
            readings.append(readingi)

        readings = np.concatenate(readings, axis=0)

        return [tstamp, readings]

    def calibrate(self):
        [offset, scale] = self.cali_data.cali_result()
        data = self.raw_readings.copy()

        # TODO: shange mean to 50
        print(np.mean(scale))
        offset = offset.flatten()
        scale = scale.flatten()
        self.offset = offset
        self.scale = scale
        # tmp = np.mean(scale.reshape(-1, 3), axis=1, keepdims=True)
        data = (data - offset) / scale * np.mean(scale)
        # data = (data - offset) / scale * 48
        # for i in range(data.shape[1]):
        #     data[:, i] = mean_filter(data[:, i], win=3)

        self.readings = data.copy()
        # self.readings = self.readings[int(0.15 * self.readings.shape[0]
        #                                   ):int(0.99 * self.readings.shape[0])]
        # self.tstamps = self.tstamps[int(0.15 * self.tstamps.shape[0]
        #                                 ):int(0.99 * self.tstamps.shape[0])]

    def show_cali_result(self):
        self.cali_data.show_cali_result()


class Calibrate_Data:
    def __init__(self, data_path):
        super().__init__()
        self.df = pd.read_csv(data_path)
        if "IMU" in self.df.columns:
            self.df = self.df.drop(['IMU'], axis=1)
        self.build_dict()

    def build_dict(self):
        self.nSensor = len(self.df.keys()) - 2
        old = False
        if old:
            tstamps = []
            readings = []

            results = []
            pool = Pool()

            for i in range(self.df.shape[0]):
                # self.cal_thread(i)
                results.append(pool.apply_async(self.cal_thread, args=(i, )))
            pool.close()
            pool.join()

            for result in results:
                [tstamp, readingsi] = result.get()
                tstamps.append(tstamp)
                readings.append(readingsi)

            self.tstamps = np.array(tstamps)
            self.readings = np.array(readings)
            self.raw_readings = np.array(readings).copy()

            # sort the data according the time stamp
            index = np.argsort(self.tstamps)
            self.tstamps = self.tstamps[index]
            self.readings = self.readings[index]
            # print("Debug")
        else:
            self.tstamps = self.df['Time Stamp'].map(time_mapping).to_numpy()

            readings = []
            for i in range(self.nSensor):
                readings.append(
                    np.stack(self.df['Sensor {}'.format(i+1)].map(data_mapping).to_numpy()))
            self.raw_readings = np.concatenate(readings, axis=1)
            self.readings = self.raw_readings.copy()

    def __len__(self):
        return self.readings.shape[0]

    def __getitem__(self, i):
        return [self.tstamps[i], self.readings[i]]

    def shape(self):
        return self.readings.shape

    def cal_thread(self, i):
        str_time = self.df['Time Stamp'][i]
        # ex. 2021-01-11 16:11:48.255801
        tdata = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S.%f')
        origin_day = datetime(year=tdata.year,
                              month=tdata.month,
                              day=tdata.day)
        tstamp = (tdata - origin_day).total_seconds()

        readings = []
        datai = self.df[self.df.columns[2:]].iloc[i]
        for iSensor in range(self.nSensor):
            readingi = datai['Sensor {}'.format(iSensor + 1)]
            readingi = np.fromstring(readingi[1:-1], dtype=float, sep=', ')
            readings.append(readingi)

        readings = np.concatenate(readings, axis=0)

        return [tstamp, readings]

    def cali_result(self):
        nsensor = self.nSensor
        data = self.readings
        offX = np.zeros(nsensor)
        offY = np.zeros(nsensor)
        offZ = np.zeros(nsensor)
        scaleX = np.zeros(nsensor)
        scaleY = np.zeros(nsensor)
        scaleZ = np.zeros(nsensor)
        for i in range(nsensor):
            mag = data[:, i * 3:i * 3 + 3]
            H = np.array([
                mag[:, 0], mag[:, 1], mag[:, 2], -mag[:, 1]**2, -mag[:, 2]**2,
                np.ones_like(mag[:, 0])
            ]).T
            w = mag[:, 0]**2
            tmp = np.matmul(np.linalg.inv(np.matmul(H.T, H)), H.T)
            X = np.matmul(np.linalg.inv(np.matmul(H.T, H)), H.T).dot(w)
            # print(X.shape)
            offX[i] = X[0] / 2
            offY[i] = X[1] / (2 * X[3])
            offZ[i] = X[2] / (2 * X[4])
            temp = X[5] + offX[i]**2 + X[3] * offY[i]**2 + X[4] * offZ[i]**2
            scaleX[i] = np.sqrt(temp)
            scaleY[i] = np.sqrt(temp / X[3])
            scaleZ[i] = np.sqrt(temp / X[4])
        offset = np.stack([offX, offY, offZ], axis=0).T
        # offset = offset.reshape(1, -1)
        scale = np.stack([scaleX, scaleY, scaleZ], axis=0).T
        # scale = scale.reshape(1, -1)
        # wsy0227 save to numpy
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        np.save(f"mytest/offset-{current_time}.npy",offset)
        np.save(f"mytest/scale-{current_time}", scale)
        return [offset, scale]

    def show_cali_result(self):
        fig = plt.figure('Before Calibration')
        ax = fig.gca(projection='3d')
        ax.set_title("Before Calibration")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        data = self.readings
        for i in range(int(data.shape[1] / 3)):
            datai = data[:, i * 3:i * 3 + 3]
            ax.scatter(datai[:, 0], datai[:, 1], datai[:, 2])

        fig = plt.figure('After Calibration')
        ax = fig.gca(projection='3d')
        ax.set_title("After Calibration")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        data = self.readings
        [offset, scale] = self.cali_result()
        print(offset, '\n', scale)

        offset = offset.reshape(1, -1)
        scale = scale.reshape(1, -1)
        data = self.readings
        data = (data - offset) / scale * np.mean(scale)
        # data = data.reshape(-1, 3*self.nSensor)

        for i in range(int(data.shape[1] / 3)):
            datai = data[:, i * 3:i * 3 + 3]

            ax.scatter(datai[:, 0], datai[:, 1], datai[:, 2])

        # plt.savefig('./result/result.jpg')
        plt.show()


class LM_data:
    # The class to manage the data from leap motion, the raw data is an csv containing the time stamp, tool id,
    # position and direction.

    T_delay = 0.012

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.build_dict()
        self.offset = np.array([0, 0, 0])
        self.ang_offset = np.array([0, 0])
        print("Finish Loading LM gt")

    def build_dict(self):
        old = False
        if old:
            tstamps = []
            positions = []
            directions = []

            results = []
            pool = Pool()

            for i in range(self.df.shape[0]):
                results.append(pool.apply_async(self.cal_thread, args=(i, )))
            pool.close()
            pool.join()

            for result in results:
                [tstamp, tmp, tmp2] = result.get()
                tstamps.append(tstamp)
                positions.append(tmp)
                directions.append(tmp2)

            self.tstamps = np.array(tstamps)
            self.positions = np.array(positions)
            self.directions = np.array(directions)

            # sort the data according the time stamp
            index = np.argsort(self.tstamps)
            self.tstamps = self.tstamps[index]
            self.positions = self.positions[index]
            self.directions = self.directions[index]
        else:
            # New method

            def data_mapping(data):
                readingi = np.fromstring(data[1:-1], dtype=float, sep=', ')
                return readingi

            def time_mapping(t):
                tdata = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
                origin_day = datetime(year=tdata.year,
                                      month=tdata.month,
                                      day=tdata.day)
                tstamp = (tdata - origin_day).total_seconds()
                return tstamp
            self.tstamps = self.df['Time Stamp'].map(time_mapping).to_numpy()

            self.positions = np.stack(
                self.df['Tool Position'].map(data_mapping).to_numpy())
            self.directions = np.stack(
                self.df['Tool Direction'].map(data_mapping).to_numpy())

        '''Post processing'''
        # sensors have a time delay due to BLE
        self.tstamps += LM_data.T_delay

        self.positions = 0.1 * self.positions  # cm

        new_positions = self.positions.copy()
        new_positions[:, 0] = self.positions[:, 2]
        new_positions[:, 1] = self.positions[:, 0]
        new_positions[:, 2] = self.positions[:, 1]

        self.positions = new_positions

        # self.positions += np.array([0, -0.85, 4.25])
        self.positions += np.array([10, 0, 0.8])

        new_direction = self.directions.copy()

        new_direction[:, 0] = self.directions[:, 2]
        new_direction[:, 1] = self.directions[:, 0]
        new_direction[:, 2] = self.directions[:, 1]

        self.directions = new_direction
        tmp = (self.directions.T /
               np.linalg.norm(self.directions, axis=1, ord=2))
        self.positions -= (self.directions.T / np.linalg.norm(
            self.directions, axis=1, ord=2)).T * 1.3
        # print("Debug")

    def get_gt(self, t):
        if isinstance(t, str):
            pass
        # simple Linear Interpolation
        if self.tstamps[-1] < t:
            idx = -1
        else:
            idx = np.where(self.tstamps > t)[0][0]

        if idx == 0:
            return [self.positions[0] - self.offset, self.directions[0]]
        else:
            c1 = (self.tstamps[idx] - t) / \
                (self.tstamps[idx] - self.tstamps[idx - 1])
            c2 = (t - self.tstamps[idx - 1]) / \
                (self.tstamps[idx] - self.tstamps[idx - 1])
            pos = c1 * self.positions[idx - 1] + c2 * self.positions[idx]
            ori = c1 * self.directions[idx - 1] + c2 * self.directions[idx]
            ori = ori / np.linalg.norm(ori, ord=2)
            # angle_gt = np.array([np.arccos(ori[2]),
            #                      np.arctan(ori[1]/ori[0])]) - self.ang_offset
            # ori = np.array([np.sin(angle_gt[0])*np.cos(angle_gt[1]),
            # np.sin(angle_gt[0])*np.sin(angle_gt[1]), np.cos(angle_gt[0])])
            return [pos - self.offset, ori]

    def cal_thread(self, i):
        str_time = self.df["Time Stamp"][i]
        tdata = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S.%f')
        origin_day = datetime(year=tdata.year,
                              month=tdata.month,
                              day=tdata.day)
        tstamp = (tdata - origin_day).total_seconds()
        a = self.df.iloc[i]['Tool Position']
        tmp = np.fromstring(a[1:-1], dtype=float, sep=', ')
        b = self.df.iloc[i]['Tool Direction']
        tmp2 = np.fromstring(b[1:-1], dtype=float, sep=', ')
        return [tstamp, tmp, tmp2]

    def plot_gt_route(self):
        fig = plt.figure('After Calibration')
        ax = fig.gca(projection='3d')
        ax.set_title("After Calibration")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        route = self.positions
        ax.plot(route[:, 0],
                route[:, 1],
                route[:, 2],
                c='g',
                label='Ground Truth Route')


class LM_data_2mag:
    # The class to manage the data from leap motion, the raw data is an csv containing the time stamp, tool id,
    # position and direction.

    T_delay = 0.012

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.build_dict()
        # self.my_offset = np.array([-0.04862988,  0.2269497, - 0.02317274])
        self.my_offset = np.array([0, 0, 0])
        self.offset = np.array([0, 0, 0])
        print("Finish Loading LM gt")

    def build_dict(self):
        tstamps1 = []
        positions1 = []
        directions1 = []

        tstamps2 = []
        positions2 = []
        directions2 = []

        results = []

        # Check the id
        tool_ids = set()
        for item in self.df['Tool ID']:
            tool_ids.add(item)
        assert (len(tool_ids) == 2)

        df1 = self.df.loc[self.df['Tool ID'] == tool_ids.pop()]
        df2 = self.df.loc[self.df['Tool ID'] == tool_ids.pop()]

        pool = Pool()
        for i in range(df1.shape[0]):
            # self.cal_thread(i, 1, df1)
            results.append(
                pool.apply_async(self.cal_thread, args=(
                    i,
                    1,
                    df1,
                )))
        for i in range(df2.shape[0]):
            # self.cal_thread(i, 2, df2)
            results.append(
                pool.apply_async(self.cal_thread, args=(
                    i,
                    2,
                    df2,
                )))
            #
        pool.close()
        pool.join()

        for result in results:
            [mag_id, tstamp, tmp, tmp2] = result.get()
            if mag_id == 1:
                tstamps1.append(tstamp)
                positions1.append(tmp)
                directions1.append(tmp2)
            elif mag_id == 2:
                tstamps2.append(tstamp)
                positions2.append(tmp)
                directions2.append(tmp2)

        # post calculation for mag 1
        self.tstamps1 = np.array(tstamps1)
        self.positions1 = np.array(positions1)
        self.directions1 = np.array(directions1)

        # sort the data according the time stamp
        index = np.argsort(self.tstamps1)
        self.tstamps1 = self.tstamps1[index]

        # sensors have a time delay due to BLE
        self.tstamps1 += LM_data.T_delay

        self.positions1 = 0.1 * self.positions1[index]  # cm

        new_positions = self.positions1.copy()
        new_positions[:, 0] = self.positions1[:, 2]
        new_positions[:, 1] = self.positions1[:, 0]
        new_positions[:, 2] = self.positions1[:, 1]

        self.positions1 = new_positions

        # self.positions += np.array([0, -0.85, 4.25])
        self.positions1 += np.array([15, 0, 0.8])

        new_direction = self.directions1.copy()

        new_direction[:, 0] = self.directions1[:, 2]
        new_direction[:, 1] = self.directions1[:, 0]
        new_direction[:, 2] = self.directions1[:, 1]

        self.directions1 = new_direction
        tmp = (self.directions1.T /
               np.linalg.norm(self.directions1, axis=1, ord=2))
        self.positions1 -= (self.directions1.T / np.linalg.norm(
            self.directions1, axis=1, ord=2)).T * 1.3
        # print("Debug")

        # post calculation for mag 2
        self.tstamps2 = np.array(tstamps2)
        self.positions2 = np.array(positions2)
        self.directions2 = np.array(directions2)

        # sort the data according the time stamp
        index = np.argsort(self.tstamps2)
        self.tstamps2 = self.tstamps2[index]

        # sensors have a time delay due to BLE
        self.tstamps2 += LM_data.T_delay

        self.positions2 = 0.1 * self.positions2[index]  # cm

        new_positions = self.positions2.copy()
        new_positions[:, 0] = self.positions2[:, 2]
        new_positions[:, 1] = self.positions2[:, 0]
        new_positions[:, 2] = self.positions2[:, 1]

        self.positions2 = new_positions

        # self.positions += np.array([0, -0.85, 4.25])
        self.positions2 += np.array([15, 0, 0.8])

        new_direction = self.directions2[index].copy()

        new_direction[:, 0] = self.directions2[:, 2]
        new_direction[:, 1] = self.directions2[:, 0]
        new_direction[:, 2] = self.directions2[:, 1]

        self.directions2 = new_direction
        tmp = (self.directions2.T /
               np.linalg.norm(self.directions2, axis=1, ord=2))
        self.positions2 -= (self.directions2.T / np.linalg.norm(
            self.directions2, axis=1, ord=2)).T * 1.3

    def get_gt(self, t):
        result = []
        if isinstance(t, str):
            pass

        # Mag one
        # simple Linear Interpolation
        if self.tstamps1[-1] < t:
            idx = -1
        else:
            idx = np.where(self.tstamps1 > t)[0][0]

        if idx == 0:
            result += [self.positions1[0] - self.offset, self.directions1[0]]
        else:
            c1 = (self.tstamps1[idx] - t) / \
                (self.tstamps1[idx] - self.tstamps1[idx - 1])
            c2 = (t - self.tstamps1[idx - 1]) / \
                (self.tstamps1[idx] - self.tstamps1[idx - 1])
            pos = c1 * self.positions1[idx - 1] + c2 * self.positions1[idx]
            ori = c1 * self.directions1[idx - 1] + c2 * self.directions1[idx]
            result += [pos - self.offset - self.my_offset, ori]

        # Mag Two
        # simple Linear Interpolation
        if self.tstamps2[-1] < t:
            idx = -1
        else:
            idx = np.where(self.tstamps2 > t)[0][0]

        if idx == 0:
            result += [self.positions2[0] - self.offset, self.directions2[0]]
        else:
            c1 = (self.tstamps2[idx] - t) / \
                (self.tstamps2[idx] - self.tstamps2[idx - 1])
            c2 = (t - self.tstamps2[idx - 1]) / \
                (self.tstamps2[idx] - self.tstamps2[idx - 1])
            pos = c1 * self.positions2[idx - 1] + c2 * self.positions2[idx]
            ori = c1 * self.directions2[idx - 1] + c2 * self.directions2[idx]

            result += [pos - self.offset - self.my_offset, ori]
        result[1] = result[1] / np.linalg.norm(result[1], ord=2)
        result[3] = result[3] / np.linalg.norm(result[3], ord=2)
        return result

    def set_offset(self, t, p0, p1, ang0, ang1):
        gt_p = self.get_gt(t)
        offset1 = gt_p[0] - p0
        offset2 = gt_p[2] - p0
        dis11 = gt_p[2] - p1 - offset1
        dis21 = gt_p[0] - p1 - offset2
        d11 = np.linalg.norm(dis11)
        d21 = np.linalg.norm(dis21)
        print('d11 = {}, d21= {}'.format(d11, d21))
        if d11 < d21:
            self.idx = 1  # the mag one coordinate with the gt 1
            self.offset = offset1
        else:
            self.idx = 2  # the mag one coordinate with the gt 2
            self.offset = offset2
        print("The offset is ", self.offset)
        # self.offset =
        # self.idx = 2    # the mag one coordinate with the gt 1
        # self.offset = offset2

    def cal_thread(self, i, mag_id, df):
        # str_time = df["Time Stamp"][i]
        str_time = df.iloc[i]["Time Stamp"]
        tdata = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S.%f')
        origin_day = datetime(year=tdata.year,
                              month=tdata.month,
                              day=tdata.day)
        tstamp = (tdata - origin_day).total_seconds()
        a = df.iloc[i]['Tool Position']
        tmp = np.fromstring(a[1:-1], dtype=float, sep=', ')
        b = df.iloc[i]['Tool Direction']
        tmp2 = np.fromstring(b[1:-1], dtype=float, sep=', ')
        return [mag_id, tstamp, tmp, tmp2]

    def plot_gt_route(self):
        fig = plt.figure('After Calibration')
        ax = fig.gca(projection='3d')
        ax.set_title("After Calibration")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        route = self.positions1
        ax.plot(route[:, 0],
                route[:, 1],
                route[:, 2],
                c='g',
                label='Ground Truth Route')
        route = self.positions2
        ax.plot(route[:, 0],
                route[:, 1],
                route[:, 2],
                c='g',
                label='Ground Truth Route')
        plt.show()
