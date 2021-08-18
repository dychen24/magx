import datetime
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
import sympy as sp
from multiprocessing import Pool
import os
import cppsolver as cs
from tqdm import tqdm

from ..filter import Magnet_UKF, Magnet_KF
from ..solver import Solver, Solver_jac


class Simu_Data:
    def __init__(self, gt, snr, result):
        self.gt = gt
        self.snr = snr
        self.result = result

    def __len__(self):
        return self.gt.shape[0]

    def store(self):
        np.savez('result/test.npz', gt=self.gt, data=self.result)


class expression:
    def __init__(self, mag_count=1):
        if mag_count == 1:
            x, y, z, M, theta, phy, gx, gy, gz, xs, ys, zs = sp.symbols(
                'x, y, z, M, theta, phy, gx, gy, gz, xs, ys, zs', real=True)
            G = sp.Matrix([[gx], [gy], [gz]])
            # theta2 = sp.tanh(theta)
            # phy2 = sp.tanh(phy)
            vecR = sp.Matrix([xs - x, ys - y, zs - z]).reshape(3, 1)
            # vecR = sp.Matrix([x, y, z]).reshape(3, 1)
            dis = sp.sqrt(vecR[0]**2 + vecR[1]**2 + vecR[2]**2)
            # VecM = M*sp.Matrix([sp.sin(theta2)*sp.cos(phy2),
            #                     sp.sin(theta2)*sp.sin(phy2), sp.cos(theta2)])
            VecM = 1e-7 * sp.exp(M) * sp.Matrix([
                sp.sin(theta) * sp.cos(phy),
                sp.sin(theta) * sp.sin(phy),
                sp.cos(theta)
            ])
            VecB = 3 * vecR * (VecM.T * vecR) / dis**5 - VecM / dis**3 + G
            VecB *= 1e6
            # convert to function for faster evaluation
            self.VecB = sp.lambdify(
                [gx, gy, gz, xs, ys, zs, x, y, z, M, theta, phy],
                VecB, 'numpy')

        elif mag_count == 2:
            x0, y0, z0, M0, theta0, phy0, x1, y1, z1, M1, theta1, phy1, gx, gy, gz, xs, ys, zs = sp.symbols(
                'x0, y0, z0, M0, theta0, phy0, x1, y1, z1, M1, theta1, phy1, gx, gy, gz, xs, ys, zs', real=True)
            G = sp.Matrix([[gx], [gy], [gz]])
            # theta2 = sp.tanh(theta)
            # phy2 = sp.tanh(phy)
            x = [x0, x1]
            y = [y0, y1]
            z = [z0, z1]
            M = [M0, M1]
            theta = [theta0, theta1]
            phy = [phy0, phy1]
            VecB = G
            for i in range(mag_count):
                vecR = sp.Matrix(
                    [xs - x[i], ys - y[i], zs - z[i]]).reshape(3, 1)
                # vecR = sp.Matrix([x, y, z]).reshape(3, 1)
                dis = sp.sqrt(vecR[0] ** 2 + vecR[1] ** 2 + vecR[2] ** 2)
                # VecM = M*sp.Matrix([sp.sin(theta2)*sp.cos(phy2),
                # sp.sin(theta2)*sp.sin(phy2), sp.cos(theta2)])
                VecMi = 1e-7 * sp.exp(M[i]) * sp.Matrix([sp.sin(theta[i]) * sp.cos(
                    phy[i]), sp.sin(theta[i]) * sp.sin(phy[i]), sp.cos(theta[i])])
                VecBi = 3 * vecR * (VecMi.T * vecR) / \
                    dis ** 5 - VecMi / dis ** 3
                VecB += VecBi

            VecB = 1e6 * VecB
            # convert to function for faster evaluation
            self.VecB = sp.lambdify(
                [gx, gy, gz, xs, ys, zs, x0, y0, z0, M0, theta0, phy0, x1, y1,
                 z1, M1, theta1, phy1],
                VecB, 'numpy')


class Result_Handler:
    def __init__(self, simu_data, scale):
        self.track_result = []
        self.simu_data = simu_data
        self.scale = scale

    def __add__(self, new):
        self.track_result.append(new)
        return self

    def get_gt_result(self):
        a = self.simu_data.gt
        b = []
        for i in range(len(self.track_result)):
            b.append(np.array([
                self.track_result[i]['X0'], self.track_result[i]['Y0'],
                self.track_result[i]['Z0']
            ]))
        b = np.stack(b)

        return [a, b]

    def cal_loss(self):
        dist = []
        loss = []
        for i in range(len(self.simu_data)):
            point_gt = self.simu_data.gt[i]
            point_estimate = np.array([
                self.track_result[i]['X0'], self.track_result[i]['Y0'],
                self.track_result[i]['Z0']
            ])
            dist.append(np.linalg.norm(point_gt, 2))
            loss.append(np.linalg.norm(point_gt - point_estimate, 2))

        dist = 1e2 * np.array(dist)
        loss = 1e2 * np.array(loss)
        return [self.scale, dist, loss]

    def gt_and_route(self):
        dist = []
        route = []
        for i in range(len(self.simu_data)):
            point_gt = self.simu_data.gt[i]
            dist.append(np.linalg.norm(point_gt, 2))
            route.append(np.array([
                self.track_result[i]['X0'], self.track_result[i]['Y0'],
                self.track_result[i]['Z0']
            ]))

        dist = np.array(dist)
        route = np.stack(route, axis=0)

        idx = np.argsort(dist)
        gt = self.simu_data.gt[idx]
        route = route[idx]
        return [gt, route]
        # plt.plot(dist, loss, label='scale = {}'.format(self.scale))
        # plt.legend()
        # print('debug')


class Simu_Test:
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

    def build_expression(self):
        x, y, z, M, theta, phy, gx, gy, gz, xs, ys, zs = sp.symbols(
            'x, y, z, M, theta, phy, gx, gy, gz, xs, ys, zs', real=True)
        G = sp.Matrix([[gx], [gy], [gz]])
        # theta2 = sp.tanh(theta)
        # phy2 = sp.tanh(phy)
        vecR = sp.Matrix([xs - x, ys - y, zs - z]).reshape(3, 1)
        # vecR = sp.Matrix([x, y, z]).reshape(3, 1)
        dis = sp.sqrt(vecR[0]**2 + vecR[1]**2 + vecR[2]**2)
        # VecM = M*sp.Matrix([sp.sin(theta2)*sp.cos(phy2),
        #                     sp.sin(theta2)*sp.sin(phy2), sp.cos(theta2)])
        VecM = 1e-7 * sp.exp(M) * sp.Matrix([
            sp.sin(theta) * sp.cos(phy),
            sp.sin(theta) * sp.sin(phy),
            sp.cos(theta)
        ])
        VecB = 3 * vecR * (VecM.T * vecR) / dis**5 - VecM / dis**3 + G
        VecB *= 1e6
        # convert to function for faster evaluation
        self.VecB = sp.lambdify(
            [gx, gy, gz, xs, ys, zs, x, y, z, M, theta, phy], VecB, 'numpy')

    def build_route(self, start, stop, resolution):
        # linear route
        theta = 90 / 180.0 * np.pi
        route = np.linspace(start, stop, resolution)
        route = np.stack([route * np.cos(theta), route * np.sin(theta)]).T
        route = np.pad(route, ((0, 0), (1, 0)),
                       mode='constant',
                       constant_values=0)
        self.route = 1e-2 * route

        # curvy route
        tmp = np.linspace(start, stop, resolution)
        route = np.stack([np.sin((tmp-start)/(stop-start) * np.pi * 5),
                          np.cos((tmp-start)/(stop-start) * np.pi * 5), tmp], axis=0).T
        self.route = 1e-2 * route

    def build_psensor(self):
        self.pSensor = 1e-2 * np.array([
            [1, 1, 1],
            [-1, 1, 1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
            [1, -1, -1],
        ])

    def simulate_process(self, scale):
        print(scale)
        pSensori = scale * self.pSensor
        simu = self.estimate_B(pSensori)
        simu.store()
        model = Solver_jac(1, self.route[0, 0], self.route[0, 1],
                           self.route[0, 2])
        model.fit_params['m0'].value = np.log(self.M)
        model.fit_params['m0'].vary = False

        results = Result_Handler(simu, scale)
        for i in range(simu.result.shape[0]):
            datai = simu.result[i].reshape(-1, 3)
            result = model.solve(datai, pSensori,
                                 not model.fit_params['m0'].vary)
            results += result
        return results.cal_loss()

    def gt_and_result(self):
        pSensori = 1 * self.pSensor
        simu = self.estimate_B(pSensori)
        simu.store()
        model = Solver_jac(1, self.route[0, 0], self.route[0, 1],
                           self.route[0, 2])
        model.fit_params['m0'].value = np.log(self.M)
        model.fit_params['m0'].vary = False

        results = Result_Handler(simu, 1)
        for i in range(simu.result.shape[0]):
            datai = simu.result[i].reshape(-1, 3)
            result = model.solve(datai, pSensori,
                                 not model.fit_params['m0'].vary)
            results += result
        return results.get_gt_result()

    def compare_noise_thread(self, choice):

        scale = 5
        pSensori = scale * self.pSensor

        if choice == 1:
            simu = self.estimate_B(pSensori)
        elif choice == 0:
            simu = self.estimate_B_even_noise(pSensori)
        elif choice == 2:
            simu = self.estimate_B_singular_noise(pSensori)

        model = Solver_jac(1, self.route[0, 0], self.route[0, 1],
                           self.route[0, 2])
        model.fit_params['m0'].value = np.log(self.M)
        model.fit_params['m0'].vary = False

        results = Result_Handler(simu, scale)
        for i in range(simu.result.shape[0]):
            datai = simu.result[i].reshape(-1, 3)
            result = model.solve(datai, pSensori,
                                 not model.fit_params['m0'].vary)
            results += result
        [tmp, dist, loss] = results.cal_loss()
        return [choice, dist, loss]

    def compare_3_noise(self, loop):
        results = []
        pool = Pool()
        for i in range(loop):
            # self.calculate_process(scale)
            results.append(
                pool.apply_async(self.compare_noise_thread, args=(0, )))
            results.append(
                pool.apply_async(self.compare_noise_thread, args=(1, )))
            results.append(
                pool.apply_async(self.compare_noise_thread, args=(2, )))
        pool.close()
        pool.join()

        # print('debug')

        loss_dict = {}
        dist_dict = {}
        for result in results:
            [scale, dist, loss] = result.get()
            if not str(scale) in loss_dict.keys():
                loss_dict[str(scale)] = loss
                dist_dict[str(scale)] = dist
            else:
                loss_dict[str(scale)] += loss

        msg = ['Even Noise', 'Raw Noise', 'Single Noise']

        for key in dist_dict.keys():
            plt.plot(dist_dict[key],
                     loss_dict[key] / loop,
                     label=msg[int(key)])
            plt.legend()

        plt.ylabel('Error(cm)')
        plt.xlabel('Distance(cm)')
        plt.savefig('result/butterfly.jpg', dpi=900)

    def compare_noise_type(self, loop):
        results = []
        pool = Pool()
        for i in range(loop):
            # self.calculate_process(scale)
            results.append(
                pool.apply_async(self.compare_noise_type_thread, args=(0, )))
            results.append(
                pool.apply_async(self.compare_noise_type_thread, args=(1, )))
            results.append(
                pool.apply_async(self.compare_noise_type_thread, args=(2, )))
        pool.close()
        pool.join()

        # print('debug')

        loss_dict = {}
        dist_dict = {}
        for result in results:
            [scale, dist, loss] = result.get()
            if not str(scale) in loss_dict.keys():
                loss_dict[str(scale)] = loss
                dist_dict[str(scale)] = dist
            else:
                loss_dict[str(scale)] += loss

        msg = ['ALL Noise', 'Only Noise', 'Only Precision']

        for key in dist_dict.keys():
            plt.plot(dist_dict[key],
                     loss_dict[key] / loop,
                     label=msg[int(key)])
            plt.legend()

        plt.ylabel('Error(cm)')
        plt.xlabel('Distance(cm)')
        plt.savefig('result/compare_noise_type.jpg', dpi=900)

    def compare_noise_type_thread(self, choice):
        scale = 5
        pSensori = scale * self.pSensor
        simu = self.estimate_B(pSensori, choice)
        model = Solver_jac(1, self.route[0, 0], self.route[0, 1],
                           self.route[0, 2])
        model.fit_params['m0'].value = np.log(self.M)
        model.fit_params['m0'].vary = False

        results = Result_Handler(simu, scale)
        for i in range(simu.result.shape[0]):
            datai = simu.result[i].reshape(-1, 3)
            result = model.solve(datai, pSensori,
                                 not model.fit_params['m0'].vary)
            results += result
        [tmp, dist, loss] = results.cal_loss()
        return [choice, dist, loss]

    def simulate(self, loop=1):
        results = []
        pool = Pool()
        for scale in self.scales:
            # self.calculate_process(scale)
            # test(self, scale)
            for i in range(loop):
                # self.simulate_process(scale)
                results.append(
                    pool.apply_async(self.simulate_process, args=(scale, )))
        pool.close()
        pool.join()

        # print('debug')

        loss_dict = {}
        dist_dict = {}
        for result in results:
            [scale, dist, loss] = result.get()
            if not str(scale) in loss_dict.keys():
                loss_dict[str(scale)] = loss
                dist_dict[str(scale)] = dist
            else:
                loss_dict[str(scale)] += loss

        for key in dist_dict.keys():
            plt.plot(dist_dict[key],
                     loss_dict[key] / loop,
                     label='scale = {} cm'.format(int(key) * 2))
            plt.legend()
        plt.ylabel('Error(cm)')
        plt.xlabel('Distance(cm)')
        name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.savefig('result/compare_scale/{}.jpg'.format(name), dpi=900)

    def simu_readings(self, pSensor):
        simu = self.estimate_B(pSensor, noise_type=3)
        simu.store()

    def simu_gt_and_result(self, pSensor, route, path, name):
        pSensori = pSensor
        simu = self.estimate_B(pSensori, route=route)
        # simu.store()
        # params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, np.log(
        # self.M), 1e-2 * route[0, 0], 1e-2 * (route[0, 1]), 1e-2 * (route[0,
        # 2]), 0, 0])

        model = Solver_jac(1, route[0, 0], route[0, 1], route[0, 2])
        model.fit_params['m0'].value = np.log(self.M)
        model.fit_params['m0'].vary = False
        gt_ang = []
        rec_ang = []
        results = Result_Handler(simu, 1)
        for i in tqdm(range(simu.result.shape[0])):
            datai = simu.result[i].reshape(-1, 3)
            result = model.solve(datai, pSensori,
                                 not model.fit_params['m0'].vary)
            results += result
            gt_ang.append(np.array([0, 0, 1]))
            t1 = result['theta0'].value
            t2 = result['phy0'].value
            rec_ang.append(
                np.array(
                    [np.sin(t1) * np.cos(t2),
                     np.sin(t1) * np.sin(t2),
                     np.cos(t1)]))

        [gt, route] = results.gt_and_route()
        gt_ang = np.stack(gt_ang)
        rec_ang = np.stack(rec_ang)
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez(os.path.join(path, name), gt=gt * 1e2, result=route *
                 1e2, gt_ang=gt_ang, result_ang=rec_ang)

    def compare_layout_thread(self, index, pSensori):

        overall_noise = np.random.randn(3)
        simu = self.estimate_B(pSensori)

        model = Solver_jac(1, self.route[0, 0], self.route[0, 1],
                           self.route[0, 2])
        model.fit_params['m0'].value = np.log(self.M)
        model.fit_params['m0'].vary = False

        results = Result_Handler(simu, 1)
        for i in range(simu.result.shape[0]):
            datai = simu.result[i].reshape(-1, 3)
            result = model.solve(datai, pSensori,
                                 not model.fit_params['m0'].vary)
            results += result
        [tmp, dist, loss] = results.cal_loss()
        return [index, dist, loss]

    def compare_layouts(self, pSensors, loop=1):
        results = []
        pool = Pool()
        for index, pSensor in enumerate(pSensors):
            # self.calculate_process(scale)
            # test(self, scale)
            for i in range(loop):
                # self.calculate_process(scale)
                # self.compare_layout_thread(index, pSensor)
                results.append(
                    pool.apply_async(self.compare_layout_thread,
                                     args=(index, pSensor)))
        pool.close()
        pool.join()

        # print('debug')

        loss_dict = {}
        dist_dict = {}
        for result in results:
            [scale, dist, loss] = result.get()
            if not str(scale) in loss_dict.keys():
                loss_dict[str(scale)] = loss
                dist_dict[str(scale)] = dist
            else:
                loss_dict[str(scale)] += loss

        # msg = ['Plane Layout(MIT)', 'Our Current Layout', 'Cube Layout']
        msg = ['Best Layout', 'Current Layout']
        for key in dist_dict.keys():
            plt.plot(dist_dict[key],
                     loss_dict[key] / loop,
                     label=msg[int(key)])
            plt.legend()
        plt.ylabel('Error(cm)')
        plt.xlabel('Distance(cm)')
        name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # plt.savefig('result/compare_layout/{}.jpg'.format(name), dpi=900)
        plt.show()

    def estimate_B(
            self,
            pSensor,
            route=None,
            noise_type=0,
            overall_noise=None):
        # noise type: 0: noise+precision, 1:only noise, 2: only precision
        # 3:none
        result = []
        exp = expression()
        if route is None:
            route = self.route
        for i in range(route.shape[0]):
            routei = route[i]
            tmp = []
            for j in range(pSensor.shape[0]):
                param = [
                    self.params['gx'], self.params['gy'], self.params['gz'],
                    pSensor[j][0], pSensor[j][1], pSensor[j][2], routei[0],
                    routei[1], routei[2], self.params['m'],
                    self.params['theta'], self.params['phy']
                ]

                tmp.append(exp.VecB(*param).squeeze())
            tmp = np.concatenate(tmp, axis=0).reshape(-1)
            result.append(tmp)

        result = np.concatenate(result, axis=0).reshape(-1, 3)
        Noise_x = 0.8 * np.random.randn(result.shape[0])
        Noise_y = 0.8 * np.random.randn(result.shape[0])
        Noise_z = 1.2 * np.random.randn(result.shape[0])
        Noise = np.stack([Noise_x, Noise_y, Noise_z]).T

        if noise_type != 3:
            if noise_type != 2:
                result += Noise
            if overall_noise is not None:
                result += overall_noise
            # add sensor resolution
            if noise_type != 1:
                result = np.floor(result * 100.0)
                result = result - np.mod(result, 15)
                result = 1e-2 * result

        # compute SNR
        G = 1e6 * np.array(
            [self.params['gx'], self.params['gy'], self.params['gz']])
        signal_power = np.sum(np.power(result - Noise, 2), 1)
        noise_power = np.sum(np.power(G + Noise, 2), 1)
        SNR = 10 * np.log(signal_power / noise_power)

        result = result.reshape(-1, pSensor.size)
        SNR = SNR.reshape(-1, pSensor.shape[0])
        # print('Debug')
        return Simu_Data(route, SNR, result)

    def estimate_B_even_noise(self, pSensor):
        result = []
        exp = expression()
        for i in range(self.route.shape[0]):
            routei = self.route[i]

            tmp = []
            for j in range(pSensor.shape[0]):
                param = [
                    self.params['gx'], self.params['gy'], self.params['gz'],
                    pSensor[j][0], pSensor[j][1], pSensor[j][2], routei[0],
                    routei[1], routei[2], self.params['m'],
                    self.params['theta'], self.params['phy']
                ]

                tmp.append(exp.VecB(*param).squeeze())
            tmp = np.concatenate(tmp, axis=0).reshape(-1)
            result.append(tmp)

        result = np.concatenate(result, axis=0).reshape(-1, 3)
        Noise_x = np.sqrt(2) / 2 * np.random.randn(result.shape[0])
        Noise_y = np.sqrt(2) / 2 * np.random.randn(result.shape[0])
        Noise_z = np.sqrt(2) / 2 * np.random.randn(result.shape[0])
        Noise = np.stack([Noise_x, Noise_y, Noise_z]).T
        result += Noise

        # add sensor resolution
        result = np.floor(result * 100.0)
        result = result - np.mod(result, 15)
        result = 1e-2 * result

        # compute SNR
        G = 1e6 * np.array(
            [self.params['gx'], self.params['gy'], self.params['gz']])
        signal_power = np.sum(np.power(result - Noise, 2), 1)
        noise_power = np.sum(np.power(G + Noise, 2), 1)
        SNR = 10 * np.log(signal_power / noise_power)

        result = result.reshape(-1, pSensor.size)
        SNR = SNR.reshape(-1, pSensor.shape[0])

        # print('Debug')
        return Simu_Data(self.route, SNR, result)

    def compare_method_thread(self, choice):
        pSensori = 5 * self.pSensor
        simu = self.estimate_B(pSensori)

        if choice == 0:
            model = Solver_jac(1, self.route[0, 0], self.route[0, 1],
                               self.route[0, 2])
            model.fit_params['m0'].value = np.log(self.M)
            model.fit_params['m0'].vary = False

            results = Result_Handler(simu, choice)
            for i in range(simu.result.shape[0]):
                datai = simu.result[i].reshape(-1, 3)
                result = model.solve(datai, pSensori,
                                     not model.fit_params['m0'].vary)
                results += result

        if choice == 1:
            sensor_count = pSensori.shape[0]
            my_filter = Magnet_UKF(
                1, pSensori, R_std=[0.8, 0.8, 1.5] * sensor_count)

            my_filter.lm_model.fit_params['m0'].value = np.log(self.M)
            my_filter.lm_model.fit_params['m0'].vary = False

            my_filter.lm_model.fit_params['X0'].value = self.route[0, 0]
            my_filter.lm_model.fit_params['Y0'].value = self.route[0, 1]
            my_filter.lm_model.fit_params['Z0'].value = self.route[0, 2]

            my_filter.ukf.x[0] = self.params['gx']
            my_filter.ukf.x[1] = self.params['gy']
            my_filter.ukf.x[2] = self.params['gz']

            my_filter.kf.x[0] = self.params['gx']
            my_filter.kf.x[1] = self.params['gy']
            my_filter.kf.x[2] = self.params['gz']

            my_filter.kf.x[3] = self.route[0, 0]
            my_filter.ukf.x[3] = self.route[0, 0]
            my_filter.kf.x[5] = self.route[0, 1]
            my_filter.ukf.x[5] = self.route[0, 1]
            my_filter.kf.x[7] = self.route[0, 2]
            my_filter.ukf.x[7] = self.route[0, 2]

            my_filter.kf.x[9] = self.params['theta']
            my_filter.ukf.x[9] = self.params['theta']
            my_filter.kf.x[11] = self.params['phy']
            my_filter.ukf.x[11] = self.params['phy']

            results = Result_Handler(simu, choice)
            for i in range(simu.result.shape[0]):
                my_filter.predict()

                datai = simu.result[i].reshape(-1)
                result = my_filter.update(datai)
                results += result

        if choice == 2:  # simple kf
            sensor_count = pSensori.shape[0]
            my_filter = Magnet_KF(1, pSensori, R_std=[
                                  0.8, 0.8, 1.5] * sensor_count)

            my_filter.lm_model.fit_params['m0'].value = np.log(self.M)
            my_filter.lm_model.fit_params['m0'].vary = False

            my_filter.lm_model.fit_params['X0'].value = self.route[0, 0]
            my_filter.lm_model.fit_params['Y0'].value = self.route[0, 1]
            my_filter.lm_model.fit_params['Z0'].value = self.route[0, 2]

            my_filter.kf.x[0] = self.params['gx']
            my_filter.kf.x[1] = self.params['gy']
            my_filter.kf.x[2] = self.params['gz']

            my_filter.kf.x[3] = self.route[0, 0]
            my_filter.kf.x[5] = self.route[0, 1]
            my_filter.kf.x[7] = self.route[0, 2]

            my_filter.kf.x[9] = self.params['theta']
            my_filter.kf.x[11] = self.params['phy']

            results = Result_Handler(simu, choice)
            for i in range(simu.result.shape[0]):
                my_filter.predict()

                datai = simu.result[i].reshape(-1, 3)
                result = my_filter.update(datai)
                results += result

        if choice == 3:  # simple kf
            sensor_count = pSensori.shape[0]
            my_filter = Magnet_KF(
                1, pSensori, R_std=[0.8, 0.8, 1.5] * sensor_count, ord=3)

            my_filter.lm_model.fit_params['m0'].value = np.log(self.M)
            my_filter.lm_model.fit_params['m0'].vary = False

            my_filter.lm_model.fit_params['X0'].value = self.route[0, 0]
            my_filter.lm_model.fit_params['Y0'].value = self.route[0, 1]
            my_filter.lm_model.fit_params['Z0'].value = self.route[0, 2]

            my_filter.kf.x[0] = self.params['gx']
            my_filter.kf.x[1] = self.params['gy']
            my_filter.kf.x[2] = self.params['gz']

            my_filter.kf.x[3] = self.route[0, 0]
            my_filter.kf.x[6] = self.route[0, 1]
            my_filter.kf.x[9] = self.route[0, 2]

            my_filter.kf.x[12] = self.params['theta']
            my_filter.kf.x[15] = self.params['phy']

            results = Result_Handler(simu, choice)
            for i in range(simu.result.shape[0]):
                my_filter.predict()

                datai = simu.result[i].reshape(-1, 3)
                result = my_filter.update(datai)
                results += result

        return results.cal_loss()

    def compare_method(self, loop):
        results = []
        pool = Pool()
        for i in range(loop):
            # self.compare_method_thread(1)
            results.append(
                pool.apply_async(self.compare_method_thread, args=(0, )))
            results.append(
                pool.apply_async(self.compare_method_thread, args=(2, )))
            # results.append(
            #     pool.apply_async(self.compare_method_thread, args=(2, )))
            # results.append(
            #     pool.apply_async(self.compare_method_thread, args=(3, )))
        pool.close()
        pool.join()

        # print('debug')

        loss_dict = {}
        dist_dict = {}
        for result in results:
            [scale, dist, loss] = result.get()
            if not str(scale) in loss_dict.keys():
                loss_dict[str(scale)] = loss
                dist_dict[str(scale)] = dist
            else:
                loss_dict[str(scale)] += loss

        msg = ['LM', 'MY UKF', "KF on LM results", "KF on LM results ord=3"]
        for key in dist_dict.keys():
            plt.plot(dist_dict[key],
                     loss_dict[key] / loop,
                     label=msg[int(key)])
            plt.legend()
        plt.ylabel('Error(cm)')
        plt.xlabel('Distance(cm)')

        name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.savefig('result/compare_method/{}.jpg'.format(name), dpi=600)

    def compare_softiron(self, loop):
        results = []
        pool = Pool()
        for i in range(loop):
            # self.compare_method_thread(1)
            results.append(
                pool.apply_async(self.compare_softiron_thread, args=(0, )))
            results.append(
                pool.apply_async(self.compare_softiron_thread, args=(1, )))
            # results.append(
            #     pool.apply_async(self.compare_method_thread, args=(2, )))
            # results.append(
            #     pool.apply_async(self.compare_method_thread, args=(3, )))
        pool.close()
        pool.join()

        # print('debug')

        loss_dict = {}
        dist_dict = {}
        for result in results:
            [scale, dist, loss] = result.get()
            if not str(scale) in loss_dict.keys():
                loss_dict[str(scale)] = loss
                dist_dict[str(scale)] = dist
            else:
                loss_dict[str(scale)] += loss

        msg = ['origin', 'Add softiron', ]
        for key in dist_dict.keys():
            plt.plot(dist_dict[key],
                     loss_dict[key] / loop,
                     label=msg[int(key)])
            plt.legend()
        plt.ylabel('Error(cm)')
        plt.xlabel('Distance(cm)')

        name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        root = 'result/compare_softiron'
        if not os.path.exists(root):
            os.makedirs(root)
        plt.savefig(os.path.join(root, '{}.jpg'.format(name)), dpi=600)

    def compare_softiron_thread(self, choice):
        pSensori = 5 * self.pSensor
        simu = self.estimate_B(pSensori)

        if choice == 0:
            init_param = np.array([0, 0, 0, np.log(
                self.M), self.route[0, 0], self.route[0, 1], self.route[0, 2], 0, 0])
            param = init_param.copy()

            results = Result_Handler(simu, choice)
            for i in range(simu.result.shape[0]):
                datai = simu.result[i].reshape(-1, 3)
                result = cs.solve_1mag(
                    datai.reshape(-1), pSensori.reshape(-1), param)
                param = result.copy()
                results += {'X0': param[4], 'Y0': param[5], 'Z0': param[6]}

        if choice == 1:
            init_param = np.array([0, 0, 0, np.log(
                self.M), self.route[0, 0], self.route[0, 1], self.route[0, 2], 0, 0])
            param = init_param.copy()

            results = Result_Handler(simu, choice)
            soft_iron_param = 0.05 * np.random.randn(
                simu.result.size//simu.result.shape[0])+1
            for i in range(simu.result.shape[0]):
                datai = simu.result[i].reshape(-1)
                datai *= soft_iron_param
                result = cs.solve_1mag(
                    datai.reshape(-1), pSensori.reshape(-1), param)
                param = result.copy()
                results += {'X0': param[4], 'Y0': param[5], 'Z0': param[6]}

        return results.cal_loss()

    def compare_hardiron(self, loop):
        results = []
        pool = Pool()
        for i in range(loop):
            # self.compare_method_thread(1)
            results.append(
                pool.apply_async(self.compare_hardiron_thread, args=(0, )))
            results.append(
                pool.apply_async(self.compare_hardiron_thread, args=(1, )))
            # results.append(
            #     pool.apply_async(self.compare_method_thread, args=(2, )))
            # results.append(
            #     pool.apply_async(self.compare_method_thread, args=(3, )))
        pool.close()
        pool.join()

        # print('debug')

        loss_dict = {}
        dist_dict = {}
        for result in results:
            [scale, dist, loss] = result.get()
            if not str(scale) in loss_dict.keys():
                loss_dict[str(scale)] = loss
                dist_dict[str(scale)] = dist
            else:
                loss_dict[str(scale)] += loss

        msg = ['origin', 'Add hardiron', ]
        for key in dist_dict.keys():
            plt.plot(dist_dict[key],
                     loss_dict[key] / loop,
                     label=msg[int(key)])
            plt.legend()
        plt.ylabel('Error(cm)')
        plt.xlabel('Distance(cm)')

        name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        root = 'result/compare_hardiron'
        if not os.path.exists(root):
            os.makedirs(root)
        plt.savefig(os.path.join(root, '{}.jpg'.format(name)), dpi=600)

    def compare_hardiron_thread(self, choice):
        pSensori = 5 * self.pSensor
        simu = self.estimate_B(pSensori, noise_type=0)

        if choice == 0:
            init_param = np.array([0, 0, 0, np.log(
                self.M), self.route[0, 0], self.route[0, 1], self.route[0, 2], 0, 0])
            param = init_param.copy()

            results = Result_Handler(simu, choice)
            for i in range(simu.result.shape[0]):
                datai = simu.result[i].reshape(-1, 3)
                result = cs.solve_1mag(
                    datai.reshape(-1), pSensori.reshape(-1), param)
                param = result.copy()
                results += {'X0': param[4], 'Y0': param[5], 'Z0': param[6]}

        if choice == 1:
            init_param = np.array([0, 0, 0, np.log(
                self.M), self.route[0, 0], self.route[0, 1], self.route[0, 2], 0, 0])
            param = init_param.copy()

            results = Result_Handler(simu, choice)
            soft_iron_param = 5.0 * np.random.randn(
                simu.result.size//simu.result.shape[0])+1
            for i in range(simu.result.shape[0]):
                datai = simu.result[i].reshape(-1)
                datai += soft_iron_param
                result = cs.solve_1mag(
                    datai.reshape(-1), pSensori.reshape(-1), param)
                param = result.copy()
                results += {'X0': param[4], 'Y0': param[5], 'Z0': param[6]}

        return results.cal_loss()

    def estimate_B_singular_noise(self, pSensor):
        result = []
        exp = expression()
        for i in range(self.route.shape[0]):
            routei = self.route[i]

            tmp = []
            for j in range(pSensor.shape[0]):
                param = [
                    self.params['gx'], self.params['gy'], self.params['gz'],
                    pSensor[j][0], pSensor[j][1], pSensor[j][2], routei[0],
                    routei[1], routei[2], self.params['m'],
                    self.params['theta'], self.params['phy']
                ]

                tmp.append(exp.VecB(*param).squeeze())
            tmp = np.concatenate(tmp, axis=0).reshape(-1)
            result.append(tmp)

        result = np.concatenate(result, axis=0).reshape(-1, 3)
        Noise_x = np.sqrt(1.5) * np.random.randn(result.shape[0])
        Noise_y = 0 * np.random.randn(result.shape[0])
        Noise_z = 0 * np.random.randn(result.shape[0])
        Noise = np.stack([Noise_x, Noise_y, Noise_z]).T
        result += Noise

        # add sensor resolution
        result = np.floor(result * 100.0)
        result = result - np.mod(result, 15)
        result = 1e-2 * result

        # compute SNR
        G = 1e6 * np.array(
            [self.params['gx'], self.params['gy'], self.params['gz']])
        signal_power = np.sum(np.power(result - Noise, 2), 1)
        noise_power = np.sum(np.power(G + Noise, 2), 1)
        SNR = 10 * np.log(signal_power / noise_power)

        result = result.reshape(-1, pSensor.size)
        SNR = SNR.reshape(-1, pSensor.shape[0])

        # print('Debug')
        return Simu_Data(self.route, SNR, result)


def simulate_2mag_3type_thread(pSensor, params, typ, i):
    tmp = []
    for j in range(pSensor.shape[0]):
        param = [
            params['gx'], params['gy'], params['gz'],
            pSensor[j][0], pSensor[j][1], pSensor[j][2], params['X0'],
            params['Y0'], params['Z0'], params['m'],
            params['theta0'], params['phy0'], params['X1'],
            params['Y1'], params['Z1'], params['m'],
            params['theta1'], params['phy1'],
        ]
        tmp.append(simulate_2mag_3type.exp.VecB(*param).squeeze())

    tmp = np.concatenate(tmp, axis=0)
    tmp = tmp.reshape(-1)
    print(i, ' finished ')
    return [tmp, typ]


def simulate_2mag_3type_delta_thread(pSensor, params, typ, i):
    tmp = []
    for j in range(pSensor.shape[0]):
        param = [
            params['gx'], params['gy'], params['gz'],
            pSensor[j][0], pSensor[j][1], pSensor[j][2], params['X0'],
            params['Y0'], params['Z0'], params['m'],
            params['theta0'], params['phy0'], params['X1'],
            params['Y1'], params['Z1'], params['m'],
            params['theta1'], params['phy1'],
        ]

        # the result after a short period of time
        r = 1 * 1e-2 * np.random.rand()
        theta = np.random.rand() * np.pi
        phy = np.random.rand() * 2 * np.pi
        dx0 = r * np.sin(theta) * np.cos(phy)
        dy0 = r * np.sin(theta) * np.sin(phy)
        dz0 = r * np.cos(theta)

        r = 1 * 1e-2 * np.random.rand()
        theta = np.random.rand() * np.pi
        phy = np.random.rand() * 2 * np.pi
        dx1 = r * np.sin(theta) * np.cos(phy)
        dy1 = r * np.sin(theta) * np.sin(phy)
        dz1 = r * np.cos(theta)

        param2 = [
            params['gx'], params['gy'], params['gz'],
            pSensor[j][0], pSensor[j][1], pSensor[j][2], params['X0'] + dx0,
            params['Y0'] + dy0, params['Z0'] + dz0, params['m'],
            params['theta0'], params['phy0'], params['X1'] + dx1,
            params['Y1'] + dy1, params['Z1'] + dz1, params['m'],
            params['theta1'], params['phy1'],
        ]
        aaa = np.concatenate(
            [simulate_2mag_3type.exp.VecB(*param).squeeze(),
             simulate_2mag_3type.exp.VecB(*param2).squeeze() -
             simulate_2mag_3type.exp.VecB(*param).squeeze()],
            axis=0)
        tmp.append(aaa)
        print(aaa.shape)

    tmp = np.concatenate(tmp, axis=0)
    tmp = tmp.reshape(-1)
    print(i, ' finished ')
    return [tmp, typ]


def simulate_2mag_3type(pSensor, size=1000, cls=3, edge=20):
    size = int(size)
    results = []
    types = []
    simulate_2mag_3type.exp = expression(2)
    pool = Pool()
    pool_results = []
    i = 0
    # for i in range(size * cls):
    while(i < size * cls):
        # G's Spherical Coordinates
        t1 = np.pi * np.random.rand()
        t2 = 2 * np.pi * np.random.rand()

        # P1's Spherical Coordinates
        tt1 = np.pi * np.random.rand()
        pp1 = 2 * np.pi * np.random.rand()

        # P2's Spherical Coordinates
        tt2 = np.pi * np.random.rand()
        pp2 = 2 * np.pi * np.random.rand()

        typ = i % cls
        G = 38.6600
        # G = 0.0
        if cls == 3:
            if typ == 0:
                r1 = np.random.rand() * 20 + edge
                r2 = np.random.rand() * 20 + edge
                params = {
                    'm': np.log(2.7),
                    'gx': G * np.sin(t1) * np.cos(t2) * 1e-6,
                    'gy': G * np.sin(t1) * np.sin(t2) * 1e-6,
                    'gz': G * np.cos(t1) * 1e-6,
                    'X0': 1e-2 * r1 * np.sin(tt1) * np.cos(pp1),
                    'Y0': 1e-2 * r1 * np.sin(tt1) * np.sin(pp1),
                    'Z0': 1e-2 * r1 * np.cos(tt1),
                    'theta0': np.pi * np.random.rand(),
                    'phy0': 2 * np.pi * np.random.rand(),
                    'X1': 1e-2 * r2 * np.sin(tt2) * np.cos(pp2),
                    'Y1': 1e-2 * r2 * np.sin(tt2) * np.sin(pp2),
                    'Z1': 1e-2 * r2 * np.cos(tt2),
                    'theta1': np.pi * np.random.rand(),
                    'phy1': 2 * np.pi * np.random.rand(),
                }
            elif typ == 1:
                r1 = np.random.rand() * 20 + edge
                r2 = np.random.rand() * (edge - 5) + 5
                params = {
                    'm': np.log(2.7),
                    'gx': G * np.sin(t1) * np.cos(t2) * 1e-6,
                    'gy': G * np.sin(t1) * np.sin(t2) * 1e-6,
                    'gz': G * np.cos(t1) * 1e-6,
                    'X0': 1e-2 * r1 * np.sin(tt1) * np.cos(pp1),
                    'Y0': 1e-2 * r1 * np.sin(tt1) * np.sin(pp1),
                    'Z0': 1e-2 * r1 * np.cos(tt1),
                    'theta0': np.pi * np.random.rand(),
                    'phy0': 2 * np.pi * np.random.rand(),
                    'X1': 1e-2 * r2 * np.sin(tt2) * np.cos(pp2),
                    'Y1': 1e-2 * r2 * np.sin(tt2) * np.sin(pp2),
                    'Z1': 1e-2 * r2 * np.cos(tt2),
                    'theta1': np.pi * np.random.rand(),
                    'phy1': 2 * np.pi * np.random.rand(),
                }
            elif typ == 2:
                r1 = np.random.rand() * (edge - 5) + 5
                r2 = np.random.rand() * (edge - 5) + 5
                params = {
                    'm': np.log(2.7),
                    'gx': G * np.sin(t1) * np.cos(t2) * 1e-6,
                    'gy': G * np.sin(t1) * np.sin(t2) * 1e-6,
                    'gz': G * np.cos(t1) * 1e-6,
                    'X0': 1e-2 * r1 * np.sin(tt1) * np.cos(pp1),
                    'Y0': 1e-2 * r1 * np.sin(tt1) * np.sin(pp1),
                    'Z0': 1e-2 * r1 * np.cos(tt1),
                    'theta0': np.pi * np.random.rand(),
                    'phy0': 2 * np.pi * np.random.rand(),
                    'X1': 1e-2 * r2 * np.sin(tt2) * np.cos(pp2),
                    'Y1': 1e-2 * r2 * np.sin(tt2) * np.sin(pp2),
                    'Z1': 1e-2 * r2 * np.cos(tt2),
                    'theta1': np.pi * np.random.rand(),
                    'phy1': 2 * np.pi * np.random.rand(),
                }
        elif cls == 2:
            if typ == 0:
                r1 = np.random.rand() * 20 + 30
                r2 = np.random.rand() * 20 + 10
                params = {
                    'm': np.log(2.7),
                    'gx': G * np.sin(t1) * np.cos(t2) * 1e-6,
                    'gy': G * np.sin(t1) * np.sin(t2) * 1e-6,
                    'gz': G * np.cos(t1) * 1e-6,
                    'X0': 1e-2 * r1 * np.sin(tt1) * np.cos(pp1),
                    'Y0': 1e-2 * r1 * np.sin(tt1) * np.sin(pp1),
                    'Z0': 1e-2 * r1 * np.cos(tt1),
                    'theta0': np.pi * np.random.rand(),
                    'phy0': 2 * np.pi * np.random.rand(),
                    'X1': 1e-2 * r2 * np.sin(tt2) * np.cos(pp2),
                    'Y1': 1e-2 * r2 * np.sin(tt2) * np.sin(pp2),
                    'Z1': 1e-2 * r2 * np.cos(tt2),
                    'theta1': np.pi * np.random.rand(),
                    'phy1': 2 * np.pi * np.random.rand(),
                }
            elif typ == 1:
                r1 = np.random.rand() * 20 + 10
                r2 = np.random.rand() * 20 + 10
                params = {
                    'm': np.log(2.7),
                    'gx': G * np.sin(t1) * np.cos(t2) * 1e-6,
                    'gy': G * np.sin(t1) * np.sin(t2) * 1e-6,
                    'gz': G * np.cos(t1) * 1e-6,
                    'X0': 1e-2 * r1 * np.sin(tt1) * np.cos(pp1),
                    'Y0': 1e-2 * r1 * np.sin(tt1) * np.sin(pp1),
                    'Z0': 1e-2 * r1 * np.cos(tt1),
                    'theta0': np.pi * np.random.rand(),
                    'phy0': 2 * np.pi * np.random.rand(),
                    'X1': 1e-2 * r2 * np.sin(tt2) * np.cos(pp2),
                    'Y1': 1e-2 * r2 * np.sin(tt2) * np.sin(pp2),
                    'Z1': 1e-2 * r2 * np.cos(tt2),
                    'theta1': np.pi * np.random.rand(),
                    'phy1': 2 * np.pi * np.random.rand(),
                }

        # check G and R
        # GG = np.linalg.norm(np.array([params['gx'],params['gy'],params['gz']]), ord=2)
        # print(GG)
        # check if two point are too close to each other
        dis = np.linalg.norm(
            np.array(
                [params['X0'] - params['X1'],
                 params['Y0'] - params['Y1'],
                 params['Z0'] - params['Z1']]),
            ord=2)
        # if dis < 5*1e-2:
        #     print(dis)
        #     continue

        i += 1
        # [tmp, typ] = simulate_2mag_3type_thread(pSensor, params, typ, i)
        pool_results.append(pool.apply_async(
            simulate_2mag_3type_thread, args=(pSensor, params, typ, i)))
    pool.close()
    pool.join()

    for pool_result in pool_results:
        [tmp, typ] = pool_result.get()
        results.append(tmp)
        types.append(typ)

    results = np.concatenate(results, axis=0).reshape(-1, 3)
    Noise_x = 0.7 * np.random.randn(results.shape[0])
    Noise_y = 0.7 * np.random.randn(results.shape[0])
    Noise_z = 1.2 * np.random.randn(results.shape[0])
    Noise = np.stack([Noise_x, Noise_y, Noise_z]).T

    # TODO: Desides whether to use the noise
    # results += Noise
    # results = np.floor(results * 100.0)
    # results = results - np.mod(results, 15)
    # results = 1e-2 * results

    # compute SNR
    G = 1e6 * np.array(
        [params['gx'], params['gy'], params['gz']])
    signal_power = np.sum(np.power(results - Noise, 2), 1)
    noise_power = np.sum(np.power(G + Noise, 2), 1)
    SNR = 10 * np.log(signal_power / noise_power)

    results = results.reshape(size * cls, -1)
    # np.save('result/3types.npy', result)
    types = np.array(types)
    return results, types


def simulate_2mag_3type_test(pSensor, size=1000, cls=3):
    size = int(size)
    results = []
    types = []
    simulate_2mag_3type.exp = expression(2)
    pool = Pool()
    pool_results = []
    for i in range(size * cls):
        # G's Spherical Coordinates
        t1 = np.pi * np.random.rand()
        t2 = 2 * np.pi * np.random.rand()

        # P1's Spherical Coordinates
        tt1 = np.pi * np.random.rand()
        pp1 = 2 * np.pi * np.random.rand()

        # P2's Spherical Coordinates
        tt2 = np.pi * np.random.rand()
        pp2 = 2 * np.pi * np.random.rand()

        typ = i % cls
        G = 38.6600
        if cls == 3:
            if typ == 0:
                r1 = np.random.rand() * 20 + 25
                r2 = np.random.rand() * 20 + 25
                params = {
                    'm': np.log(2.7),
                    'gx': G * np.sin(t1) * np.cos(t2) * 1e-6,
                    'gy': G * np.sin(t1) * np.sin(t2) * 1e-6,
                    'gz': G * np.cos(t1) * 1e-6,
                    'X0': 1e-2 * r1 * np.sin(tt1) * np.cos(pp1) + 0e-2,
                    'Y0': 1e-2 * r1 * np.sin(tt1) * np.sin(pp1),
                    'Z0': 1e-2 * r1 * np.cos(tt1),
                    'theta0': np.pi * np.random.rand(),
                    'phy0': 2 * np.pi * np.random.rand(),
                    'X1': 1e-2 * r2 * np.sin(tt2) * np.cos(pp2) + 0e-2,
                    'Y1': 1e-2 * r2 * np.sin(tt2) * np.sin(pp2),
                    'Z1': 1e-2 * r2 * np.cos(tt2),
                    'theta1': np.pi * np.random.rand(),
                    'phy1': 2 * np.pi * np.random.rand(),
                }
            elif typ == 1:
                r1 = np.random.rand() * 20 + 25
                r2 = np.random.rand() * 20 + 5
                params = {
                    'm': np.log(2.7),
                    'gx': G * np.sin(t1) * np.cos(t2) * 1e-6,
                    'gy': G * np.sin(t1) * np.sin(t2) * 1e-6,
                    'gz': G * np.cos(t1) * 1e-6,
                    'X0': 1e-2 * r1 * np.sin(tt1) * np.cos(pp1) + 0e-2,
                    'Y0': 1e-2 * r1 * np.sin(tt1) * np.sin(pp1),
                    'Z0': 1e-2 * r1 * np.cos(tt1),
                    'theta0': np.pi * np.random.rand(),
                    'phy0': 2 * np.pi * np.random.rand(),
                    'X1': 1e-2 * r2 * np.sin(tt2) * np.cos(pp2) + 0e-2,
                    'Y1': 1e-2 * r2 * np.sin(tt2) * np.sin(pp2),
                    'Z1': 1e-2 * r2 * np.cos(tt2),
                    'theta1': np.pi * np.random.rand(),
                    'phy1': 2 * np.pi * np.random.rand(),
                }
            elif typ == 2:
                r1 = np.random.rand() * 20 + 5
                r2 = np.random.rand() * 20 + 5
                params = {
                    'm': np.log(2.7),
                    'gx': G * np.sin(t1) * np.cos(t2) * 1e-6,
                    'gy': G * np.sin(t1) * np.sin(t2) * 1e-6,
                    'gz': G * np.cos(t1) * 1e-6,
                    'X0': 1e-2 * r1 * np.sin(tt1) * np.cos(pp1) + 0e-2,
                    'Y0': 1e-2 * r1 * np.sin(tt1) * np.sin(pp1),
                    'Z0': 1e-2 * r1 * np.cos(tt1),
                    'theta0': np.pi * np.random.rand(),
                    'phy0': 2 * np.pi * np.random.rand(),
                    'X1': 1e-2 * r2 * np.sin(tt2) * np.cos(pp2) + 0e-2,
                    'Y1': 1e-2 * r2 * np.sin(tt2) * np.sin(pp2),
                    'Z1': 1e-2 * r2 * np.cos(tt2),
                    'theta1': np.pi * np.random.rand(),
                    'phy1': 2 * np.pi * np.random.rand(),
                }
        if typ == 1:
            if np.abs(r1 - r2) < 5:
                print(np.abs(r1 - r2))
                typ = 2
            # [tmp, typ] = simulate_2mag_3type_thread(pSensor, params, typ, i)
        pool_results.append(pool.apply_async(
            simulate_2mag_3type_thread, args=(pSensor, params, typ, i)))
    pool.close()
    pool.join()

    for pool_result in pool_results:
        [tmp, typ] = pool_result.get()
        results.append(tmp)
        types.append(typ)

    results = np.concatenate(results, axis=0).reshape(-1, 3)
    Noise_x = 0.7 * np.random.randn(results.shape[0])
    Noise_y = 0.7 * np.random.randn(results.shape[0])
    Noise_z = 1.2 * np.random.randn(results.shape[0])
    Noise = np.stack([Noise_x, Noise_y, Noise_z]).T

    # TODO: Desides whether to use the noise
    # results += Noise
    # results = np.floor(results * 100.0)
    # results = results - np.mod(results, 15)
    # results = 1e-2 * results

    # compute SNR
    G = 1e6 * np.array(
        [params['gx'], params['gy'], params['gz']])
    signal_power = np.sum(np.power(results - Noise, 2), 1)
    noise_power = np.sum(np.power(G + Noise, 2), 1)
    SNR = 10 * np.log(signal_power / noise_power)

    results = results.reshape(size * cls, -1)
    # np.save('result/3types.npy', result)
    types = np.array(types)
    return results, types


def simulate_2mag_3type_box(pSensor, size=1000, cls=3):
    size = int(size)
    results = []
    types = []
    simulate_2mag_3type.exp = expression(2)
    pool = Pool()
    pool_results = []
    for i in range(size * cls):
        # G's Spherical Coordinates
        t1 = np.pi * np.random.rand()
        t2 = 2 * np.pi * np.random.rand()

        # P1's Spherical Coordinates
        tt1 = np.pi * np.random.rand()
        pp1 = 2 * np.pi * np.random.rand()

        # P2's Spherical Coordinates
        tt2 = np.pi * np.random.rand()
        pp2 = 2 * np.pi * np.random.rand()

        typ = i % cls
        G = 38.6600
        if cls == 3:
            if typ == 0:
                r1 = np.random.rand() * 20 + 25
                r2 = np.random.rand() * 20 + 25
                params = {
                    'm': np.log(2.7),
                    'gx': G * np.sin(t1) * np.cos(t2) * 1e-6,
                    'gy': G * np.sin(t1) * np.sin(t2) * 1e-6,
                    'gz': G * np.cos(t1) * 1e-6,
                    'X0': 1e-2 * (np.random.rand() * 20 + 25),
                    'Y0': 1e-2 * (np.random.rand() * 20 + 25),
                    'Z0': 1e-2 * (np.random.rand() * 20 + 25),
                    'theta0': np.pi * np.random.rand(),
                    'phy0': 2 * np.pi * np.random.rand(),
                    'X1': 1e-2 * (np.random.rand() * 20 + 25),
                    'Y1': 1e-2 * (np.random.rand() * 20 + 25),
                    'Z1': 1e-2 * (np.random.rand() * 20 + 25),
                    'theta1': np.pi * np.random.rand(),
                    'phy1': 2 * np.pi * np.random.rand(),
                }
            elif typ == 1:
                r1 = np.random.rand() * 20 + 25
                r2 = np.random.rand() * 20 + 5
                params = {
                    'm': np.log(2.7),
                    'gx': G * np.sin(t1) * np.cos(t2) * 1e-6,
                    'gy': G * np.sin(t1) * np.sin(t2) * 1e-6,
                    'gz': G * np.cos(t1) * 1e-6,
                    'X0': 1e-2 * (np.random.rand() * 20 + 25),
                    'Y0': 1e-2 * (np.random.rand() * 20 + 25),
                    'Z0': 1e-2 * (np.random.rand() * 20 + 25),
                    'theta0': np.pi * np.random.rand(),
                    'phy0': 2 * np.pi * np.random.rand(),
                    'X1': 1e-2 * (np.random.rand() * 20 + 5),
                    'Y1': 1e-2 * (np.random.rand() * 20 + 5),
                    'Z1': 1e-2 * (np.random.rand() * 20 + 5),
                    'theta1': np.pi * np.random.rand(),
                    'phy1': 2 * np.pi * np.random.rand(),
                }
            elif typ == 2:
                r1 = np.random.rand() * 20 + 5
                r2 = np.random.rand() * 20 + 5
                params = {
                    'm': np.log(2.7),
                    'gx': G * np.sin(t1) * np.cos(t2) * 1e-6,
                    'gy': G * np.sin(t1) * np.sin(t2) * 1e-6,
                    'gz': G * np.cos(t1) * 1e-6,
                    'X0': 1e-2 * (np.random.rand() * 20 + 5),
                    'Y0': 1e-2 * (np.random.rand() * 20 + 5),
                    'Z0': 1e-2 * (np.random.rand() * 20 + 5),
                    'theta0': np.pi * np.random.rand(),
                    'phy0': 2 * np.pi * np.random.rand(),
                    'X1': 1e-2 * (np.random.rand() * 20 + 5),
                    'Y1': 1e-2 * (np.random.rand() * 20 + 5),
                    'Z1': 1e-2 * (np.random.rand() * 20 + 5),
                    'theta1': np.pi * np.random.rand(),
                    'phy1': 2 * np.pi * np.random.rand(),
                }
        # if typ == 1:
        #     if np.abs(r1 - r2) < 20:
        #         print(np.abs(r1-r2))
        #         typ = 2
        # [tmp, typ] = simulate_2mag_3type_thread(pSensor, params, typ, i)
        pool_results.append(pool.apply_async(
            simulate_2mag_3type_thread, args=(pSensor, params, typ, i)))
    pool.close()
    pool.join()

    for pool_result in pool_results:
        [tmp, typ] = pool_result.get()
        results.append(tmp)
        types.append(typ)

    results = np.concatenate(results, axis=0).reshape(-1, 3)
    Noise_x = 0.7 * np.random.randn(results.shape[0])
    Noise_y = 0.7 * np.random.randn(results.shape[0])
    Noise_z = 1.2 * np.random.randn(results.shape[0])
    Noise = np.stack([Noise_x, Noise_y, Noise_z]).T

    # TODO: Desides whether to use the noise
    # results += Noise
    # results = np.floor(results * 100.0)
    # results = results - np.mod(results, 15)
    # results = 1e-2 * results

    # compute SNR
    G = 1e6 * np.array(
        [params['gx'], params['gy'], params['gz']])
    signal_power = np.sum(np.power(results - Noise, 2), 1)
    noise_power = np.sum(np.power(G + Noise, 2), 1)
    SNR = 10 * np.log(signal_power / noise_power)

    results = results.reshape(size * cls, -1)
    # np.save('result/3types.npy', result)
    types = np.array(types)
    return results, types


if __name__ == "__main__":

    # generate_route()
    # generate_route_2mag()

    pSensor = 1e-2 * np.array([
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, -1],
    ])
    pSensor = 1e-2 * np.array([[2.675, -5.3, 1.5], [-2.675, -5.3, 1.5],
                               [2.675, 0, 4.76], [-2.675, 0, 4.76],
                               [2.675, 5.3, 1.5], [-2.675, 5.3, 1.5]])

    pSensor1 = 5e-2 * np.array([
        [1, 1, 0],
        [1, 0, 0],
        [1, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [-1, 1, 0],
        [-1, 0, 0],
        [-1, -1, 0],
    ])
    pSensor2 = 1e-2 * np.array([[2.675, -5.3, 1.5], [-2.675, -5.3, 1.5],
                                [2.675, 0, 4.76], [-2.675, 0, 4.76],
                                [2.675, 5.3, 1.5], [-2.675, 5.3, 1.5]])
    pSensor_Cube = 5e-2 / np.sqrt(3) * np.array([
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, -1],
    ])
    pSensor_Cube = 1e-2 * np.array([
        [4.89, 4.89, -4.9],
        [4.89, -4.89, -4.9],
        [-4.89, 4.89, -4.9],
        [-4.89, -4.89, -4.9],
        [4.89, 4.89, 4.9],
        [4.89, -4.89, 4.9],
        [-4.89, 4.89, 4.9],
        [-4.89, -4.89, 4.9],
    ])
    pSensor_test = np.load('result/best_loc/2021-01-27 23:24_Final.npy')

    simulate_2mag_3type(pSensor_Cube, 10000)
    # pSensors = [pSensor_Cube, pSensor_test]
    # testmodel = Simu_Test(21, 40, [4, 5, 6, 7, 8], resolution=100)
    # testmodel.simulate(1)
    # testmodel.compare_3_noise(20)
    # testmodel.compare_layouts([pSensor_Cube], 10)
    # testmodel.compare_method(1)

    # visuliza three sensor layout
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_xlim([-25, 25])
    # ax.set_ylim([-25, 25])
    # # ax.set_zlim([-2, 8])
    # ax.set_title("Reconstructed Magnet Position")
    # ax.set_xlabel('x(cm)')
    # ax.set_ylabel('y(cm)')
    # ax.set_zlabel('z(cm)')
    # ax.scatter(1e2 * pSensors[0][:, 0], 1e2 * pSensors[0][:, 1],
    #            1e2 * pSensors[0][:, 2], s=1, alpha=0.5)
    # ax.scatter(1e2 * pSensors[1][:, 0], 1e2 * pSensors[1][:, 1],
    #            1e2 * pSensors[1][:, 2], s=1, alpha=0.5)
    # ax.scatter(1e2 * pSensors[2][:, 0], 1e2 * pSensors[2][:, 1],
    #            1e2 * pSensors[2][:, 2], s=1, alpha=0.5)
    # plt.show()
