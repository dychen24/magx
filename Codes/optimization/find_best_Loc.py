from os import TMP_MAX
import numpy as np
from numpy.core.fromnumeric import size
from multiprocessing import Pool
import datetime
from tqdm import tqdm
from filterpy.kalman import MerweScaledSigmaPoints, unscented_transform
import datetime
import os
from src.solver import Solver_jac
from src.simulation import expression, Simu_Data
import cppsolver as cs
import copy


class gen_simulation:
    def __init__(self):
        self.dises = np.linspace(10, 30, 2)
        self.route = 1e-2 * np.array([
            [10, 10, 20],
            [10, -10, 20],
            [-10, -10, 20],
            [-10, 10, 20],
            [15, 15, 30],
            [15, -15, 30],
            [-15, -15, 30],
            [-15, 15, 30],
        ])

        self.build_route()

        self.exp = expression()
        self.params = {
            'm0': np.log(1.85),
            'theta': 0,
            'phy': 0,
            'gx': 50 / np.sqrt(2) * 1e-6,
            'gy': 50 / np.sqrt(2) * 1e-6,
            'gz': 0,
        }

    def build_route(self):
        theta = 90 / 180.0 * np.pi
        route = np.linspace(10, 30, 20)
        route = np.stack([route * np.cos(theta), route * np.sin(theta)]).T
        # route_back = np.linspace(start, stop, 1000)[::-1]
        # route = np.concatenate(
        #     [route, route_back, route, route_back]).reshape(-1, 1)
        route = np.pad(route, ((0, 0), (1, 0)),
                       mode='constant', constant_values=0)
        self.route = 1e-2 * route

    def simulate(self, pSensor):
        result = []
        for i in range(self.route.shape[0]):
            routei = self.route[i]
            tmp = []
            for j in range(pSensor.shape[0]):
                param = [self.params['gx'], self.params['gy'], self.params['gz'],
                         pSensor[j][0], pSensor[j][1], pSensor[j][2], routei[0], routei[1], routei[2], self.params['m0'],
                         self.params['theta'], self.params['phy']]

                tmp.append(self.exp.VecB(*param).squeeze())
            tmp = np.concatenate(tmp, axis=0).reshape(-1)
            result.append(tmp)

        result = np.concatenate(result, axis=0).reshape(-1, 3)
        Noise_x = 0.7 * np.random.randn(result.shape[0])
        Noise_y = 0.7 * np.random.randn(result.shape[0])
        Noise_z = 1.2 * np.random.randn(result.shape[0])
        Noise = np.stack([Noise_x, Noise_y, Noise_z]).T

        result += Noise

        # add sensor resolution
        result = np.floor(result * 100.0)
        result = result - np.mod(result, 15)
        result = 1e-2 * result

        # compute SNR
        G = 1e6 * np.array([self.params['gx'],
                            self.params['gy'], self.params['gz']])
        signal_power = np.sum(np.power(result - Noise, 2), 1)
        noise_power = np.sum(np.power(G + Noise, 2), 1)
        SNR = 10 * np.log(signal_power / noise_power)

        result = result.reshape(-1, pSensor.size)
        SNR = SNR.reshape(-1, pSensor.shape[0])
        # print('Debug')
        return Simu_Data(self.route, SNR, result)

    def cal_ut_loss(self, pSensor):
        results = []
        points = MerweScaledSigmaPoints(
            pSensor.size, alpha=1e-3, beta=2., kappa=3-pSensor.size)
        for dis in self.dises:
            dis = dis/np.sqrt(2)
            result = []
            for theta in [0.25*np.pi, 0.75*np.pi]:
                for phi in [0.25*np.pi, 0.75*np.pi, 1.25*np.pi, 1.75*np.pi]:
                    params = np.array([50 / np.sqrt(2) * 1e-6, 50 / np.sqrt(2) * 1e-6, 0, np.log(
                        1.35), 1e-2 * dis * np.sin(theta)*np.cos(phi), 1e-2 * dis * np.sin(theta)*np.sin(phi), 1e-2 * dis * np.cos(theta), 0, 0])
                    # print(np.linalg.norm(np.array(params[4:7]), ord=2))
                    B = cs.calB(pSensor.reshape(-1), params.reshape(-1))
                    P = np.diag([0.6, 0.6, 1.2]*(B.size//3))

                    sigmas = points.sigma_points(B, P)
                    sigmas_f = []
                    for i, s in enumerate(sigmas):
                        cs_result = cs.solve_1mag(
                            s.reshape(-1), pSensor.reshape(-1), params)
                        tmp = cs_result[4:7]
                        sigmas_f.append(tmp)
                    sigmas_f = np.stack(sigmas_f, axis=0)
                    x, P = unscented_transform(
                        sigmas_f, points.Wm, points.Wc, residual_fn=np.subtract)
                    [w, v] = np.linalg.eig(P)
                    result.append(np.linalg.norm(w, ord=2))
            results.append(np.mean(np.array(result)))
        results = np.mean(np.stack(results))
        # print(results)
        return results

    def cal_loss(self, pSensor):
        simu = self.simulate(pSensor)

        loss = 0

        # + loss
        params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, np.log(
            1.85), self.route[0, 0], self.route[0, 1], self.route[0, 2], 0, 0])
        for i in range(simu.result.shape[0]):
            datai = simu.result[i].reshape(-1, 3)

            result = cs.solve_1mag(
                datai.reshape(-1), pSensor.reshape(-1), params)
            loss += np.linalg.norm(np.array(
                [result[4], result[5], result[6]])-self.route[i], 2)

        # - loss
        params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, np.log(
            1.85), -self.route[0, 0], -self.route[0, 1], -self.route[0, 2], 0, 0])
        for i in range(simu.result.shape[0]):
            datai = -(simu.result[i].reshape(-1, 3))

            result = cs.solve_1mag(
                datai.reshape(-1), pSensor.reshape(-1), params)
            loss += np.linalg.norm(np.array(
                [result[4], result[5], result[6]])-(-self.route[i]), 2)
        return loss/(simu.result.shape[0]*2)


class Point:
    # Used to store the info of a certain sensor
    def __init__(self, theta=None, phi=None):
        self.r = 5*1e-2
        self.fixed = True
        if theta is None:
            theta = (np.random.rand()) * np.pi  # [0, pi]
            self.fixed = False
        if phi is None:
            phi = 2.0*np.random.rand() * np.pi  # [0, 2pi]
            self.fixed = False
        self.P = np.array([theta, phi])
        self.v = np.random.randn(2)  # speed for theta and phi

        self.omega = 0.9

    def loc(self):
        return self.r*np.array([np.sin(self.P[0])*np.cos(self.P[1]), np.sin(self.P[0])*np.sin(self.P[1]), np.cos(self.P[0])])

    def update(self, local_best, global_best):
        if self.fixed:
            return
        # update V and location
        self.v = self.omega * self.v + np.random.rand() * Big_Group.c1 * (
            local_best.P - self.P) + np.random.rand() * Big_Group.c2 * (
            global_best.P - self.P)

        # update P
        self.P += self.v

    def __lt__(self, other):
        if self.P[0] != other.P[0]:
            return self.P[0] < other.P[0]
        else:
            return self.P[1] < other.P[1]

    def __gt__(self, other):
        if self.P[0] != other.P[0]:
            return self.P[0] > other.P[0]
        else:
            return self.P[1] > other.P[1]

    def __str__(self):
        return "({},{})".format(self.P[0], self.P[1])


class Point_Plane:
    # Used to store the info of a certain sensor on two circle plane
    def __init__(self):
        r = np.random.rand()
        theta = 2.0*np.random.rand()*np.pi
        z = 2*(np.random.rand()-0.5)
        self.P = np.array([r, theta, z])
        self.v = 0.1 * np.random.randn(3)  # speed for theta and phi

        self.omega = 0.9

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def loc(self):
        if (self.sigmoid(self.P[2]) > 0.5):
            z = 2
        else:
            z = -2
        return 1e-2 * np.array([10 * np.sqrt(2) * self.sigmoid(self.P[0])*np.cos(self.P[1]), 10 * np.sqrt(2) * self.sigmoid(self.P[0])*np.sin(self.P[1]), z])

    def update(self, local_best, global_best):
        # update V and location
        self.v = self.omega * self.v + np.random.rand() * Big_Group.c1 * (
            local_best.P - self.P) + np.random.rand() * Big_Group.c2 * (
            global_best.P - self.P)

        # update P
        self.P += self.v

    def __lt__(self, other):
        if self.P[2] != other.P[2]:
            return self.P[2] < other.P[2]
        elif self.P[0] != other.P[0]:
            return self.P[0] < other.P[0]
        elif self.P[1] != other.P[1]:
            return self.P[1] < other.P[1]

    def __gt__(self, other):
        if self.P[2] != other.P[2]:
            return self.P[2] < other.P[2]
        elif self.P[0] != other.P[0]:
            return self.P[0] < other.P[0]
        elif self.P[1] != other.P[1]:
            return self.P[1] < other.P[1]

    def __str__(self):
        return "({},{},{})".format(self.P[0], self.P[1], self.P[2])


class Small_Group():

    def __init__(self, nsensor=8, use_cube=None):
        self.best_P = []
        self.best_loss = np.inf

        self.Points = []
        if use_cube == 0:
            self.Points.append(Point(np.pi/180 * 45, np.pi/180 * 45))
            self.Points.append(Point(np.pi/180 * 45, np.pi/180 * 135))
            self.Points.append(Point(np.pi/180 * 45, np.pi/180 * 225))
            self.Points.append(Point(np.pi/180 * 45, np.pi/180 * 315))

            self.Points.append(Point(np.pi/180 * 135, np.pi/180 * 45))
            self.Points.append(Point(np.pi/180 * 135, np.pi/180 * 135))
            self.Points.append(Point(np.pi/180 * 135, np.pi/180 * 225))
            self.Points.append(Point(np.pi/180 * 135, np.pi/180 * 315))
        else:
            for i in range(nsensor):
                self.Points.append(Point_Plane())
        self.Points = sorted(self.Points)

    def loop(self, idx, test_pSensors=None):
        self.Points = sorted(self.Points)
        # update the particle position

        # if there is no best P, set it to the current
        if len(self.best_P) == 0:
            self.best_P = self.Points
            Big_Group.best_P = self.Points

        for i, P in enumerate(self.Points):
            P.update(self.best_P[i], Big_Group.best_P[i])
        self.Points = sorted(self.Points)

        # simulate B
        tmp = []
        result = []
        pSensor = []
        for p in self.Points:
            pSensor.append(p.loc())
        pSensor = np.array(pSensor)

        if not test_pSensors is None:
            pSensor = test_pSensors
        simu_model = gen_simulation()
        try:
            loss = simu_model.cal_ut_loss(pSensor)
        except ValueError:
            print("ValueError")
            loss = np.Inf

        # update local minimum
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_P = self.Points

        return [idx, copy.copy(self.Points), copy.copy(self.best_loss), copy.copy(self.best_P)]


class Big_Group():
    c1 = 1.4961*1
    c2 = 1.4961*1

    best_loss = np.inf
    best_P = []

    def __init__(self, loops=5000, nSmallGroup=500):
        self.particles = []
        self.loops = loops
        for i in range(nSmallGroup):
            self.particles.append(Small_Group())

    def run(self):
        start_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for iter in tqdm(range(self.loops)):
            pools = Pool()
            results = []
            for i in range(len(self.particles)):
                # tmp = self.particles[i].loop(i)
                # [idx, t1, t2, t3] = tmp
                # self.particles[idx].Points = t1
                # self.particles[idx].best_loss = t2
                # self.particles[idx].best_P = t3

                results.append(pools.apply_async(
                    self.particles[i].loop, args=(i,)))
            pools.close()
            pools.join()

            for result in results:
                tmp = result.get()
                [idx, t1, t2, t3] = tmp
                self.particles[idx].Points = t1
                self.particles[idx].best_loss = t2
                self.particles[idx].best_P = t3
                # update global minimum
                if t2 < Big_Group.best_loss:
                    print(t2)
                    Big_Group.best_loss = t2
                    Big_Group.best_P = t3

            if (iter % 1 == 0):
                best = []
                for P in Big_Group.best_P:
                    best.append(P.loc())
                best = np.array(best)
                print(best)
                global run_time
                if not os.path.exists('result/best_loc'):
                    os.makedirs('result/best_loc')
                np.save(
                    'result/best_loc/{}_Loop{}.npy'.format(run_time, iter+1), best)

        best = []
        for P in Big_Group.best_P:
            best.append(P.loc())
        best = np.array(best)
        return best


if __name__ == '__main__':

    run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    PSO = Big_Group(loops=1000, nSmallGroup=50)
    result = PSO.run()

    np.save('result/best_loc/{}_Final.npy'.format(run_time), result)
    print("debug")
