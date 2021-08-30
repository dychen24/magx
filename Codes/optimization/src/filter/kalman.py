import numpy as np
from numpy.random import randn
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, unscented_transform
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import math
import sympy as sp
import cppsolver as cs
from lmfit import Parameters

from ..solver import Solver_jac, Solver


class My_UKF(UKF):
    def __init__(self, dim_x, dim_z, dt, hx, fx, points, sqrt_fn=None, x_mean_fn=None, z_mean_fn=None, residual_x=None, residual_z=None):
        super().__init__(dim_x, dim_z, dt, hx, fx, points, sqrt_fn=sqrt_fn, x_mean_fn=x_mean_fn,
                         z_mean_fn=z_mean_fn, residual_x=residual_x, residual_z=residual_z)

    # TODO: calculate the Reading according to both the estimate and the real
    def cal_Z(self, z, R=None, UT=None, hx=None, **hx_args):
        if z is None:
            self.z = np.array([[None]*self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self._dim_z) * R

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, **hx_args))

        sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, S = UT(sigmas_h, self.Wm, self.Wc,
                   R, self.z_mean, self.residual_z)
        SI = self.inv(S)

        K = np.dot(S - R, SI)        # Kalman gain
        y = self.residual_z(z, zp)   # residual

        Zpp = zp + np.dot(K, y)
        return Zpp


class Magnet_UKF:
    def __init__(self, mag_count, pSensor, R_std, dt=1/30, ord=2):
        self.ord = ord
        self.mag_count = mag_count
        self.pSensor = pSensor
        self.R_std = R_std
        self.dt = dt
        self.lm_model = Solver_jac(mag_count)
        self.lm_model.fit_params['m0'].value = 1
        self.lm_model.fit_params['m0'].vary = False

        self.__build_exp()
        self.__setup_UFK()
        self.__setup_UF()

    def __setup_UFK(self):
        points = MerweScaledSigmaPoints(
            3+(5*2)*self.mag_count, alpha=1e-3, beta=2., kappa=3-(3+(5*2)*self.mag_count))
        self.ukf = My_UKF(3+(5*2)*self.mag_count, len(self.R_std), self.dt,
                          fx=self.mag_Fx, hx=self.mag_Hx, points=points)

        # process noise
        # TODO: change the Q of the UKF
        self.ukf.Q[0:3, 0:3] = np.diag([1e-6 ** 2, 1e-6 ** 2, 1e-6 ** 2])
        for i in range(self.mag_count):
            # x y z
            self.ukf.Q[3 + 5*i: 3 + 5*i+2, 3 + 5*i: 3 + 5*i +
                       2] = Q_discrete_white_noise(2, dt=self.dt, var=5e-2 ** 2)
            self.ukf.Q[3 + 5*i+2:3 + 5*i+4, 3 + 5*i+2:3 + 5*i +
                       4] = Q_discrete_white_noise(2, dt=self.dt, var=5e-2 ** 2)
            self.ukf.Q[3 + 5*i+4: 3 + 5*i+6, 3 + 5*i+4: 3 + 5*i +
                       6] = Q_discrete_white_noise(2, dt=self.dt, var=5e-2 ** 2)
            # theta phy
            self.ukf.Q[3 + 5*i+6:3 + 5*i+8, 3 + 5*i+6:3 + 5*i +
                       8] = Q_discrete_white_noise(2, dt=self.dt, var=5e-2)
            self.ukf.Q[3 + 5*i+8:3 + 5*i+10, 3 + 5*i+8:3 + 5*i +
                       10] = Q_discrete_white_noise(2, dt=self.dt, var=5e-2)

        # measurement noise
        self.ukf.R = np.diag(self.R_std)
        self.ukf.R = self.ukf.R @ self.ukf.R  # square to get variance

        # initialization of state
        self.ukf.x = np.array(
            [0, 0, 0,  0e-2, 0, 5e-2, 0, 5e-2, 0, 0.2, 0.0, 0.0, 0.0, ])

        # initialization of state variance
        # TODO: change the state varience of the UKF
        tmp = [1e-2**2]*3
        for i in range(self.mag_count):
            tmp += [5e-1 ** 2]*3*2 + [5e-1 ** 2]*2*2
        self.ukf.P = np.diag(tmp)

    def __setup_UF(self):
        tracker = KalmanFilter(dim_x=3+(5*2)*self.mag_count,
                               dim_z=3+(5*1)*self.mag_count)

        # F matrix
        F = np.identity(3+5*2*self.mag_count, dtype=float)
        for i in range(5*self.mag_count):
            F[3 + 2*i, 3+2*i+1] = self.dt
        tracker.F = F

        # H matrix
        H = np.zeros([3+5*1*self.mag_count, 3+5*2*self.mag_count], dtype=float)
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        for i in range(self.mag_count*5):
            H[3+i, 3 + 2*i] = 1

        tracker.H = H

        tracker.R = np.eye(3+(5*1)*self.mag_count) * 2e-3 ** 2

        tracker.Q = self.ukf.Q.copy()
        tracker.x = self.ukf.x.copy()
        tracker.P = self.ukf.P.copy()

        self.kf = tracker

    def __build_exp(self):
        x, y, z, M, theta, phy, gx, gy, gz, xs, ys, zs = sp.symbols(
            'x, y, z, M, theta, phy, gx, gy, gz, xs, ys, zs', real=True)
        G = sp.Matrix([[gx], [gy], [gz]])
        # theta2 = sp.tanh(theta)
        # phy2 = sp.tanh(phy)
        vecR = sp.Matrix([xs - x, ys - y, zs - z]).reshape(3, 1)
        # vecR = sp.Matrix([x, y, z]).reshape(3, 1)
        dis = sp.sqrt(vecR[0] ** 2 + vecR[1] ** 2 + vecR[2] ** 2)
        # VecM = M*sp.Matrix([sp.sin(theta2)*sp.cos(phy2),
        #                     sp.sin(theta2)*sp.sin(phy2), sp.cos(theta2)])
        VecM = 1e-7 * sp.exp(M) * sp.Matrix([sp.sin(theta) * sp.cos(phy),
                                             sp.sin(theta) * sp.sin(phy), sp.cos(theta)])
        VecB = 3 * vecR * (VecM.T * vecR) / dis ** 5 - VecM / dis ** 3 + G
        VecB = 1e6 * VecB
        self.lam_VecB = sp.lambdify(
            [M, xs, ys, zs,  gx, gy, gz, x, y, z, theta, phy], VecB, 'numpy')

    def mag_Fx(self, x, dt):
        F = np.identity(3+5*2*self.mag_count, dtype=float)
        for i in range(5*self.mag_count):
            F[3 + 2*i, 3+2*i+1] = dt
        # F = np.array([[1, dt, 0, 0],
        #               [0,  1, 0, 0],
        #               [0,  0, 1, dt],
        #               [0,  0, 0, 1]], dtype=float)
        # result = F@x
        return F @ x

    def mag_Hx(self, x):
        est_reading = []
        mask = np.array(
            [1, 1, 1, *[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]*self.mag_count], dtype=np.bool)
        for i in range(self.pSensor.shape[0]):
            params = [np.log(1.85)] + \
                self.pSensor[i].tolist() + x[mask].tolist()
            est_reading.append(self.lam_VecB(*params))
        est_reading = np.concatenate(est_reading, axis=0).reshape(-1)
        return est_reading

    def predict(self):
        self.ukf.predict()
        self.kf.predict()

    def update(self, z):
        self.ukf.update(z.reshape(-1))
        result = Parameters()
        result.add('gx', self.ukf.x[0])
        result.add('gy', self.ukf.x[1])
        result.add('gz', self.ukf.x[2])
        result.add('m0', 1)
        for i in range(self.mag_count):
            result.add('X{}'.format(i), self.ukf.x[3+i*5*self.ord])
            result.add('Y{}'.format(i), self.ukf.x[3+self.ord+i*5*self.ord])
            result.add('Z{}'.format(i), self.ukf.x[3+2*self.ord+i*5*self.ord])
            result.add('theta{}'.format(
                i), self.ukf.x[3+3*self.ord+i*5*self.ord])
            result.add('phy{}'.format(
                i), self.ukf.x[3+4*self.ord+i*5*self.ord])
        return result

        est_B = self.cal_z(z).reshape(-1, 3)

        result = self.lm_model.solve(est_B, self.pSensor,
                                     not self.lm_model.fit_params['m0'].vary)

        zz = np.array([result['gx'].value, result['gy'].value, result['gz'].value, result['X0'].value,
                       result['Y0'].value, result['Z0'].value, result['theta0'].value, result['phy0'].value])
        # print(zz[3:6])
        self.kf.update(zz)
        self.ukf.x = self.kf.x.copy()
        self.ukf.P = self.kf.P.copy()

        return result

    def cal_z(self, z):
        return self.ukf.cal_Z(z)


class Magnet_KF:
    def __init__(self, mag_count, pSensor, R_std, dt=1/30, ord=2):
        self.mag_count = mag_count
        self.pSensor = pSensor
        self.R_std = R_std
        self.dt = dt
        self.ord = ord
        self.lm_model = Solver_jac(mag_count)
        self.lm_model.fit_params['m0'].value = 1
        self.lm_model.fit_params['m0'].vary = False

        self.__build_exp()
        self.__setup_KF()

    def __setup_KF(self):
        tracker = KalmanFilter(dim_x=3+(5*self.ord)*self.mag_count,
                               dim_z=3+(5*1)*self.mag_count)

        # F matrix
        F = np.identity(3+5*self.ord*self.mag_count, dtype=float)
        delta = [1, self.dt, 0.5 * self.dt * self.dt]
        for i in range(5*self.mag_count):
            for j in range(self.ord):
                # update r
                F[3 + self.ord*i, 3+self.ord*i+j] = delta[j]
            for j in range(1, self.ord):
                # update v
                F[3 + self.ord*i + 1, 3+self.ord*i+j] = delta[j-1]
        tracker.F = F

        # H matrix
        H = np.zeros([3+5*1*self.mag_count, 3+5 *
                      self.ord*self.mag_count], dtype=float)
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        for i in range(self.mag_count*5):
            H[3+i, 3 + self.ord*i] = 1

        tracker.H = H

        # measure noise
        # tracker.R = np.eye(3+(5*1)*self.mag_count) * 5e-3 ** 2
        tracker.R = np.diag([1e-5**2]*3 + 5*self.mag_count*[1e-2**2])

        # process noise
        # TODO: change the var of the process noise
        tracker.Q[0:3, 0:3] = np.diag([1e-5 ** 2, 1e-5 ** 2, 1e-5 ** 2])
        for i in range(self.mag_count):
            tracker.Q[3 + 5*self.ord*i: 3 + 5*self.ord*i + self.ord, 3 + 5*self.ord*i: 3 + 5*self.ord*i +
                      self.ord] = Q_discrete_white_noise(self.ord, dt=self.dt, var=1e-1 ** 2)
            tracker.Q[3 + 5*self.ord*i+self.ord:3 + 5*self.ord*i+2*self.ord, 3 + 5*self.ord*i+self.ord:3 + 5*self.ord*i +
                      2*self.ord] = Q_discrete_white_noise(self.ord, dt=self.dt, var=1e-1 ** 2)
            tracker.Q[3 + 5*self.ord*i+2*self.ord: 3 + 5*self.ord*i+3*self.ord, 3 + 5*self.ord*i+2*self.ord: 3 + 5*self.ord*i +
                      3*self.ord] = Q_discrete_white_noise(self.ord, dt=self.dt, var=1e-1 ** 2)
            # theta phy
            tracker.Q[3 + 5*self.ord*i+3*self.ord:3 + 5*self.ord*i+4*self.ord, 3 + 5*self.ord*i+3*self.ord:3 + 5*self.ord*i +
                      4*self.ord] = Q_discrete_white_noise(self.ord, dt=self.dt, var=5e-2)
            tracker.Q[3 + 5*self.ord*i+4*self.ord:3 + 5*self.ord*i+5*self.ord, 3 + 5*self.ord*i+4*self.ord:3 + 5*self.ord*i +
                      5*self.ord] = Q_discrete_white_noise(self.ord, dt=self.dt, var=5e-2)

        tracker.x = np.array(
            [0, 0, 0] + [0e-2, 0, 5e-2, 0, 5e-2, 0, 0.2, 0.0, 0.0, 0.0]*self.mag_count)
        if self.ord == 3:
            tracker.x = np.array(
                [0, 0, 0]+[0e-2, 0, 0, 5e-2, 0, 0, 5e-2, 0, 0, 0.2, 0, 0.0, 0.0, 0, 0.0]*self.mag_count)

        # TODO: change the var of the initial state noise
        tmp = [1e-2**2]*3
        for i in range(self.mag_count):
            tmp += [5e-2 ** 2]*3*self.ord + [5e-2 ** 2]*2*self.ord
        tracker.P = tmp

        self.kf = tracker

    def __build_exp(self):
        x, y, z, M, theta, phy, gx, gy, gz, xs, ys, zs = sp.symbols(
            'x, y, z, M, theta, phy, gx, gy, gz, xs, ys, zs', real=True)
        G = sp.Matrix([[gx], [gy], [gz]])
        # theta2 = sp.tanh(theta)
        # phy2 = sp.tanh(phy)
        vecR = sp.Matrix([xs - x, ys - y, zs - z]).reshape(3, 1)
        # vecR = sp.Matrix([x, y, z]).reshape(3, 1)
        dis = sp.sqrt(vecR[0] ** 2 + vecR[1] ** 2 + vecR[2] ** 2)
        # VecM = M*sp.Matrix([sp.sin(theta2)*sp.cos(phy2),
        #                     sp.sin(theta2)*sp.sin(phy2), sp.cos(theta2)])
        VecM = 1e-7 * sp.exp(M) * sp.Matrix([sp.sin(theta) * sp.cos(phy),
                                             sp.sin(theta) * sp.sin(phy), sp.cos(theta)])
        VecB = 1e6 * (3 * vecR * (VecM.T * vecR) /
                      dis ** 5 - VecM / dis ** 3) + G
        # VecB = 1e6 * VecB
        self.lam_VecB = sp.lambdify(
            [M, xs, ys, zs,  gx, gy, gz, x, y, z, theta, phy], VecB, 'numpy')

    def predict(self):
        self.kf.predict()

    def update(self, z):
        # est_B = self.cal_z(z).reshape(-1, 3)

        result = self.lm_model.solve(z, self.pSensor,
                                     not self.lm_model.fit_params['m0'].vary)

        zz = [result['gx'].value, result['gy'].value, result['gz'].value]
        for i in range(self.mag_count):
            zz += [result['X{}'.format(i)].value, result['Y{}'.format(i)].value, result['Z{}'.format(
                i)].value, result['theta{}'.format(i)].value, result['phy{}'.format(i)].value]
        # print(zz[3:6])
        assert (len(zz) == 3 + self.mag_count*5)
        self.kf.update(zz)

        # result = Parameters()
        # result.add('m0', self.lm_model.fit_params['m0'].value)
        # result.add('gx', self.kf.x[0])
        # result.add('gy', self.kf.x[1])
        # result.add('gz', self.kf.x[2])
        # result.add('X0', self.kf.x[3])
        # result.add('Y0', self.kf.x[3+self.ord])
        # result.add('Z0', self.kf.x[3+2*self.ord])
        # result.add('theta0', self.kf.x[3+3*self.ord])
        # result.add('phy0', self.kf.x[3+4*self.ord])

        # update the lm_model parameters
        # result['gx'].value = self.kf.x[0]
        # result['gy'].value = self.kf.x[1]
        # result['gz'].value = self.kf.x[2]
        for i in range(self.mag_count):
            result['X{}'.format(i)].value = self.kf.x[3+i*5*self.ord]
            result['Y{}'.format(i)].value = self.kf.x[3+self.ord+i*5*self.ord]
            result['Z{}'.format(i)].value = self.kf.x[3 +
                                                      2*self.ord+i*5*self.ord]
            result['theta{}'.format(i)].value = self.kf.x[3 +
                                                          3*self.ord+i*5*self.ord]
            result['phy{}'.format(i)].value = self.kf.x[3 +
                                                        4*self.ord+i*5*self.ord]
        self.lm_model.fit_params = result
        return result


class Magnet_KF_cpp:
    def __init__(self, mag_count, pSensor, R_std, params, M=3, dt=1/30, ord=2):
        self.mag_count = mag_count
        self.pSensor = pSensor
        self.R_std = R_std
        self.dt = dt
        self.ord = ord
        self.M = M

        self.__build_exp()
        self.__setup_KF()
        self.params = params
        self.kf.x[:3] = params[:3]
        self.kf.x[3::self.ord] = params[4:]

    def __setup_KF(self):
        tracker = KalmanFilter(dim_x=3+(5*self.ord)*self.mag_count,
                               dim_z=3+(5*1)*self.mag_count)

        # F matrix
        F = np.identity(3+5*self.ord*self.mag_count, dtype=float)
        delta = [1, self.dt, 0.5 * self.dt * self.dt]
        for i in range(5*self.mag_count):
            for j in range(self.ord):
                # update r
                F[3 + self.ord*i, 3+self.ord*i+j] = delta[j]
            for j in range(1, self.ord):
                # update v
                F[3 + self.ord*i + 1, 3+self.ord*i+j] = delta[j-1]
        tracker.F = F

        # H matrix
        H = np.zeros([3+5*1*self.mag_count, 3+5 *
                      self.ord*self.mag_count], dtype=float)
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        for i in range(self.mag_count*5):
            H[3+i, 3 + self.ord*i] = 1

        tracker.H = H

        # measure noise
        # tracker.R = np.eye(3+(5*1)*self.mag_count) * 5e-3 ** 2
        tracker.R = np.diag([1e-5**2]*3 + 5*self.mag_count*[1e-2**2])

        # process noise
        # TODO: change the var of the process noise
        tracker.Q[0:3, 0:3] = np.diag([1e-4 ** 2, 1e-4 ** 2, 1e-4 ** 2])
        for i in range(self.mag_count):
            tracker.Q[3 + 5*self.ord*i: 3 + 5*self.ord*i + self.ord, 3 + 5*self.ord*i: 3 + 5*self.ord*i +
                      self.ord] = Q_discrete_white_noise(self.ord, dt=self.dt, var=1e-2 ** 2)
            tracker.Q[3 + 5*self.ord*i+self.ord:3 + 5*self.ord*i+2*self.ord, 3 + 5*self.ord*i+self.ord:3 + 5*self.ord*i +
                      2*self.ord] = Q_discrete_white_noise(self.ord, dt=self.dt, var=1e-2 ** 2)
            tracker.Q[3 + 5*self.ord*i+2*self.ord: 3 + 5*self.ord*i+3*self.ord, 3 + 5*self.ord*i+2*self.ord: 3 + 5*self.ord*i +
                      3*self.ord] = Q_discrete_white_noise(self.ord, dt=self.dt, var=1e-2 ** 2)
            # theta phy
            tracker.Q[3 + 5*self.ord*i+3*self.ord:3 + 5*self.ord*i+4*self.ord, 3 + 5*self.ord*i+3*self.ord:3 + 5*self.ord*i +
                      4*self.ord] = Q_discrete_white_noise(self.ord, dt=self.dt, var=5e-2)
            tracker.Q[3 + 5*self.ord*i+4*self.ord:3 + 5*self.ord*i+5*self.ord, 3 + 5*self.ord*i+4*self.ord:3 + 5*self.ord*i +
                      5*self.ord] = Q_discrete_white_noise(self.ord, dt=self.dt, var=5e-2)

        tracker.x = np.array(
            [0, 0, 0] + [0e-2, 0, 5e-2, 0, 5e-2, 0, 0.2, 0.0, 0.0, 0.0]*self.mag_count)
        if self.ord == 3:
            tracker.x = np.array(
                [0, 0, 0]+[0e-2, 0, 0, 5e-2, 0, 0, 5e-2, 0, 0, 0.2, 0, 0.0, 0.0, 0, 0.0]*self.mag_count)

        # TODO: change the var of the initial state noise
        tmp = [1e-1**2]*3
        for i in range(self.mag_count):
            tmp += [5e-1 ** 2]*3*self.ord + [5e-1 ** 2]*2*self.ord
        tracker.P = tmp

        self.kf = tracker

    def __build_exp(self):
        x, y, z, M, theta, phy, gx, gy, gz, xs, ys, zs = sp.symbols(
            'x, y, z, M, theta, phy, gx, gy, gz, xs, ys, zs', real=True)
        G = sp.Matrix([[gx], [gy], [gz]])
        # theta2 = sp.tanh(theta)
        # phy2 = sp.tanh(phy)
        vecR = sp.Matrix([xs - x, ys - y, zs - z]).reshape(3, 1)
        # vecR = sp.Matrix([x, y, z]).reshape(3, 1)
        dis = sp.sqrt(vecR[0] ** 2 + vecR[1] ** 2 + vecR[2] ** 2)
        # VecM = M*sp.Matrix([sp.sin(theta2)*sp.cos(phy2),
        #                     sp.sin(theta2)*sp.sin(phy2), sp.cos(theta2)])
        VecM = 1e-7 * sp.exp(M) * sp.Matrix([sp.sin(theta) * sp.cos(phy),
                                             sp.sin(theta) * sp.sin(phy), sp.cos(theta)])
        VecB = 1e6 * (3 * vecR * (VecM.T * vecR) /
                      dis ** 5 - VecM / dis ** 3) + G
        # VecB = 1e6 * VecB
        self.lam_VecB = sp.lambdify(
            [M, xs, ys, zs,  gx, gy, gz, x, y, z, theta, phy], VecB, 'numpy')

    def predict(self):
        self.kf.predict()

    def update(self, z):
        # est_B = self.cal_z(z).reshape(-1, 3)

        if self.mag_count == 1:
            result = cs.solve_1mag(
                z.reshape(-1), self.pSensor.reshape(-1), self.params)

        elif self.mag_count == 2:
            result = cs.solve_2mag(
                z.reshape(-1), self.pSensor.reshape(-1), self.params)
        self.params = result.copy()
        zz = np.array(list(result[:3]) + list(result[4:]))
        # print(zz[3:6])
        assert (len(zz) == 3 + self.mag_count*5)
        self.kf.update(zz)

        # result = Parameters()
        # result.add('m0', self.lm_model.fit_params['m0'].value)
        # result.add('gx', self.kf.x[0])
        # result.add('gy', self.kf.x[1])
        # result.add('gz', self.kf.x[2])
        # result.add('X0', self.kf.x[3])
        # result.add('Y0', self.kf.x[3+self.ord])
        # result.add('Z0', self.kf.x[3+2*self.ord])
        # result.add('theta0', self.kf.x[3+3*self.ord])
        # result.add('phy0', self.kf.x[3+4*self.ord])

        # update the lm_model parameters
        result[:3] = self.kf.x[:3]
        result[4:] = self.kf.x[3::self.ord]
        # for i in range(self.mag_count):

        # self.lm_model.fit_params = result
        return result
