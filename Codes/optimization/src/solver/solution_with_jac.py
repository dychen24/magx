import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy
from lmfit import Parameters, minimize, report_fit, Minimizer


class Solver_jac:
    def __init__(self, mag_count, p1=-0.04, p2=0.04, p3=0.04):
        self.fit_params = Parameters()
        self.mag_count = mag_count
        self.fit_params.add('gx', value=0)
        self.fit_params.add('gy', value=0)
        self.fit_params.add('gz', value=0)
        for i in range(mag_count):
            self.fit_params.add('X{}'.format(i), value=p1)
            self.fit_params.add('Y{}'.format(i), value=p2)
            self.fit_params.add('Z{}'.format(i), value=p3)
            self.fit_params.add('m{}'.format(i), value=np.log(1.35), vary=True)
            self.fit_params.add('theta{}'.format(i), value=0.2)
            self.fit_params.add('phy{}'.format(i), value=0)
        self.build_symbol_function()

    def build_symbol_function(self):
        if self.mag_count == 1:
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
            VecM = 1e-7 * sp.exp(M) * sp.Matrix(
                [sp.sin(theta) * sp.cos(phy),
                 sp.sin(theta) * sp.sin(phy),
                 sp.cos(theta)])
            VecB = 3 * vecR * (VecM.T * vecR) / dis ** 5 - VecM / dis ** 3 + G
            VecB = 1e6 * VecB

            JacB = VecB.jacobian([gx, gy, gz, x, y, z, M, theta, phy])
            JacB_fixedM = VecB.jacobian([gx, gy, gz, x, y, z, theta, phy])
            # convert to function for faster evaluation
            self.lam_VecB = sp.lambdify(
                [gx, gy, gz, xs, ys, zs, x, y, z, M, theta, phy],
                VecB, 'numpy')
            self.lam_JacB = sp.lambdify(
                [gx, gy, gz, xs, ys, zs, x, y, z, M, theta, phy],
                JacB, 'numpy')
            self.lam_JacB_fixedM = sp.lambdify(
                [gx, gy, gz, xs, ys, zs, x, y, z, M, theta, phy],
                JacB_fixedM, 'numpy')

        elif self.mag_count == 2:
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
            for i in range(self.mag_count):
                vecR = sp.Matrix(
                    [xs - x[i], ys - y[i], zs - z[i]]).reshape(3, 1)
                # vecR = sp.Matrix([x, y, z]).reshape(3, 1)
                dis = sp.sqrt(vecR[0] ** 2 + vecR[1] ** 2 + vecR[2] ** 2)
                # VecM = M*sp.Matrix([sp.sin(theta2)*sp.cos(phy2),
                #                     sp.sin(theta2)*sp.sin(phy2), sp.cos(theta2)])
                VecMi = 1e-7 * sp.exp(M[i]) * sp.Matrix([sp.sin(theta[i]) * sp.cos(
                    phy[i]), sp.sin(theta[i]) * sp.sin(phy[i]), sp.cos(theta[i])])
                VecBi = 3 * vecR * (VecMi.T * vecR) / \
                    dis ** 5 - VecMi / dis ** 3
                VecB += VecBi

            VecB = 1e6 * VecB
            JacB = VecB.jacobian(
                [gx, gy, gz, x0, y0, z0, M0, theta0, phy0, x1, y1, z1, M1, theta1, phy1])
            JacB_fixedM = VecB.jacobian(
                [gx, gy, gz, x0, y0, z0, theta0, phy0, x1, y1, z1, theta1, phy1])

            # convert to function for faster evaluation
            self.lam_VecB = sp.lambdify(
                [gx, gy, gz, xs, ys, zs, x0, y0, z0, M0, theta0, phy0, x1, y1,
                 z1, M1, theta1, phy1],
                VecB, 'numpy')
            self.lam_JacB = sp.lambdify(
                [gx, gy, gz, xs, ys, zs, x0, y0, z0, M0, theta0, phy0, x1, y1,
                 z1, M1, theta1, phy1],
                JacB, 'numpy')
            self.lam_JacB_fixedM = sp.lambdify(
                [gx, gy, gz, xs, ys, zs, x0, y0, z0, M0, theta0, phy0, x1, y1,
                 z1, M1, theta1, phy1],
                JacB_fixedM, 'numpy')

    def calculate_B(self, p, data, pSensor):
        result = []
        for i in range(pSensor.shape[0]):
            if self.mag_count == 1:
                vecb = self.lam_VecB(
                    p['gx'],
                    p['gy'],
                    p['gz'],
                    pSensor[i, 0],
                    pSensor[i, 1],
                    pSensor[i, 2],
                    p['X0'],
                    p['Y0'],
                    p['Z0'],
                    p['m0'],
                    p['theta0'],
                    p['phy0'])
            elif self.mag_count == 2:
                vecb = self.lam_VecB(
                    p['gx'],
                    p['gy'],
                    p['gz'],
                    pSensor[i, 0],
                    pSensor[i, 1],
                    pSensor[i, 2],
                    p['X0'],
                    p['Y0'],
                    p['Z0'],
                    p['m0'],
                    p['theta0'],
                    p['phy0'],
                    p['X1'],
                    p['Y1'],
                    p['Z1'],
                    p['m1'],
                    p['theta1'],
                    p['phy1'])
            vecb = np.array(vecb)
            result.append(vecb.flatten())
        result = np.array(result)
        return result

    def calculate_jac(self, p, data, pSensor, fixedM):
        result = []
        for i in range(pSensor.shape[0]):
            if self.mag_count == 1:
                if not fixedM:
                    jac = self.lam_JacB(
                        p['gx'],
                        p['gy'],
                        p['gz'],
                        pSensor[i, 0],
                        pSensor[i, 1],
                        pSensor[i, 2],
                        p['X0'],
                        p['Y0'],
                        p['Z0'],
                        p['m0'],
                        p['theta0'],
                        p['phy0'])
                else:
                    jac = self.lam_JacB_fixedM(
                        p['gx'],
                        p['gy'],
                        p['gz'],
                        pSensor[i, 0],
                        pSensor[i, 1],
                        pSensor[i, 2],
                        p['X0'],
                        p['Y0'],
                        p['Z0'],
                        p['m0'],
                        p['theta0'],
                        p['phy0'])
            elif self.mag_count == 2:
                if not fixedM:
                    jac = self.lam_JacB(
                        p['gx'],
                        p['gy'],
                        p['gz'],
                        pSensor[i, 0],
                        pSensor[i, 1],
                        pSensor[i, 2],
                        p['X0'],
                        p['Y0'],
                        p['Z0'],
                        p['m0'],
                        p['theta0'],
                        p['phy0'],
                        p['X1'],
                        p['Y1'],
                        p['Z1'],
                        p['m1'],
                        p['theta1'],
                        p['phy1'])
                else:
                    jac = self.lam_JacB_fixedM(
                        p['gx'],
                        p['gy'],
                        p['gz'],
                        pSensor[i, 0],
                        pSensor[i, 1],
                        pSensor[i, 2],
                        p['X0'],
                        p['Y0'],
                        p['Z0'],
                        p['m0'],
                        p['theta0'],
                        p['phy0'],
                        p['X1'],
                        p['Y1'],
                        p['Z1'],
                        p['m1'],
                        p['theta1'],
                        p['phy1'])
            result.append(jac)
        result = np.concatenate(result, axis=0)
        return result

    def residual(self, p, data, pSensor, fixedM):
        predicted = self.calculate_B(p, data, pSensor)
        res = predicted - data
        return res.flatten()

    def solve(self, data, pSensor, fixedM):
        t0 = time.time()
        model = Minimizer(self.residual, self.fit_params,
                          fcn_args=(data, pSensor, fixedM))
        out = model.leastsq(Dfun=self.calculate_jac, col_deriv=0)
        # print(time.time()-t0)
        self.fit_params = out.params
        # report_fit(out)
        return out.params


if __name__ == '__main__':
    data = np.load('result/synthesized_route_data_2mag.npz')
    B_observed = data['result'][0].reshape(6, 3)
    route = data['position'][0]
    pSensor = 1e-2*np.array([
        [2.6, -5.5, 0],
        [-2.7,	-5.5, 0],
        [2.6, 0,	1.9],
        [-2.6, 0, 1.9],
        [2.6, 5.4, 0],
        [-2.7, 5.4, 0],
    ])
    pSensor = 1e-2 * np.array([
        [2.675, -5.3, 1.5],
        [-2.675, -5.3, 1.5],
        [2.675, 0, 4.76],
        [-2.675, 0, 4.76],
        [2.675, 5.3, 1.5],
        [-2.675, 5.3, 1.5]
    ])
    solver = Solver_jac(2)
    solver = solver
    solver.fit_params['X0'].value = -4e-2
    solver.fit_params['Y0'].value = -4e-2
    solver.fit_params['Z0'].value = 5e-2
    solver.fit_params['m0'].value = 1
    solver.fit_params['X1'].value = 1e-2
    solver.fit_params['Y1'].value = -4e-2
    solver.fit_params['Z1'].value = 4e-2
    solver.fit_params['m1'].value = 1
    solver.fit_params['m0'].vary = False
    solver.fit_params['m1'].vary = False

    tmp = solver.solve(B_observed, pSensor, True)
