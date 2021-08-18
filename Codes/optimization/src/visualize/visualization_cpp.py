from os import read
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys
from tqdm import tqdm
import cppsolver as cs
from ..solver import Solver, Solver_jac
from ..preprocess import Reading_Data, LM_data, LM_data_2mag
from ..filter import lowpass_filter, mean_filter, median_filter, Magnet_KF, Magnet_UKF, Magnet_KF_cpp
from ..preprocess import read_data


def ang_convert(x):
    a = x//(2*np.pi)
    result = x-a*(2*np.pi)
    if result > np.pi:
        result -= np.pi * 2
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def show_track_1mag_csv_cpp(reading_path, cali_path, gt_path, pSensor, My_M, use_kalman=False):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim([-10, 15])
    ax.set_ylim([-10, 15])
    ax.set_zlim([0, 25])
    # ax.set_title("Reconstructed Magnet Position")
    ax.set_xlabel('x(cm)')
    ax.set_ylabel('y(cm)')
    ax.set_zlabel('z(cm)')
    # M_choice = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # M_choice = [0.8, 1, 1.2, 1.4]
    M_choice = [2]

    reading_data = Reading_Data(data_path=reading_path, cali_path=cali_path)
    data = reading_data.readings
    lm_data = LM_data(gt_path)
    # set the origin of the gt
    lm_data.offset = np.array([-1.5614192,  -0.31039926, 0.90800506])
    result_parameter = []

    color = ['r', 'b', 'g', 'y', 'm']
    for index, M in enumerate(M_choice):

        # model = Solver(1)
        # model = Finexus_Solver(-5e-2, -5e-2, 8e-2)
        pred_position = []
        changingM = []

        changingG = []
        changingTheta = []
        changingPhy = []
        directions = []
        SNR = []

        cut = 5

        starting_point = lm_data.get_gt(reading_data.tstamps[cut])[0]

        if use_kalman:
            kf_params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, np.log(
                My_M), 1e-2 * starting_point[0], 1e-2 * starting_point[1], 1e-2 * starting_point[2], 0, 0])
            model = Magnet_KF_cpp(
                1, pSensor, [0.8, 0.8, 1.5]*pSensor.shape[0], kf_params, dt=1/17, ord=3)

        else:
            params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, np.log(
                My_M), 1e-2 * starting_point[0], 1e-2 * starting_point[1], 1e-2 * starting_point[2], 0, 0])
            params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, np.log(
                My_M), 1e-2 * (-2), 1e-2 * (2), 1e-2 * (20), 0, 0])

        for i in tqdm(range(cut, data.shape[0] - cut)):
            # fix m value and gx gy gz

            datai = data[i].reshape(-1, 3)
            if use_kalman:
                model.predict()
                result = model.update(datai)
            else:
                result = cs.solve_1mag(
                    datai.reshape(-1), pSensor.reshape(-1), params)
                params = result.copy()

            [x0, y0, z0, Gx, Gy, Gz] = [
                result[4] * 1e2, result[5] * 1e2,
                result[6] * 1e2, result[0],
                result[1], result[2]
            ]
            # [m, theta, phy] = [np.exp(result['m0'].value), np.pi * sigmoid(
            #     result['theta0'].value), np.pi * np.tanh(result['phy0'].value)]
            [m, theta, phy, direction] = [
                np.exp(result[3]),
                ang_convert(result[7]),
                ang_convert(result[8]),
                np.array([np.sin(ang_convert(result[7]))*np.cos(ang_convert(result[8])),
                          np.sin(ang_convert(result[7]))*np.sin(ang_convert(result[8])), np.cos(ang_convert(result[7]))]),
            ]

            # [x, y, z, m] = [result['X'].value*1e2, result['Y'].value*1e2,
            #                 result['Z'].value*1e2, result['m'].value]
            G = np.array([Gx, Gy, Gz])
            noise = np.linalg.norm(G, 2)
            signal = np.linalg.norm(datai - G, 2)

            pred_position.append(x0)
            pred_position.append(y0)
            pred_position.append(z0)
            changingM.append(m)
            changingTheta.append(theta)
            changingPhy.append(phy)
            changingG.append([Gx, Gy, Gz])
            directions.append(direction)

        changingG = np.array(changingG)
        changingM = np.array(changingM)
        changingTheta = np.array(changingTheta)
        changingPhy = np.array(changingPhy)
        changingAng = np.stack([changingTheta, changingPhy], axis=0).T
        directions = np.stack(directions, axis=0)
        pred_position = np.array(pred_position).reshape(-1, 3)
        compare_label = [' ', '(fixing G)']
        ax.plot(pred_position[:, 0],
                pred_position[:, 1],
                pred_position[:, 2],
                c=color[index % len(color)],
                label='Magnet')
        print(np.mean(pred_position, axis=0))

    # sensor position
    ax.scatter(1e2 * pSensor[:, 0],
               1e2 * pSensor[:, 1],
               1e2 * pSensor[:, 2],
               c='r',
               s=1,
               alpha=0.5)

    # calculate loss
    gt_route = []
    losses = {}
    losses_count = {}

    gt_directions = []
    losses_angle = {}
    losses_count_angle = {}
    for i in range(pred_position.shape[0]):
        # Get gt
        gt = lm_data.get_gt(reading_data.tstamps[i + cut])
        gt_pos = gt[0]
        gt_route.append(gt_pos)

        gt_direction = gt[1]

        gt_directions.append(gt_direction)

        # calculate loss
        dis = np.linalg.norm(gt_pos - np.mean(pSensor, axis=0), 2)
        loss1 = np.linalg.norm(gt_pos - pred_position[i], 2)
        loss2 = np.arccos(np.dot(gt_direction, directions[i]))

        # store route loss
        if not dis in losses.keys():
            losses[dis] = loss1
            losses_count[dis] = 1
        else:
            losses[dis] += loss1
            losses_count[dis] += 1

        # store ang loss
        if not dis in losses_angle.keys():
            losses_angle[dis] = loss2
            losses_count_angle[dis] = 1
        else:
            losses_angle[dis] += loss2
            losses_count_angle[dis] += 1

    gt_route = np.stack(gt_route, axis=0)
    gt_directions = np.stack(gt_directions, axis=0)
    ax.plot(gt_route[:, 0],
            gt_route[:, 1],
            gt_route[:, 2],
            c='b',
            alpha=0.5,
            linewidth=2,
            label='Ground Truth')
    plt.legend()

    # store the gt route and the reconstructed route
    tmp = reading_path.split('/')
    file_name = tmp[-1].split('.')[0] + '.npz'
    tmp.pop(0)
    tmp.pop(-1)
    result_path = os.path.join('result', 'reconstruction_result', *tmp)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    np.savez(os.path.join(result_path, file_name),
             gt=gt_route,
             result=pred_position, gt_ang=gt_directions, result_ang=directions, G=changingG)

    fig5 = plt.figure()
    plt.title("Reconstuct Loss")
    plot_loss_data = []
    for dis in sorted(losses.keys()):
        plot_loss_data.append(dis)
        plot_loss_data.append(losses[dis] / losses_count[dis])
    plot_loss_data = np.array(plot_loss_data).reshape(-1, 2)
    plt.plot(plot_loss_data[:, 0],
             plot_loss_data[:, 1], label='Position loss')
    plt.legend()

    fig6 = plt.figure()
    plt.title("Reconstuct angle Loss")
    plot_loss_data = []
    for dis in sorted(losses_angle.keys()):
        plot_loss_data.append(dis)
        plot_loss_data.append(losses_angle[dis] / losses_count_angle[dis])
    plot_loss_data = np.array(plot_loss_data).reshape(-1, 2)
    plt.plot(plot_loss_data[:, 0], plot_loss_data[:, 1], label='Ang loss')
    plt.legend()

    fig2 = plt.figure()
    plt.title("Magnet Moment")
    # plt.ylim(0, 10)
    plt.plot(changingM, label='M')
    plt.legend()

    fig3 = plt.figure()
    plt.title("G")
    plt.plot(changingG[:, 0], label='Gx')
    plt.plot(changingG[:, 1], label='Gy')
    plt.plot(changingG[:, 2], label='Gz')
    plt.legend()

    fig4 = plt.figure()
    plt.title("orientation")
    plt.ylim(-5, 5)
    plt.plot(changingTheta, label='theta')
    plt.plot(changingPhy, label='phy')
    plt.legend()

    plt.show()
    # plt.savefig("result/result.jpg", dpi=900)


def show_track_2mag_csv_cpp(reading_path, cali_path, gt_path, pSensor, My_M, use_kalman=False):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    # ax.set_zlim([-2, 30])
    ax.set_title("Reconstructed Magnet Position")
    ax.set_xlabel('x(cm)')
    ax.set_ylabel('y(cm)')
    ax.set_zlabel('z(cm)')
    # M_choice = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # M_choice = [0.8, 1, 1.2, 1.4]
    M_choice = [2]

    reading_data = Reading_Data(data_path=reading_path, cali_path=cali_path)
    data = reading_data.readings
    lm_data = LM_data_2mag(gt_path)
    # set the origin of the gt
    lm_data.offset = np.array([-1.5614192,  -0.31039926, 0.90800506])
    result_parameter = []

    color = ['r', 'b', 'g', 'y', 'm']
    for index, M in enumerate(M_choice):
        pred_position = []
        changingM = []

        changingG = []
        changingTheta = []
        changingPhy = []
        changingTheta2 = []
        changingPhy2 = []
        changingDir = []
        changingDir2 = []
        SNR = []

        cut = 0

        starting_point = lm_data.get_gt(reading_data.tstamps[cut])
        params = {
            'X0': 1e-2 * starting_point[0][0],
            'Y0': 1e-2 * starting_point[0][1],
            'Z0': 1e-2 * starting_point[0][2],
            'm0': np.log(My_M),
            'theta0': 0.1,
            'phy0': 0.1,

            'X1': 1e-2 * starting_point[2][0],
            'Y1': 1e-2 * starting_point[2][1],
            'Z1': 1e-2 * starting_point[2][2],
            'm1': np.log(My_M),
            'theta1': 0.1,
            'phy1': 0.1,

            'gx': 0,
            'gy': 0,
            'gz': 0,
        }
        params = np.array([
            40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, np.log(My_M),
            1e-2 * starting_point[0][0], 1e-2 *
            starting_point[0][1], 1e-2 * starting_point[0][2], 0, 0,
            1e-2 * starting_point[2][0], 1e-2 *
            starting_point[2][1], 1e-2 * starting_point[2][2], 0, 0,
        ])

        params = np.array([
            40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, np.log(3),
            1e-2 * 11, 1e-2 * 1, 1e-2 * (-2), np.pi*0.5, np.pi*0.5,
            1e-2 * 5, 1e-2 * (7), 1e-2 * (-4), np.pi*0.5, np.pi*0.25,
        ])

        for i in tqdm(range(cut, data.shape[0] - cut)):
            # if i > 5:
            #     gt = lm_data.get_gt(reading_data.tstamps[i])
            #     params[4:7] = gt[0]*1e-2
            #     params[9:12] = gt[2]*1e-2
            datai = data[i].reshape(-1, 3)
            result = cs.solve_2mag(
                datai.reshape(-1), pSensor.reshape(-1), params)
            params = result.copy()

            result_parameter.append(result)
            # print('the m is ', result['m0'])

            [x0, y0, z0, x1, y1, z1, Gx, Gy, Gz] = [
                result[4] * 1e2, result[5] * 1e2, result[6] * 1e2, result[9] *
                1e2, result[10] * 1e2, result[11] *
                1e2, result[0],
                result[1], result[2]
            ]
            # [m, theta, phy] = [np.exp(result['m0'].value), np.pi * sigmoid(
            #     result['theta0'].value), np.pi * np.tanh(result['phy0'].value)]
            [m, theta1, phy1, theta2, phy2] = [
                np.exp(result[3]),
                ang_convert(result[7]),
                ang_convert(result[8]),
                ang_convert(result[12]),
                ang_convert(result[13]),
            ]

            # [x, y, z, m] = [result['X'].value*1e2, result['Y'].value*1e2,
            #                 result['Z'].value*1e2, result['m'].value]
            G = np.array([Gx, Gy, Gz])
            noise = np.linalg.norm(G, 2)
            signal = np.linalg.norm(datai - G, 2)

            pred_position.append(x0)
            pred_position.append(y0)
            pred_position.append(z0)
            pred_position.append(x1)
            pred_position.append(y1)
            pred_position.append(z1)
            changingM.append(m)
            changingTheta.append(theta1)
            changingPhy.append(phy1)
            changingDir.append(np.array([np.sin(theta1)*np.cos(phy1), np.sin(
                theta1)*np.sin(phy1), np.cos(theta1), np.sin(theta2)*np.cos(phy2), np.sin(
                theta2)*np.sin(phy2), np.cos(theta2)]))
            changingTheta2.append(theta2)
            changingPhy2.append(phy2)
            changingG.append([Gx, Gy, Gz])

        changingG = np.array(changingG)
        changingM = np.array(changingM)
        changingTheta = np.array(changingTheta)
        changingPhy = np.array(changingPhy)
        changingAng = np.stack([changingTheta, changingPhy], axis=0).T
        changingTheta2 = np.array(changingTheta2)
        changingPhy2 = np.array(changingPhy2)
        changingAng2 = np.stack([changingTheta2, changingPhy2], axis=0).T
        changingDir = np.stack(changingDir, axis=0)
        pred_position = np.array(pred_position).reshape(-1, 6)
        compare_label = [' ', '(fixing G)']
        ax.scatter(pred_position[:, 0],
                   pred_position[:, 1],
                   pred_position[:, 2],
                   s=1,
                   c=color[index % len(color)],
                   label='Magnet 1')
        ax.scatter(pred_position[:, 3],
                   pred_position[:, 4],
                   pred_position[:, 5],
                   s=1,
                   c=color[index % len(color)],
                   label='Magnet 2')

    # sensor position
    ax.scatter(1e2 * pSensor[:, 0],
               1e2 * pSensor[:, 1],
               1e2 * pSensor[:, 2],
               c='r',
               s=1,
               alpha=0.5)

    # calculate loss
    gt_route = []
    gt_angle = []
    losses1 = {}
    losses_count1 = {}
    losses1_ang = {}
    losses_count1_ang = {}

    losses2 = {}
    losses_count2 = {}
    losses2_ang = {}
    losses_count2_ang = {}
    for i in range(pred_position.shape[0]):
        # mag one
        gt = lm_data.get_gt(reading_data.tstamps[i + cut])
        if lm_data.idx == 1:
            gt_pos1 = gt[0]
            gt_ang1 = gt[1]

            gt_pos2 = gt[2]
            gt_ang2 = gt[3]

        else:
            gt_pos1 = gt[2]
            gt_ang1 = gt[3]

            gt_pos2 = gt[0]
            gt_ang2 = gt[1]

        dis = np.linalg.norm(gt_pos1 - np.mean(pSensor, axis=0), 2)
        loss = np.linalg.norm(gt_pos1 - pred_position[i][:3], 2)

        ang = np.array([np.sin(changingAng[i][0])*np.cos(changingAng[i][1]), np.sin(
            changingAng[i][0])*np.sin(changingAng[i][1]), np.cos(changingAng[i][0])])
        loss_angle = np.arccos(np.dot(gt_ang1, changingDir[i, :3]))

        if not dis in losses1.keys():
            losses1[dis] = loss
            losses_count1[dis] = 1
        else:
            losses1[dis] += loss
            losses_count1[dis] += 1
        if not dis in losses1_ang.keys():
            losses1_ang[dis] = loss_angle
            losses_count1_ang[dis] = 1
        else:
            losses1_ang[dis] += loss_angle
            losses_count1_ang[dis] += 1

        # mag two

        dis = np.linalg.norm(gt_pos2 - np.mean(pSensor, axis=0), 2)
        loss = np.linalg.norm(gt_pos2 - pred_position[i][3:], 2)

        ang2 = np.array([np.sin(changingAng2[i][0])*np.cos(changingAng2[i][1]), np.sin(
            changingAng2[i][0])*np.sin(changingAng2[i][1]), np.cos(changingAng2[i][0])])
        loss_angle = np.arccos(np.dot(gt_ang2, changingDir[i, 3:]))
        if not dis in losses2.keys():
            losses2[dis] = loss
            losses_count2[dis] = 1
        else:
            losses2[dis] += loss
            losses_count2[dis] += 1

        if not dis in losses2_ang.keys():
            losses2_ang[dis] = loss_angle
            losses_count2_ang[dis] = 1
        else:
            losses2_ang[dis] += loss_angle
            losses_count2_ang[dis] += 1

        gt_route.append(np.concatenate([gt_pos1, gt_pos2], axis=0))
        gt_angle.append(np.concatenate([gt_ang1, gt_ang2], axis=0))

    gt_route = np.stack(gt_route, axis=0).reshape(-1, 6)
    gt_angle = np.stack(gt_angle, axis=0)
    # ax.scatter(gt_route[:, 0],
    #            gt_route[:, 1],
    #            gt_route[:, 2],
    #            c='b',
    #            s=1,
    #            alpha=0.5)
    # ax.scatter(gt_route[:, 3],
    #            gt_route[:, 4],
    #            gt_route[:, 5],
    #            c='b',
    #            s=1,
    #            alpha=0.5)
    # plt.legend()

    # store the gt route and the reconstructed route
    tmp = reading_path.split('/')
    file_name = tmp[-1].split('.')[0] + '_2mag.npz'
    tmp.pop(0)
    tmp.pop(-1)
    result_path = os.path.join(
        'result', 'reconstruction_result', *tmp)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    np.savez(os.path.join(result_path, file_name),
             gt=gt_route,
             result=pred_position,
             gt_ang=gt_angle,
             result_ang=changingDir)

    fig5 = plt.figure()
    plt.title("Reconstuct Loss")
    plot_loss_data = []
    for dis in sorted(losses1.keys()):
        plot_loss_data.append(dis)
        plot_loss_data.append(losses1[dis] / losses_count1[dis])

    # TODO: Need some quantifiable result
    plot_loss_data = np.array(
        plot_loss_data).reshape(-1, 2)
    # cut the array so that the dis lies within 21cm
    # idx = np.where(plot_loss_data[:, 0] > 21)[0][0]
    # plot_loss_data = plot_loss_data[:idx]
    plt.plot(plot_loss_data[:, 0],
             plot_loss_data[:, 1], label='loss 1')

    plot_loss_data = []
    for dis in sorted(losses2.keys()):
        plot_loss_data.append(dis)
        plot_loss_data.append(losses2[dis] / losses_count2[dis])

    plot_loss_data = np.array(
        plot_loss_data).reshape(-1, 2)
    # cut the array so that the dis lies within 21cm
    # idx = np.where(plot_loss_data[:, 0] > 21)[0][0]
    # plot_loss_data = plot_loss_data[:idx]
    plt.plot(plot_loss_data[:, 0],
             plot_loss_data[:, 1], label='loss 2')

    plt.legend()

    fig6 = plt.figure()
    plt.title("Reconstuct Ang Loss")
    plot_loss_data = []
    for dis in sorted(losses1_ang.keys()):
        plot_loss_data.append(dis)
        plot_loss_data.append(losses1_ang[dis] / losses_count1_ang[dis])

    # TODO: Need some quantifiable result
    plot_loss_data = np.array(
        plot_loss_data).reshape(-1, 2)
    plt.plot(plot_loss_data[:, 0],
             plot_loss_data[:, 1], label='loss 1')

    plot_loss_data = []
    for dis in sorted(losses2_ang.keys()):
        plot_loss_data.append(dis)
        plot_loss_data.append(losses2_ang[dis] / losses_count2_ang[dis])

    plot_loss_data = np.array(
        plot_loss_data).reshape(-1, 2)
    plt.plot(plot_loss_data[:, 0],
             plot_loss_data[:, 1], label='loss 2')

    plt.legend()

    fig2 = plt.figure()
    plt.title("Magnet Moment")
    # plt.ylim(0, 10)
    plt.plot(changingM, label='M')
    plt.legend()

    fig3 = plt.figure()
    plt.title("G")
    plt.plot(changingG[:, 0], label='Gx')
    plt.plot(changingG[:, 1], label='Gy')
    plt.plot(changingG[:, 2], label='Gz')
    plt.legend()

    fig4 = plt.figure()
    plt.title("orientation")
    plt.ylim(-5, 5)
    plt.plot(changingTheta, label='theta')
    plt.plot(changingPhy, label='phy')
    plt.legend()

    plt.show()


if __name__ == "__main__":

    # plot_test_layout('result/best_loc/2021-01-23 12:34_Final.npy')
    # sys.exit(0)
    # track_path = "data/20_12.01/12:01 Brick/0_Not_Fixed.txt"
    # track_path = "data/21_1.15/01:15 Advanced GT/24.txt"
    # calibrate_path = "data/20_12.01/12:01 Brick/raw_calibrate.txt"
    # calibrate_path = "data/21_1.11/01:11 Moving platform/raw_calibrate 4.txt"
    # calibrate_path = "data/21_1.15/01:15 Advanced GT/calibrate_24.txt"
    # show_track_1mag(track_path, calibrate_path)
    # show_loss()

    # show_calibration(calibrate_path, calibrate_path)

    # large pcb high
    pSensor = 1e-2 * np.array([[2.675, -5.3, 1.5], [-2.675, -5.3, 1.5],
                               [2.675, 0, 4.76], [-2.675, 0, 4.76],
                               [2.675, 5.3, 1.5], [-2.675, 5.3, 1.5]])

    # 4 sensors
    pSensor4 = 1e-2 * np.array([
        [4.89, 4.89, 0],
        [4.89, -4.89, 0],
        [-4.89, 4.89, 0],
        [-4.89, -4.89, 0],
    ])

    # 5 sensors
    pSensor5 = 1e-2 * np.array([
        [4.89, 4.89, 0],
        [4.89, -4.89, 0],
        [-4.89, 4.89, 0],
        [-4.89, -4.89, 0],
        [0, 0, 0],
    ])

    # 6 sensors
    pSensor6 = 1e-2 * np.array([
        [4.89, 4.89, 0],
        [4.89, 0, 0],
        [4.89, -4.89, 0],
        [-4.89, 4.89, 0],
        [-4.89, 0, 0],
        [-4.89, -4.89, 0],
    ])

    # 7 sensors
    pSensor7 = 1e-2 * np.array([
        [4.89, 4.89, 0],
        [4.89, 0, 0],
        [4.89, -4.89, 0],
        [-4.89, 4.89, 0],
        [-4.89, 0, 0],
        [-4.89, -4.89, 0],
        [0, 0, 0],
    ])

    # 8 sensors
    pSensor8 = 1e-2 * np.array([
        [4.89, 4.89, 0],
        [4.89, 0, 0],
        [4.89, -4.89, 0],
        [0, 4.89, 0],
        [0, -4.89, 0],
        [-4.89, 4.89, 0],
        [-4.89, 0, 0],
        [-4.89, -4.89, 0],
    ])

    # 9 sensors
    pSensor9 = 1e-2 * np.array([
        [4.89, 4.89, 0],
        [4.89, 0, 0],
        [4.89, -4.89, 0],
        [0, 4.89, 0],
        [0, 0, 0],
        [0, -4.89, 0],
        [-4.89, 4.89, 0],
        [-4.89, 0, 0],
        [-4.89, -4.89, 0],
    ])

    # cube layout
    pSensor_cube = 1e-2 * np.array([
        [4.89, 4.89, -4.9],
        [4.89, -4.89, -4.9],
        [-4.89, 4.89, -4.9],
        [-4.89, -4.89, -4.9],
        [4.89, 4.89, 4.9],
        [4.89, -4.89, 4.9],
        [-4.89, 4.89, 4.9],
        [-4.89, -4.89, 4.9],
    ])

    pSensor_cube2 = 1e-2 * np.array([
        [4.89, 4.89, -1.6],
        [4.89, -4.89, -1.6],
        [-4.89, 4.89, -1.6],
        [-4.89, -4.89, -1.6],
        [4.89, 4.89, 1.6],
        [4.89, -4.89, 1.6],
        [-4.89, 4.89, 1.6],
        [-4.89, -4.89, 1.6],
    ])

    pSensor = 1e-2 * np.array([
        [1.27, -2.54, 1.6],
        [-1.27, -2.54, 1.6],
        [1.27, 0, 4.36],
        [-1.27,	0, 4.36],
        [1.27, 	2.54, 1.6],
        [1.27, 2.54, 1.6],
    ])

    # pSensors = [pSensor4, pSensor5, pSensor6, pSensor7, pSensor8, pSensor9]
    # root_path = 'data/21_1.22'
    # for path1 in ['near', 'far']:
    #     for path2 in range(4, 10):
    #         calibrate_path = os.path.join(root_path, path1, str(path2),
    #                                       'calibrate{}.csv'.format(path2))
    #         for idx in range(1, 4):
    #             Reading_path = os.path.join(root_path, path1, str(path2),
    #                                         '{}_{}.csv'.format(path2, idx))
    #             LM_path = os.path.join(root_path, path1, str(path2),
    #                                         '{}_{}_truth.csv'.format(path2, idx))
    #             print(Reading_path)

    LM_path = 'data/21_1.25/01:25 Large/2mag_1_truth.csv'
    Reading_path = 'data/21_1.25/01:25 Large/2mag_1.csv'
    calibrate_path = 'data/21_1.25/01:25 Large/calibrate.csv'
    show_track_2mag_csv(Reading_path, calibrate_path, LM_path, pSensor_cube)
