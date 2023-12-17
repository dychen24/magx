from os import read
import queue
from codetiming import Timer
import asyncio
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from itertools import count
import time
from matplotlib.animation import FuncAnimation
from numpy.core.numeric import True_
import matplotlib
import queue
import asyncio
import struct
import sys
import time
import datetime
import atexit
import time
import numpy as np
from bleak import BleakClient
import matplotlib.pyplot as plt
from bleak import exc
import pandas as pd
import atexit
from multiprocessing import Pool
import multiprocessing
import keyboard

from src.solver import Solver_jac, Solver
from src.filter import Magnet_KF, Magnet_UKF
from src.preprocess import Calibrate_Data
from config import pSensor_smt, pSensor_large_smt, pSensor_small_smt, pSensor_median_smt, pSensor_imu
import cppsolver as cs

'''The parameter user should change accordingly'''
# Change pSensor if a different sensor layout is used
pSensor = pSensor_small_smt

# Change this parameter for different initial value for 1 magnet
params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6,
                   0, np.log(3), 1e-2 * (-2), 1e-2 * (2), 1e-2 * (11), 0, 0])
# Change this parameter for different initial value for 2 magnets
params2 = np.array([
    40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, np.log(3),
    1e-2 * 6, 1e-2 * 0, 1e-2 * (-1), 0, 0,
    1e-2 * 5, 1e-2 * (4), 1e-2 * (-1), 0, 0,
])

# Your adafruit nrd52832 ble address
ble_address = "2A59A2D4-BCD8-4AF7-B750-E51195C1CA13"
# Absolute or relative path to the calibration data, stored in CSV
cali_path = 'Path to the calibration data, stored in CSV'


'''The calculation and visualization process'''


matplotlib.use('Qt5Agg')


t = 0
# Nordic NUS characteristic for RX, which should be writable
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
# Nordic NUS characteristic for TX, which should be readable
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"


result = []

worklist = multiprocessing.Manager().Queue()

results = multiprocessing.Manager().Queue()
results2 = multiprocessing.Manager().Queue()


trigger_calibration = multiprocessing.Manager().Queue()


def end():
    print('End of the program')
    sys.exit(0)


def classification(pos):
    # pos = data[:3] + 15 * data[3:]
    names = ['left cheek', 'left eye', 'mouth',
             'nose', 'right cheek', 'right eye', 'no touch']

    # all boundaries are in the sensor's frame of reference
    # change according to user's face shape and sensor position
    left_boundary = -16  # y, the left most y coordinate of one's face
    right_boundary = 18  # y, the right most y coordinate of one's face
    front_boundary = 26  # z, the front most z coordinate of one's face
    low_boundary = 7   # x, the down most x coordinate of one's face

    # center coordinate of different part of the user's face
    centers = np.array([
        [5, -5, 14],        # left cheek
        [11, -1, 17],       # left eye
        [1.6, 2, 14],       # mouth
        [7, 2.4, 18.5],     # nose
        [8.8, 12.3, 12.4],  # right cheek
        [10.8, 6, 18.2],    # right eye
    ])

    if pos[2] > front_boundary or pos[1] < left_boundary or pos[1] > right_boundary or pos[0] < low_boundary:
        print(names[6])
        return
    dis = []
    for i in range(centers.shape[0]):
        dis.append(np.linalg.norm(pos - centers[i], ord=2))
    dis = np.stack(dis)
    pred_typ = np.argmin(dis)
    print(names[pred_typ])


def calculation_parallel(magcount=1, use_kf=0, use_wrist=False):
    global worklist
    global params
    global params2
    global results
    global results2
    global pSensor
    global trigger_calibration
    calibration = np.load('result/calibration.npz')
    offset = calibration['offset'].reshape(-1)
    scale = calibration['scale'].reshape(-1)
    local_trigger = 0
    calibration_offset = np.zeros_like(pSensor).reshape(-1)
    myparams1 = params
    myparams2 = params2
    while True:
        if not worklist.empty():
            raw_datai = worklist.get()
            if not trigger_calibration.empty():
                trigger_calibration.get()
                calibration_offset = raw_datai
                print('calibrated')
                continue

            datai = (raw_datai-calibration_offset)
            datai = datai.reshape(-1, 3)
            if magcount == 1:
                if np.max(np.abs(myparams1[4:7])) > 1:
                    myparams1 = params
                left_boundary = -16  # y
                right_boundary = 18
                front_boundary = 26  # z
                low_boundary = 7   # x

                resulti = cs.solve_1mag(
                    datai.reshape(-1), pSensor.reshape(-1), myparams1)
                myparams1 = resulti
                tmp = np.array([(resulti[4] + 0.175*np.sin(resulti[7])*np.cos(resulti[8])) * 1e2,
                                (resulti[5] + 0.175*np.sin(resulti[7])*np.sin(resulti[8])) * 1e2, (resulti[6] + 0.175*np.cos(resulti[7])) * 1e2])
                # print("Real Pos: ", tmp)
                classification(tmp)
                if tmp[2] > front_boundary or tmp[1] < left_boundary or tmp[1] > right_boundary or tmp[0] < low_boundary:
                    myparams1 = params
            elif magcount == 2:
                if np.max(
                        np.abs(myparams2[4: 7])) > 1 or np.max(
                        np.abs(myparams2[9: 12])) > 1:
                    myparams2 = params2

                resulti = cs.solve_2mag(
                    datai.reshape(-1), pSensor.reshape(-1), myparams2)
                myparams2 = resulti
            result = [resulti[4] * 1e2,
                      resulti[5] * 1e2, resulti[6] * 1e2]

            if magcount == 2:
                result2 = [resulti[9] * 1e2,
                           resulti[10] * 1e2, resulti[11] * 1e2]
                results2.put(result2)

            if magcount == 1:
                pass

            if magcount == 2:
                print(
                    "Mag 1 Position: {:.2f}, {:.2f}, {:.2f}, dis={:.2f} \n Mag 2 Position: {:.2f}, {:.2f}, {:.2f}, dis={:.2f}". format(
                        result[0],
                        result[1],
                        result[2],
                        np.sqrt(
                            result[0] ** 2 +
                            result[1] ** 2 +
                            result[2] ** 2),
                        result2[0],
                        result2[1],
                        result2[2],
                        np.sqrt(
                            result2[0] ** 2 +
                            result2[1] ** 2 +
                            result2[2] ** 2)))
                d1 = np.sqrt(result[0]**2 + result[1]**2 + result[2]**2)
                d2 = np.sqrt(result2[0]**2 + result2[1]**2 + result2[2]**2)
                if (d1 < 16 and d2 < 16):
                    print("Two mags")
                elif (d1 > 16 and d2 > 16):
                    print("Zero mag")
                else:
                    print("One mag")
            results.put(tmp)


async def show_mag(magcount=1):
    global t
    global pSensor
    global results
    global results2
    myresults = np.array([[0, 0, 10]])
    myresults2 = np.array([[0, 0, 10]])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    ax.view_init(70, -120)
    # TODO: add title
    ax.set_xlabel('x(cm)')
    ax.set_ylabel('y(cm)')
    ax.set_zlabel('z(cm)')
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-10, 40])
    Xs = 1e2 * pSensor[:, 0]
    Ys = 1e2 * pSensor[:, 1]
    Zs = 1e2 * pSensor[:, 2]

    XXs = Xs
    YYs = Ys
    ZZs = Zs
    ax.scatter(XXs, YYs, ZZs, c='r', s=1, alpha=0.5)

    (magnet_pos,) = ax.plot(t / 100.0 * 5, t / 100.0 * 5, t /
                            100.0 * 5, linewidth=3, animated=True)
    if magcount == 2:
        (magnet_pos2,) = ax.plot(t / 100.0 * 5, t / 100.0 * 5, t /
                                 100.0 * 5, linewidth=3, animated=True)
    plt.show(block=False)
    plt.pause(0.1)
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(magnet_pos)
    fig.canvas.blit(fig.bbox)
    # timer = Timer(text=f"frame elapsed time: {{: .5f}}")

    while True:
        # timer.start()
        fig.canvas.restore_region(bg)
        # update the artist, neither the canvas state nor the screen have
        # changed

        # update myresults
        if not results.empty():
            myresult = results.get()
            myresults = np.concatenate(
                [myresults, np.array(myresult).reshape(1, -1)])

        if myresults.shape[0] > 5:
            myresults = myresults[-5:]

        x = myresults[:, 0]
        y = myresults[:, 1]
        z = myresults[:, 2]

        xx = x
        yy = y
        zz = z

        magnet_pos.set_xdata(xx)
        magnet_pos.set_ydata(yy)
        magnet_pos.set_3d_properties(zz, zdir='z')
        # re-render the artist, updating the canvas state, but not the screen
        ax.draw_artist(magnet_pos)

        if magcount == 2:
            if not results2.empty():
                myresult2 = results2.get()
                myresults2 = np.concatenate(
                    [myresults2, np.array(myresult2).reshape(1, -1)])

            if myresults2.shape[0] > 30:
                myresults2 = myresults2[-30:]
            x = myresults2[:, 0]
            y = myresults2[:, 1]
            z = myresults2[:, 2]

            xx = x
            yy = y
            zz = z

            magnet_pos2.set_xdata(xx)
            magnet_pos2.set_ydata(yy)
            magnet_pos2.set_3d_properties(zz, zdir='z')
            ax.draw_artist(magnet_pos2)

        # copy the image to the GUI state, but screen might not changed yet
        fig.canvas.blit(fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        fig.canvas.flush_events()
        await asyncio.sleep(0)
        # timer.stop()


@ atexit.register
def clean():
    print("Output csv")
    # test = pd.DataFrame(columns=name, data=result)
    # test.to_csv("22.csv")
    print("Exited")


def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    num = int(pSensor.size/3)

    global worklist
    all_data = []
    sensors = np.zeros((num, 3))
    current = [datetime.datetime.now()]
    calibration = np.load('result/calibration.npz')
    offset = calibration['offset'].reshape(-1)
    scale = calibration['scale'].reshape(-1)
    for i in range(num):
        sensors[i, 0] = struct.unpack('f', data[12 * i: 12 * i + 4])[0]
        sensors[i, 1] = struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
        sensors[i, 2] = struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
        # print("Sensor " + str(i+1)+": "+str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2]))
        current.append(
            "(" + str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " +
            str(sensors[i, 2]) + ")")
        # battery_voltage = struct.unpack('f', data[12 * num: 12 * num + 4])[0]
        # print("Battery voltage: " + str(battery_voltage))
    sensors = sensors.reshape(-1)
    sensors = (sensors - offset) / scale * np.mean(scale)

    if len(all_data) > 3:
        sensors = (sensors + all_data[-1] + all_data[-2]) / 3
    all_data.append(sensors)
    worklist.put(sensors)
    # print("############")


async def run_ble(address, loop):
    async with BleakClient(address, loop=loop) as client:
        # wait for BLE client to be connected
        x = await client.is_connected()
        print("Connected: {0}".format(x))
        print("Press Enter to quit...")
        # wait for data to be sent from client
        await client.start_notify(UART_TX_UUID, notification_handler)
        while True:
            await asyncio.sleep(0.01)
            # data = await client.read_gatt_char(UART_TX_UUID)


async def main(magcount=1):
    """
    This is the main entry point for the program
    """
    # Address of the BLE device
    global ble_address
    address = (ble_address)

    # Run the tasks
    with Timer(text="\nTotal elapsed time: {:.1f}"):
        multiprocessing.Process(
            target=calculation_parallel, args=(magcount, 1, False)).start()
        await asyncio.gather(
            asyncio.create_task(run_ble(address, asyncio.get_event_loop())),
            asyncio.create_task(show_mag(magcount)),
        )


if __name__ == '__main__':
    if True:
        calibration = Calibrate_Data(cali_path)
        [offset, scale] = calibration.cali_result()
        np.savez('result/calibration.npz', offset=offset, scale=scale)
        print(np.mean(scale))
        # sys.exit(0)

    def trigger(e):
        print("You triggered the calibration")
        global trigger_calibration
        trigger_calibration.put(1)

    keyboard.on_press_key("r", trigger)
    asyncio.run(main(1))
