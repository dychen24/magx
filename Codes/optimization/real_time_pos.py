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
import os
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

from src.solver import Solver_jac, Solver
from src.filter import Magnet_KF, Magnet_UKF
from src.preprocess import Calibrate_Data
from config import pSensor_smt, pSensor_large_smt, pSensor_small_smt, pSensor_median_smt, pSensor_imu
import cppsolver as cs

'''The parameter user should change accordingly'''
# Change pSensor if a different sensor layout is used
pSensor = pSensor_large_smt

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
t = 0
matplotlib.use('Qt5Agg')
# Nordic NUS characteristic for RX, which should be writable
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
# Nordic NUS characteristic for TX, which should be readable
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
result = []
worklist = multiprocessing.Manager().Queue()

results = multiprocessing.Manager().Queue()
results2 = multiprocessing.Manager().Queue()


def end():
    print('End of the program')
    sys.exit(0)


def calculation_parallel(magcount=1, use_kf=0, use_wrist=False):
    global worklist
    global params
    global params2
    global results
    global results2
    global pSensor

    myparams1 = params
    myparams2 = params2
    while True:
        if not worklist.empty():
            datai = worklist.get()
            datai = datai.reshape(-1, 3)
            # resulti [gx, gy, gz, m, x0,y0,z0, theta0, phy0, x1, y1, z1, theta1, phy1]
            if magcount == 1:
                if np.max(np.abs(myparams1[4:7])) > 1:
                    myparams1 = params
                resulti = cs.solve_1mag(
                    datai.reshape(-1), pSensor.reshape(-1), myparams1)
                myparams1 = resulti
                result = [resulti[4] * 1e2,
                          resulti[5] * 1e2, resulti[6] * 1e2]
                results.put(result)
                print("Position: {:.2f}, {:.2f}, {:.2f}, dis={:.2f}".format(
                    result[0],
                    result[1],
                    result[2],
                    np.sqrt(
                        result[0] ** 2 + result[1] ** 2 + result[2] ** 2)))
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
                results.put(result)
                result2 = [resulti[9] * 1e2,
                           resulti[10] * 1e2, resulti[11] * 1e2]
                results2.put(result2)
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


async def task(name, work_queue):
    timer = Timer(text=f"Task {name} elapsed time: {{: .1f}}")
    while not work_queue.empty():
        delay = await work_queue.get()
        print(f"Task {name} running")
        timer.start()
        await asyncio.sleep(delay)
        timer.stop()

async def show_mag(magcount=1):
    global t
    global pSensor
    global results
    global results2
    myresults = np.array([[0, 0, 10]])
    myresults2 = np.array([[0, 0, 10]])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')

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

        if myresults.shape[0] > 30:
            myresults = myresults[-30:]

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


def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    global pSensor
    global worklist
    num = int(pSensor.size/3)

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
        if not os.path.exists('result'):
            os.makedirs('result')
        np.savez('result/calibration.npz', offset=offset, scale=scale)
        print(np.mean(scale))

    asyncio.run(main(1))  # For tracking 1 magnet
    # asyncio.run(main(2)) # For tracking 2 magnet
