import asyncio
import os
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
from collections import deque
from utils.read_files import find_latest_file_with_prefix_and_suffix
# Nordic NUS characteristic for RX, which should be writable`
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
# Nordic NUS characteristic for TX, which should be readable
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
num = 3
sensors = np.zeros((num, 3))
result = []
offset = np.zeros((num,3))
scale = np.ones((num,3))
threshold = 0.3
frame_delay = 16
env_mag = np.zeros((3))
near_mag = False
readings_queue = deque(maxlen=frame_delay)
alpha = 0.2
filtered_sensors = np.zeros((num))
filter_alpha = 0.4
# name = [
#     'Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5',
#     'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 9', 'Sensor 10'
# ]
name = [
    'Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3'
]
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2',
# 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6']


def distance(b_1, b_0, p=1):
    return np.sum(np.abs(b_0-b_1)**p)**(1/p)

@atexit.register
def clean():
    print("Output csv")
    test = pd.DataFrame(columns=name, data=result)
    gesture_name = "test"
    if not os.path.exists(f"datasets/{gesture_name}"):
        os.makedirs(f"datasets/{gesture_name}")
    test.to_csv(
        f"datasets/{gesture_name}/{gesture_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
    )
    print("Exited")


def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    global num
    global sensors
    global result
    global env_mag
    global readings_queue
    global filtered_sensors
    current = [datetime.datetime.now()]
    for i in range(num):
        sensors[i, 0] = struct.unpack('f', data[12 * i:12 * i + 4])[0]
        sensors[i, 1] = struct.unpack('f', data[12 * i + 4:12 * i + 8])[0]
        sensors[i, 2] = struct.unpack('f', data[12 * i + 8:12 * i + 12])[0]

        sensors[i,:] = (sensors[i, :] - offset[i])/scale[i]

        print(f"Sensor {i+1}: {sensors[i, 0]:.6f}, {sensors[i, 1]:.6f}, {sensors[i, 2]:.6f}")
        current.append("(" + str(sensors[i, 0]) + ", " + str(sensors[i, 1]) +
                       ", " + str(sensors[i, 2]) + ")")
    filtered_sensors = filter_alpha*sensors + (1-filter_alpha)*filtered_sensors
    if (distance(filtered_sensors[0,:], filtered_sensors[-1,:], 2)>threshold):
        print("YES")
        print(f"envmag is {env_mag}")
        near_mag = True
    else:
        print("NO")
        near_mag = False
    if not near_mag:
        env_mag = filtered_sensors.copy()
        readings_queue.append(env_mag)
        # env_mag = alpha*env_mag+(1-alpha)*readings_queue[0]
    for i in range(num):
        print(f"Filtered Sensor {i+1}: {filtered_sensors[i, 0]:.6f}, {filtered_sensors[i, 1]:.6f}, {filtered_sensors[i, 2]:.6f}")
    
    # battery_voltage = struct.unpack('f', data[12 * num: 12 * num + 4])[0]
    # print("Battery voltage: " + str(battery_voltage))
    print("############")
    result.append(current)


async def run(address, loop):
    async with BleakClient(address, loop=loop) as client:
        # wait for BLE client to be connected
        x = await client.is_connected()
        print("Connected: {0}".format(x))
        print("Press Enter to quit...")
        # wait for data to be sent from client
        await client.start_notify(UART_TX_UUID, notification_handler)
        while True:
            await asyncio.sleep(0.01)
        await client.stop_notify(UART_TX_UUID)


async def main():
    global address
    await asyncio.gather(
        asyncio.create_task(run(address, asyncio.get_event_loop())))


if __name__ == '__main__':
    # address = ("D4:6B:83:ab:C5:F2")  # circle board
    # address = ("C2:3C:D5:6E:35:0A")  # joint board 2
    address = "E8:71:7E:9D:FB:53" # 3 sensor board
    num = 3
    file_folder = "mytest"
    offset_path = os.path.join(file_folder,
        find_latest_file_with_prefix_and_suffix(file_folder,"offset-"))
    scale_path = os.path.join(file_folder,
        find_latest_file_with_prefix_and_suffix(file_folder,"scale-"))
    offset = np.load(offset_path)
    scale = np.load(scale_path)
    asyncio.run(main())
