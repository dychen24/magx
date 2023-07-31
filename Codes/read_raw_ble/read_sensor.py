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
# Nordic NUS characteristic for RX, which should be writable`
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
# Nordic NUS characteristic for TX, which should be readable
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
num = 10
sensors = np.zeros((num + 1, 3))
result = []
name = [
    'Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5',
    'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 9', 'Sensor 10'
]
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2',
# 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6']


@atexit.register
def clean():
    print("Output csv")
    test = pd.DataFrame(columns=name, data=result)
    # test.to_csv("Code/read_raw_ble/sensor_reading_0607_red_1.csv")
    test.to_csv("10sensor.csv")
    print("Exited")


def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    global num
    global sensors
    global result
    current = [datetime.datetime.now()]
    # print(current)
    for i in range(num):
        sensors[i, 0] = struct.unpack('f', data[12 * i:12 * i + 4])[0]
        sensors[i, 1] = struct.unpack('f', data[12 * i + 4:12 * i + 8])[0]
        sensors[i, 2] = struct.unpack('f', data[12 * i + 8:12 * i + 12])[0]
        print("Sensor " + str(i + 1) + ": " + str(sensors[i, 0]) + ", " +
              str(sensors[i, 1]) + ", " + str(sensors[i, 2]))
        # print("Sensor " + str(i+1) + ": " + str(sensors[i, 2]))
        current.append("(" + str(sensors[i, 0]) + ", " + str(sensors[i, 1]) +
                       ", " + str(sensors[i, 2]) + ")")
    #battery_voltage = struct.unpack('f', data[12 * num: 12 * num + 4])[0]
    #print("Battery voltage: " + str(battery_voltage))
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
    address = ("C2:3C:D5:6E:35:0A")  # joint board 2
    asyncio.run(main())
