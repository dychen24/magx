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

sensors = np.zeros((9, 3))
result = []
name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
        'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']


@atexit.register
def clean():
    print("Output csv")
    test = pd.DataFrame(columns=name, data=result)
    test.to_csv("file_name.csv")
    print("Exited")


def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    num = 8
    global sensors
    global result
    current = [datetime.datetime.now()]
    for i in range(num):
        sensors[i, 0] = struct.unpack('f', data[12 * i: 12 * i + 4])[0]
        sensors[i, 1] = struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
        sensors[i, 2] = struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
        print("Sensor " + str(i+1)+": " +
              str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2]))
        current.append(
            "("+str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2])+")")
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

address = ("2A59A2D4-BCD8-4AF7-B750-E51195C1CA13")
loop = asyncio.get_event_loop()
loop.run_until_complete(run(address, loop))
