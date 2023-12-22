"""
sensor reading for a given time span
"""
import asyncio
from read_sensor import run
import keyboard


async def timer(timeout):
    # Wait for the specified timeout
    await asyncio.sleep(timeout)
    # Raise a CancelledError to stop the program
    raise asyncio.CancelledError()


async def main():
    global ADDRESS
    ADDRESS = "C2:3C:D5:6E:35:0A"  # joint board 2
    time_window = 90
    # time_window = 2.5
    try:
        await asyncio.gather(
            asyncio.create_task(run(ADDRESS, asyncio.get_event_loop())),
            timer(time_window))

    except asyncio.CancelledError:
        print("stopped")


# global ADDRESS
# ADDRESS = "C2:3C:D5:6E:35:0A"  # joint board 2

# async def read_sensor_data(time_window):
#     try:
#         await asyncio.create_task(run(ADDRESS, asyncio.get_event_loop()))
#     except asyncio.CancelledError:
#         print("stopped")

# async def main():
#     time_window = 3
#     sensor_task = asyncio.create_task(read_sensor_data(time_window))

#     # 监听键盘输入，当检测到按键时取消任务
#     # keyboard.add_hotkey('mouse1', sensor_task.cancel)

#     try:
#         await sensor_task  # 等待任务结束
#     except asyncio.CancelledError:
#         pass  # 任务被取消

if __name__ == "__main__":
    asyncio.run(main())
