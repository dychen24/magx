"""
sensor reading for a given time span
"""
import asyncio
from read_sensor import run


async def timer(timeout):
    # Wait for the specified timeout
    await asyncio.sleep(timeout)
    # Raise a CancelledError to stop the program
    raise asyncio.CancelledError()


async def main():
    global ADDRESS
    ADDRESS = "C2:3C:D5:6E:35:0A"  # joint board 2
    time_window = 3
    try:
        await asyncio.gather(
            asyncio.create_task(run(ADDRESS, asyncio.get_event_loop())),
            timer(time_window))

    except asyncio.CancelledError:
        print("stopped")


if __name__ == "__main__":
    asyncio.run(main())
