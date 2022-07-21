# All codes for MagX

## Arduino
1. batteryTest: read battery voltage and level of an Adafruit Feather
2. bleReadMultiple: read readings from 8 magnetometers and send them via bluetooth
3. localReadMultiple: read readings from 8 magnetometers and print them in serial port
4. Library: modified library for Adafruit_MLX90393

## leapmotion
Code for gathering groundtruth using a Leap Motion Controller. The ground truth is stored in a csv file. Leap Motion 2.7 python SDK must be first installed

## read_raw_ble
Read measurements from an Adafruit Feather sent by bluetooth. Before reading the measurement using read_sensor.py, use find_device.py to find the address of the device, and change the address in read_sensor.py accordingly

## cpp_solver
C++ wrapped solver of the LM algorithm.

## optimization
All codes for finding the optimal layout and tracking magnets. For more detail please refer to the readme file inside the folder.

## requirements.txt
All python libraries used
