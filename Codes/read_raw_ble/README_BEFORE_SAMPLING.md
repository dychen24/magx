# Difference between this branch and the main branch

* Add a new [read_sensor_with_timer.py](./read_sensor_with_timer.py) that basically runs the [read_sensor_with_timer.py](./read_sensor.py) with a timer that is set to 3 seconds

* Modifiy [./read_sensor.py](./read_sensor.py) so that it's output file name is formatted with the gesture name given

# How to use
1. Change the working directory to top level working directory.
2. Create a folder named "datasets".
3. Create a folder named based on the gesture you are sampling like "chinpoke", "nosepinch" etc. under the "datasets" folder.
4. Modify the **gesture_name** variable in the "clean" function in read_sensor.py. Please note that the gesture name must be the same as the previously created folder name.
5. Run the [read_sensor_with_timer.py](read_sensor_with_timer.py)

# Gestures to be sampled
## Gesture sample guidelines
* The finger moves from the far side to a position close to the face when the gesture is captured, the corresponding gesture is executed only once, and the finger stays near the face at the end of the gesture.
* Collect at least 15 samples for each gesture.
## Static gestures
* Index finger touch the left cheek
* Index finger touch the left jaw
* Index finger touch the left eye
* Index finger touch the right cheek
* Index finger touch the right jaw
* Index finger touch the right eye
## Dynamic gestures
* A gentle tap on the cheek
* A light stroke of the fingertips along the jawline
* A quick pinch of the nose with the thumb and forefinger
* A slow, deliberate rubbing of the forehead with the palm of the hand
* A poke to the chin with the index finger
* A soothing caress of the temples with the fingertips
