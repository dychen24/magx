# Leap Motion Setup Procedure

1. Download and install [Leap Motion SDK](https://developer.leapmotion.com/sdk-leap-motion-controller). We used Version 2.3.1 on macOS.
2. Put `LeapPython.so`, `libLeap.dylib` and `Leap.py` in `LeapSDK/lib` into this folder.
3. Plug the leap motion controller into your computer and run the Leap Motion Diagnostic Visualizer to check if the device is working properly.
4. Turn on the tool tracking mode and turn off the hand tracking mode of Leap Motion. On macOS, the settings are under `Tracking` of `Leap Motion Control Panel`.
5. Run `ground_truth.py` with Python 2.7. Change the output file name at the beginning of the file in advance. 
6. For more information regarding Leap Motion, please visit [this website](https://developer.leapmotion.com/desktop-leap-motion-controller/). During the first run of the code, you might encounter error `Fatal Python error: PyThreadState_Get: no current thread`. Please refer to this [page](https://stackoverflow.com/questions/42401156/leap-motion-error-fatal-python-error-pythreadstate-get-no-current-thread-abor) for a fix