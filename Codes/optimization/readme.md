# The layout of the folder
- Visulization.py: used to track the magnet, visulize the result, and compare the result to the ground truth offline. The ground truth and the data should be stored as CSV (likely gathered by codes in `../leapmotion` and `../read_raw_ble`).
- real_time_pos.py: used to show the tracking result in real-time.
    - real_time_pos_belly.py: used for endocapsule tracking demo.
    - real_time_pos_car.py: used for driver hand monitoring demo.
    - real_time_pos_face.py: used for face touching demo.
    - real_time_pos_hand.py: used for hand tracking demo.
- simulation.py: generate simulated data and use it to validate the performance of the sensor layout.
- find_best_Loc.py: use PSO to find the best layout of the sensor array given the physical constraint.