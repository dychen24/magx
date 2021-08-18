# Readme
This folder stores the cpp implementation of the LM algorithm. 
The source file locates in  *./src*

This library consist three functions:
- solve_1mag
- solve_2mag
- CalB

## Usage
### solve_1mag
- input:
  - readings: the sensor readings at time $t$
  - pSensor: the location of each sensor
  - init_param: the initialization parameters. It's a array of size 9. The meaning of each parameter goes in the following order: 
  $G_x, G_y , G_z, log(m), x, y, z, \theta, \phi$.**Noted that logrithm should be applied to magnet moment $m$**

- output:
  - result: It's a array of size 9, just like _init_param_ . This set of parameter fit the sensor reading the best.


### solve_2mag
- input:
  - readings: the sensor readings at time $t$
  - pSensor: the location of each sensor
  - init_param: the initialization parameters. It's a array of size 9. The meaning of each parameter goes in the following order: 
  $G_x, G_y , G_z, log(m), x_0, y_0, z_0, \theta_0, \phi_0, x_1, y_1, z_1, \theta_1, \phi_1$

- output:
  - result: It's a array of size 9, just like _init_param_ . This set of parameter fit the sensor reading the best. **Noted that in the function both magnet has the same magnet moment $m$**.

### CalB
- input:
  - pSensor: the location of each sensor
  - init_param: the initialization parameters. It's a array of size 9. The meaning of each parameter goes in the following order: 
  $G_x, G_y , G_z, m, x, y, z, \theta, \phi$

- output:
  - result: The result is the simulated sensor reading given the magnet position and the sensor position.

  