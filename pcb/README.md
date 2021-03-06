# PCB Design

## Folder Description
"6cmX6cm_layout" is a 6cm X 6cm sensing array, and "9.8cmX9.8cm_layout" is a 9.8cm X 9.8cm sensing array. Both of them are composed of two printed circuit board, top board and bottom board.

"bom" refers to bill of materials, i.e., the components needed to build a sensing array.

## PCB Design Tool and Library
We utilize Altium Designer (version 20.0.13) to design PCB boards. The component library of capacitive and resistance adopts the standard library. The footprint of magnetic field sensor, MLX90393, can see https://www.snapeda.com/parts/MLX90393SLW-ABA-011-RE/Melexis%20Technologies/view-part/551380/?ref=search&t=MLX90393.

## Assembly
We use 9 separate pin headers to join the two boards together. A common 3-pin linear regulator is used to stabilize the output voltage, which is set to 3.3v. A 16-pin female header and a 12-pin female header are welded to the bottom board to connect the Bluefruit nRF52 Feather.


