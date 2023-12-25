# Line detection of polymer chain in AFM data
 - Recieves grains as NumPy data from TopoStats (https://github.com/AFM-SPM/TopoStats)
 - Detecting linear part within a polymer cahin by line-tracing analysis
 - Statistics (sigma of score and angle) are in pandas DataFrame
 - One can manipulate, obtain and visualize results in Jupyter notebook 

# Files and directory
- linemol.py : Classes for line tracing analysis
  * Molecule : handles image data of a molecule
  * Line : handles position of lines as a displacement vector
  * LineDetection : Main analysis methods
  * SpmPlot : Utilities for visualization for Jupyter notebook
- test_linemol.py : tests for classes in linemol
- conftest.py: configuration for test
- data/: configuration and image data (.gwy)

## example (detected lines)
![alt text](https://github.com/iobataya/line_detection/blob/main/blob/linemol_02_result.png)

## example (angle distribution)
![alt text](https://github.com/iobataya/line_detection/blob/main/blob/linemol_04_plot.png)
