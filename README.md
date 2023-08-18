# Line detection of polymer chain in AFM data on Gwyddion
 - Picking up polymer chains in AFM data by neighboring pixel analysis
 - Detecting linear part within a polymer cahin by line-tracing analysis
 - Summary of distribution of angles of linear part

# Files
- vectors.py : basic classes for vectors and vector maps using standard python library (2.7 or above)
- line_detection.py : line tracing analysis of vector maps
- line_detection_gwy.py : Utility classes for Gwyddion console.
- test : folder containing unittests for vectors.py and line_detection.py that are not dependent on pygwy
- gwy_test: folder containing test/analysis scripts working **only on Gwyddion console**.

