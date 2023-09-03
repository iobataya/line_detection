# coding=utf-8
"""
This script analyse how much a segmented chain has linear line in the shape.
Cross product between a reference angle vector and all possible two pixels
are used to detect angle of the vector

2023 Ikuo Obataya, Quantum Design Japan
"""
# TODO: file exporting
import os,sys,math,time
sys.path.append(os.getenv("LINE_DET"))
#from importlib import reload # python 2.7 does not require this
import line_detection_gwy
reload(line_detection_gwy)
from line_detection_gwy import GwyApp, GwyField, GwyCurve, GwyVMapListIO
from line_detection_gwy import LineDetection,LineDetResult,LineDetResults
from line_detection_gwy import plot_at_angles, draw_lines_at_angles, get_vvcache
import gwy

# Conditions
MIN_LENGTH = 10
MAX_LENGTH = 50
MAX_PIXELS = 1000
ANGLE_STEP = 5

app = GwyApp()
gui_cancelled = False

SAVE_RESULTS = False
PLOT_Z_SUM = True
ADD_FIELD = False
FILE_DIR = os.getenv('LINE_DET')
FILE_NAME = 'hoge.csv'
vvcache = get_vvcache()

app.start_dialog("Calculation", "")

# input vmap range PO:80
#for i in range(len(result_vmaps)):    # ! CAUTION !
n = len(result_vmaps)
n = 3
for i in range(0,n):
    chain_vmap = result_vmaps[i]

    # 1. Line tracing
    if len(chain_vmap) > MAX_PIXELS:
        continue
#    print("Source Vmap: {}\ntotal:{}".format(chain_vmap, len(chain_vmap)))
    app.set_msg_dialog("Analyzing line in the chains ()...")

    ld = LineDetection(chain_vmap,
                        min_length=MIN_LENGTH,
                        max_length=MAX_LENGTH,
                        allowed_empty=1,
                        vvcache=None,
                        exclude_covered=True)

    line_results = LineDetResults()
    for i in range(0, ld.vv_count):
        line_vv = ld.vv_list[i].round()
        sum_z = ld.sum_along_line(ld.src_vmap, line_vv)
        if sum_z > 0:
            result = LineDetResult(line_vv, sum_z, angle_step=ANGLE_STEP)
            line_results.append(result)
        progress = (i + 1.0) / ld.vv_count
        gui_alive = app.set_progress_dialog(progress)
        if not gui_alive and not gui_cancelled:
            gui_cancelled = True
            break
    print("All combination was {}, filtered to {} by length limits.".format(ld.vv_total_count,ld.vv_count))

    # 3. Line tracing
    if SAVE_RESULTS:
        with open(os.path.join(FILE_DIR,FILE_NAME), mode='a') as f:
            f.write(line_results.get_csv_data_lines(step_angle=ANGLE_STEP))
        #print(line_results.get_csv_data_lines(step_angle=ANGLE_STEP))


# 2. plot z-sum at angles
    if PLOT_Z_SUM:
        plot_at_angles(app,line_results.sum_at_angles,ANGLE_STEP,title='sum z')

# 3. Generate lines at max sum for each angles
    if ADD_FIELD:
        draw_lines_at_angles(app, line_results,chain_vmap,ANGLE_STEP,title='Detected lines')

app.finish_dialog()