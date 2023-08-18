# coding=utf-8
"""
This script analyse how much a segmented chain has linear line in the shape.
Cross product between a reference angle vector and all possible two pixels
are used to detect angle of the vector

2023 Ikuo Obataya, Quantum Design Japan

TODO: file exporting
"""
import os,sys,math,time
sys.path.append(os.getenv("LINE_DET"))
#from importlib import reload # python 2.7 does not require this
import line_detection
reload(line_detection)
import line_detection_gwy
reload(line_detection_gwy)
from line_detection import LineDetection, GridTiler, VFilter
from line_detection_gwy import GwyApp, GwyField, GwyCurve
import gwy

# Conditions
MIN_LENGTH = 10
MAX_LENGTH = 100
MAX_PIXELS = 1000
ANGLE_STEP = 3

# 0. App
app = GwyApp()
dfield = app.get_field() # current field in data_browser of Gwyddion
gui_cancelled = False

# 1. set up conditions
chain = GwyField(dfield)
if len(chain.vmap) > MAX_PIXELS:
    raise ValueError("too many pixels !")

# TODO: MIN_LEN should be optimized
chain_diagonal = chain.vmap.get_diagonal()
(chain_width,chain_height) = (chain_diagonal.x(), chain_diagonal.y())
MIN_LEN = int(min(chain_width,chain_height)/2)
print("MIN_LEN was set to {}".format(MIN_LEN))

print("Source Vmap: {}\ntotal:{}".format(chain.vmap, len(chain.vmap)))

# 2. filtering by line minimum length
app.start_dialog("Calculation", "Filtering by length of line...")
ld = LineDetection(chain.vmap,
                    min_length=MIN_LENGTH,
                    max_length=MAX_LENGTH,
                    allowed_empty=2,
                    enable_cache=True)

# 3. Line tracing
app.set_msg_dialog("Analyzing line in the chains ()...")
result_sum = {}
result_angles = {}
angle_count = {}
weighed_angles = {}
max_sum_angles = {}
max_sum_lines = {}

ns4 = time.clock()
sum_average = None
sum_max = None

angle_step = ANGLE_STEP
for i in range(0, ld.vv_count):
    line_vv = ld.vv_list[i].round()
    line_len = line_vv.distance()
    sum_z = ld.sum_along_line(ld.src_vmap, line_vv)
    if sum_z > 0:
        result_sum[line_vv] = sum_z
        angle = line_vv.displacement().angle(quadrant12=True)
        result_angles[line_vv] = angle
        bin_start = angle - (angle % angle_step)
        if bin_start in angle_count:
            angle_count[bin_start] += 1
            weighed_angles[bin_start] += sum_z
            if sum_z > max_sum_angles[bin_start]:
                max_sum_angles[bin_start] = sum_z
                max_sum_lines[bin_start] = line_vv
        else:
            angle_count[bin_start] = 1
            weighed_angles[bin_start] = sum_z
            max_sum_angles[bin_start] = sum_z
            max_sum_lines[bin_start] = line_vv

        if sum_average == None:
            sum_average = sum_z
            sum_max = sum_z
        else:
            sum_average = (sum_average + sum_z) / 2
            sum_max = max(sum_z, sum_max)

    progress = (i + 1.0) / ld.vv_count
    gui_alive = app.set_progress_dialog(progress)
    if not gui_alive and not gui_cancelled:
        gui_cancelled = True
        break
ns5 = time.clock()
print("Line tracing took {} clocks.".format(ns5 - ns4))
app.finish_dialog()


# 3. plot angle_count
x_real = 180
y_values = []
target_y_dict = weighed_angles
total_value = 0
for i in range(0,180,ANGLE_STEP):
    angle = float(i)
    if angle in target_y_dict:
        y_values.append(target_y_dict[angle])
        total_value += target_y_dict[angle]
    else:
        y_values.append(0)

if total_value == 0:
    raise ValueError("No distinguished linear region in this chain.")

gwycurve = GwyCurve()
gwycurve.add_curve(y_values, x_real=x_real, color_idx = 0, desc = 'sum z')
app.add_curve(gwycurve.gmodel, showit = True)


# 5. Emphasize the line
emphasized = chain.vmap.copy()
for i in range(0,180,ANGLE_STEP):
    angle = float(i)
    if angle in max_sum_lines:
        if max_sum_angles[angle] < sum_average:
            continue
        emphasized.add_line(max_sum_lines[angle], value = max_sum_angles[angle])

emphasized_gf = GwyField(emphasized)
app.add_field(emphasized_gf.field, showit=True, title='Emphasized')

