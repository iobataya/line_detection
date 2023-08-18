# coding=utf-8
"""
This script find chains from a selected image
tracing by neighboring 8x pixels

vmaps containing under MIN_PIX are ignored.

2023 Ikuo Obataya, Quantum Design Japan
"""
import os,sys
sys.path.append(os.getenv("LINE_DET"))
#from importlib import reload # python 2.7 does not require this
import line_detection
reload(line_detection)
import line_detection_gwy
reload(line_detection_gwy)
from line_detection import LineDetection, GridTiler, VFilter
from line_detection_gwy import GwyApp, GwyField
import gwy, math

MIN_COUNT = 60
MAX_COUNT = 1000

# 1. Filtering by pixel count
msg = "Filtering by count {}-{}".format(MIN_COUNT,MAX_COUNT)
app.start_dialog("Filtereing", msg)
gui_cancelled = False

count_filter = VFilter(result_vmaps,MIN_COUNT,max_value=MAX_COUNT)

while(count_filter.by_count()):
    progress = count_filter._current_idx
    gui_alive = app.set_progress_dialog((progress + 1.0)/count_filter.filter_count)
    if not gui_cancelled and not gui_alive:
        gui_cancelled = True
        break

app.finish_dialog()

result_vmaps = count_filter.filtered
print("Result")
print("MIN_COUNT (int), MAX_COUNT (int) : {},{}".format(MIN_COUNT,MAX_COUNT))
print("count_filter.count (int) : count of source VMaps ({})".format(count_filter.filter_count))
print("result_vmaps (list(VMap)): resulting list of VMaps ({})".format(len(result_vmaps)))