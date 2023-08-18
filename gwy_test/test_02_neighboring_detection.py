# coding=utf-8
"""
This script find chains from a selected image
tracing by neighboring 8x pixels

2023 Ikuo Obataya, Quantum Design Japan
"""
import os,sys
sys.path.append(os.getenv("LINE_DET"))
#from importlib import reload # python 2.7 does not require this
import line_detection_gwy
reload(line_detection_gwy)
from line_detection_gwy import GwyApp, GwyField
import gwy, math

MIN_PIX = 30
MAX_CHAIN = 200

# 0. App
app = GwyApp()
app.show_browser()
dfield = app.get_field()
gui_cancelled = False

# 1. Gwyddion data -> Otsu threshold classification
app.start_dialog("Calculation","Getting neighboring...")

gf = GwyField(dfield)
vmap = gf.vmap.copy()
initial_count = len(vmap)
current_count = initial_count
neighboring = []
while(current_count > 0):
    found_vmap = vmap._get_neighboring_from(vmap.first_vec())
    if len(found_vmap) >= MIN_PIX:
        neighboring.append(found_vmap)
    current_count = len(vmap)
    if len(neighboring) >= MAX_CHAIN:
        break
    progress = initial_count - current_count
    gui_alive = app.set_progress_dialog((progress + 1.0)/initial_count)
    if not gui_cancelled and not gui_alive:
        gui_cancelled = True
        break
app.finish_dialog()

result_vmaps = neighboring
print("# Conditions ")
print("(MIN_PIX, MAX_CHAIN) = ({}, {})".format(MIN_PIX, MAX_CHAIN))
print("# Result #")
print("Neighboring VMaps ({} found) are stored in result_vmaps as list".format(len(result_vmaps)))
