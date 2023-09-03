# coding=utf-8
"""
This script generate tiled grid with chains from result_vmaps

2023 Ikuo Obataya, Quantum Design Japan
"""
import os,sys
sys.path.append(os.getenv("LINE_DET"))
import line_detection_gwy
reload(line_detection_gwy)
from line_detection_gwy import GwyApp, GwyField,GridTiler,VMapList,VMap,V
import gwy, math
################
# Parameter    #
################
GRID_COLS = 10

if not 'app' in globals():
    app =GwyApp()

vmap_list = result_vmaps
gui_cancelled = False
app.start_dialog("Calculation", "Setting up grid...")

tiler = GridTiler(vmap_list, grid_cols=GRID_COLS)
sort_tp = [VMap.SORT_BY_COUNT, VMap.SORT_BY_Y]
rev_tp = [True,False]
tiler.sort_by(sort_tp,rev_tp)

app.set_msg_dialog("Tiling all chains...")
(i, n) = (0, len(vmap_list))
for i in range(n):
    tiler.tiling_add()
    gui_alive = app.set_progress_dialog((i + 1.0)/n)
    if not gui_cancelled and not gui_alive:
        gui_cancelled = True
        break
app.finish_dialog()

tiled_vmap = tiler.tiled_vmap
sorted_chains = tiler.to_be_tiled

tiled_gf = GwyField(tiled_vmap)
app.add_field(tiled_gf.field, showit = True, title = 'tiled')

print("# Conditions ")
print("GRID_COLS (int) : number of grid columns, {}".format(GRID_COLS))
print("# Result #")
print("tiled_vmap (VMap) : tiled vmap")
print("sorted_chains (VMapList) : Sorted list of chains")

# TODO: sort by y position - drift may change angle of chains from top to bottom.
