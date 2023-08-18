# coding=utf-8
"""
This script generate tiled grid with chains from result_vmaps

2023 Ikuo Obataya, Quantum Design Japan
"""
import os,sys
sys.path.append(os.getenv("LINE_DET"))
#from importlib import reload # python 2.7 does not require this
import line_detection
reload(line_detection)
import line_detection_gwy
reload(line_detection_gwy)
from line_detection import GridTiler
from line_detection_gwy import GwyApp, GwyField
import gwy, math

GRID_COLS = 20

# 0. App
app = GwyApp()
app.show_browser()
gui_cancelled = False

# 1. Tile all of chain
chains_vmap = []
i = 0
n = len(result_vmaps)
app.start_dialog("Calculation", "Setting up grid...")
tiler = GridTiler(result_vmaps,grid_cols=GRID_COLS)
tiler.sort_by_angle(reverse=False)
tiler.sort_by_count(reverse=True)

app.set_msg_dialog("Tiling all chains...")
for i in range(n):
    chains_vmap.append(tiler.to_be_tiled[i])
    tiler.tiling_add()
    gui_alive = app.set_progress_dialog((i + 1.0)/n)
    if not gui_cancelled and not gui_alive:
        gui_cancelled = True
        break
tiled_gf = GwyField(tiler.tiled_vmap)
app.add_field(tiled_gf.field, showit = True, title = 'tiled')
app.finish_dialog()

result_vmap = tiler.tiled_vmap
result_vmaps = tiler.to_be_tiled
print("# Conditions ")
print("GRID_COLS (int) : number of grid columns, {}".format(GRID_COLS))
print("# Result #")
print("result_vmap (VMap) : tiled vmap")
print("result_vmaps (list(VMap)) : Sorted list of tiledVMaps")
