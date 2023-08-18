# coding=utf-8
"""
This script apply adaptive threshold to image
with a single parameter, GAUSS_PIX

2023 Ikuo Obataya, Quantum Design Japan
"""
import os,sys
# Set folder path of this scripts for this env. variable
sys.path.append(os.getenv("LINE_DET"))
#from importlib import reload # python 2.7 does not require this
import line_detection_gwy
reload(line_detection_gwy)

from line_detection_gwy import GwyApp, GwyField
import gwy, math

GAUSS_PIX = 8

# 0. App setup
app = GwyApp()

# 1. Duplicate selected Gwyddion data
gwy.gwy_process_func_run("fix_zero", app.con, gwy.RUN_IMMEDIATE)
df = app.get_field()
df_raw = df.duplicate()
id_raw = app.add_field(df_raw)
app.set_title(id_raw, 'raw')

# 2. Apply gaussian filtered
df_gauss = df.duplicate()
df_gauss.filter_gaussian(GAUSS_PIX)
id_gauss = app.add_field(df_gauss)
app.set_title(id_gauss, 'gauss filtered at {}px'.format(GAUSS_PIX))

# 3. Subtract gaussian field from raw field
df_diff = df_raw.duplicate()
df_diff.subtract_fields(df_diff, df_gauss)
id_diff = app.add_field(df_diff)
app.set_title(id_diff, 'raw - gauss')

# 4. Get Otsu threshold to make binary image
df_proc = df_diff.duplicate()
threshold = df_proc.otsu_threshold()
print("Otsu threshold : {}".format(threshold))
df_proc.threshold(threshold, 0, 1.0)
id_proc = app.add_field(df_proc, showit = False)
app.set_title(id_proc, 'binary ({}px)'.format(GAUSS_PIX))

# 5. Recover source data
df_proc_x_raw = df_proc.duplicate()
df_proc_x_raw.multiply_fields(df_proc_x_raw, df_raw)
id_proc_x_raw = app.add_field(df_proc_x_raw, showit = True)
app.set_title(id_proc_x_raw, 'recovered')

result_gf = GwyField(df_proc_x_raw)
result_vmap = result_gf.vmap
print("Obtained data stored in result_gf as DataField and in result_vmap as VMap.")

app.show_browser()