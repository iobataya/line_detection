# coding=utf-8
"""
This script find chains from a selected image
tracing by neighboring 8x pixels
2023 Ikuo Obataya, Quantum Design Japan
"""
import os,sys
sys.path.append(os.getenv("LINE_DET"))
import gwy
from line_detection_gwy import GwyApp, GwyField, find_chains
from vectors import VMapList

################
# Parameters   #
################
MIN_PIX = 100
MAX_CHAIN = 1000

if not 'app' in globals():
    app = GwyApp()

result_vmaps = find_chains(app, MIN_PIX, MAX_CHAIN, gui_cancelled=False)

print("# Conditions ")
print("(MIN_PIX, MAX_CHAIN) = ({}, {})".format(MIN_PIX, MAX_CHAIN))
print("# Result #")
print("Neighboring VMaps ({} found) are stored in result_vmaps as VMapList".format(len(result_vmaps)))
