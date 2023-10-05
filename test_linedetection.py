"""Tests of IO"""
from pathlib import Path
from unittest import TestCase
from datetime import datetime
import numpy as np
import pytest

from linedetection import (
    LineDetection,
    Molecule,
)

BASE_DIR = Path.cwd()

(YAXIS, XAXIS) = (0,1)

def test_init(mol_2:Molecule,mol_from_yx:Molecule) -> None:
    assert mol_2.count() == 5
    assert len(mol_2.y) == mol_2.count()
    assert mol_2.width == 5
    assert mol_2.height == 3
    assert mol_2.yxB.dtype == np.byte
    assert mol_from_yx.orig_yx.T[0][YAXIS] == 0
    assert mol_from_yx.orig_yx.T[0][XAXIS] == 10
    assert mol_from_yx.yxBT[0][YAXIS] == 0
    assert mol_from_yx.yxBT[0][XAXIS] == 0
    with pytest.raises(IndexError):
        y_ar = [0,500]
        x_ar = [10,120]
        mol = Molecule(np.array([y_ar,x_ar]))

def test_orig_x(mol_from_yx:Molecule):
    assert mol_from_yx.orig_x()[0] == 10
    assert mol_from_yx.orig_x()[3] == 40

def test_orig_y(mol_from_yx:Molecule):
    assert mol_from_yx.orig_y()[0] == 0
    assert mol_from_yx.orig_y()[3] == 3

def test_get_yx(mol_from_yx:Molecule):
    assert mol_from_yx.get_yx(0) == (0,0)


def test_get_orig_yx(mol_from_yx:Molecule):
    assert mol_from_yx.get_orig_yx(0) == (0.0,10)

def test_get_vector(mol_from_yx:Molecule):
    # idx 1: yxB (0,0)
    vec = mol_from_yx.get_vector(1)
    assert vec.dtype == np.byte
    assert vec[XAXIS] == 10
    assert vec[YAXIS] == 1

def test_get_displacement_vector(mol_from_yx:Molecule):
    # idx 0: (0,0)
    # idx 3: (30,3)
    # disp:  (-30,-3)
    disp_vec = mol_from_yx.get_displacement_vector(0,3)
    assert disp_vec.dtype == np.byte
    assert disp_vec[XAXIS] == -30
    assert disp_vec[YAXIS] == -3

def test_count(mol_2:Molecule) -> None:
    assert mol_2.count() == 5

def test_count(mol_from_yx:Molecule):
    assert mol_from_yx.count() == 4


def test_create_from_labelled_image(mol_1) -> None:
    assert mol_1.width == 1
    assert mol_1.height == 4
    assert mol_1.mol_idx == 2

def test_create_from_yx_array(mol_from_yx) -> None:
    assert mol_from_yx.width == 31
    assert mol_from_yx.height == 4
    assert mol_from_yx.mol_idx == 3

def test_get_displacement_matrix(mol_2) -> None:
    assert mol_2.width == 5
    assert mol_2.height == 3
    matrix = mol_2.get_displacement_matrix()
    assert matrix.shape == (5,5,2)
    for i in range(5):
    # diagonal elements must be (0,0)
        assert matrix[i][i][0] == 0.0
        assert matrix[i][i][1] == 0.0
        # direction of elements crossing diagonal should be opposite
        if i > 1 and i < 4:
            x = matrix[i-1][i-1][1] + matrix[i+1][i+1][1]
            y = matrix[i-1][i-1][0] + matrix[i+1][i+1][0]
            assert x == 0.0
            assert y == 0.0

def test_filter_by_quadrant():
    pass

def test_filter_by_length():
    pass

def test_get_iage():
    pass

def test_get_blank_image():
    mol = Molecule(np.array([[10, 109], [10, 109]]))
    assert mol.width == 100
    assert mol.height == 100
    blank = mol.get_blank_image(dtype=np.byte)
    assert blank.dtype == np.byte
    assert blank.shape == (100, 100)

def test_str():
    pass

def test_repr():
    pass

def test_get_line_mask():
    pass

def test_draw_line():
    mol = Molecule(np.array([[10, 109], [10, 109]]))
    blank = mol.get_blank_image(dtype=np.byte)
    # horizontal
    y = 2
    (x1, x2) = (2, 10)
    mol.draw_line(x1, y, x2, y, blank)
    assert blank[y][x1 - 1] == 0
    assert blank[y][x1    ] == 1
    assert blank[y][x2 - 1] == 1
    assert blank[y][x2    ] == 1
    assert blank[y][x2 + 1] == 0
    assert blank.sum() == 9

    # vertical
    x = 15
    (y1, y2) = (12, 20)
    mol.draw_line(x, y1, x, y2, blank)
    assert blank[y1 - 1][x] == 0
    assert blank[y1    ][x] == 1
    assert blank[y2 - 1][x] == 1
    assert blank[y2    ][x] == 1
    assert blank[y2 + 1][x] == 0
    assert blank.sum() == 18

    # line
    (x1, y1) = (30, 30)
    (x2, y2) = (50, 50)
    mol.draw_line(x1, y1, x2, y2, blank)
    assert blank[y1 - 1][x1 - 1] == 0
    assert blank[y1    ][x1    ] == 1
    assert blank[y1 + 1][x1 + 1] == 1
    assert blank[y2 - 1][x2 - 1] == 1
    assert blank[y2    ][x2    ] == 1
    assert blank[y2 + 1][x2 + 1] == 0
    assert blank.sum() == 39


