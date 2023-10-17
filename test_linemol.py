"""Tests of linemol"""
from pathlib import Path
from unittest import TestCase
from datetime import datetime
from typing import List
import numpy as np
import pytest

from linemol import (
    LineDetection as ld,
    Molecule,
)
BASE_DIR = Path.cwd()
(YAXIS, XAXIS) = (0,1)

#region test Molecule
def test_init(mol_2:Molecule, mol_from_yx:Molecule) -> None:
    assert mol_2.count() == 5
    assert len(mol_2.y) == mol_2.count()
    assert mol_2.width == 5
    assert mol_2.height == 3
    assert mol_2.yx.dtype == np.int32
    assert mol_from_yx.yxT[0][YAXIS] == 0
    assert mol_from_yx.yxT[0][XAXIS] == 0
    with pytest.raises(IndexError):
        y_ar = [0,500]
        x_ar = [10,120]
        mol = Molecule(np.array([y_ar,x_ar]))

def test_get_yx(mol_from_yx:Molecule):
    assert mol_from_yx.get_yx(0) == (0,0)

def test_get_vector(mol_from_yx:Molecule):
    # idx 1: yxB (0,0)
    vec = mol_from_yx.get_vector(1)
    assert vec.dtype == np.int32
    assert vec[XAXIS] == 10
    assert vec[YAXIS] == 1

def test_get_displacement_vector(mol_from_yx:Molecule):
    # idx 0: (0,0)
    # idx 3: (30,3)
    # disp:  (-30,-3)
    disp_vec = mol_from_yx.get_displacement_vector(0,3)
    assert disp_vec.dtype == np.int32
    assert disp_vec[XAXIS] == -30
    assert disp_vec[YAXIS] == -3

def test_count(mol_2:Molecule, mol_from_yx:Molecule) -> None:
    assert mol_2.count() == 5
    assert mol_from_yx.count() == 4

def test_create_from_labelled_image(mol_1: Molecule) -> None:
    assert mol_1.width == 1
    assert mol_1.height == 4
    assert mol_1.mol_idx == 2

def test_create_from_yx_array(mol_from_yx: Molecule) -> None:
    assert mol_from_yx.width == 31
    assert mol_from_yx.height == 4
    assert mol_from_yx.mol_idx == 3

def test_create_all_from_labelled_image(mol_list: List[Molecule]) -> None:
    assert len(mol_list) == 2

    mol1 = mol_list[0]
    assert mol1.count() == 6
    assert mol1.width == 6
    assert mol1.height == 3
    assert mol1.src_img[0][4] == 3

    mol2 = mol_list[1]
    assert mol2.count() == 5
    assert mol2.width == 5
    assert mol2.height == 2
    offset_x = mol2.src_left
    offset_y = mol2.src_top
    assert mol2.src_img[3 - offset_y][5 - offset_x] == 4


def test_get_displacement_matrix(mol_2: Molecule) -> None:
    assert mol_2.width == 5
    assert mol_2.height == 3
    matrix = mol_2.get_displacement_matrix()
    assert matrix.shape == (5, 5, 2)
    for i in range(5):
    # diagonal elements must be (0,0)
        assert matrix[i][i][0] == 0.0
        assert matrix[i][i][1] == 0.0
        # direction of elements crossing diagonal should be opposite
        if i > 1 and i < 4:
            x = matrix[i - 1][i - 1][1] + matrix[i + 1][i + 1][1]
            y = matrix[i - 1][i - 1][0] + matrix[i + 1][i + 1][0]
            assert x == 0.0
            assert y == 0.0

def test_get_mask(mol_2:Molecule) -> None:
    """ generated mask from following image
       [[0,0,0,2,1],
        [0,0,1,1,0],
        [1,1,0,2,0],
        [0,0,0,2,0]]
    """
    expected = np.array([
        [0,0,0,0,1],
        [0,0,1,1,0],
        [1,1,0,0,0]])
    mask = mol_2.get_mask()
    assert np.array_equiv(mask, expected) == True

def test_set_source_image(mol_list:Molecule) -> None:
    """ source image should be croppted to shape like
        [0,0,0,0,3,4],
        [0,0,5,2,0,0],
        [8,8,0,4,5,0],
    """
    mol = mol_list[0]
    assert mol.src_img.shape == (3, 6)
    sum = 3 + 4 + 2 + 5 + 8 + 8 + 4 + 5
    assert mol.src_img.sum().sum() == sum

def test_str(mol_1:Molecule):
    expected =  "Molecule:2:4 pixels, 1 x 4"
    assert mol_1.__str__() == expected

def test_repr(mol_1:Molecule):
    expected = "2(4 px) 1 x 4"
    assert mol_1.__repr__() == expected

#endregion

#region test LineDetection

def test_init(linedet0:ld):
    assert linedet0.mol_count() == 2
    assert linedet0.config["min_len"] == 1
    assert linedet0.config["allowed_empty"] == 0

def test_filter_by_length(linedet0:ld):
    mol = linedet0.molecules[0]
    (pix_pairs, results) = linedet0.filter_by_length(mol)
    assert results["source vectors"] == 6 * 5  / 2
    assert results["only in 12 quad"] == 6 * 5 / 2 - 4 - 4 - 1  # TODO: to be checked

#region test static methods
def test_get_blank_image() -> None:
    blank = ld.get_blank_image(200, 100, dtype=np.int32)
    assert blank.shape[0] == 200
    assert blank.shape[1] == 100

def test_get_line_mask():
    pass

def test_get_emphasized():
    pass

def test_filter_by_quadrant():
    pass

def test_draw_line():
    blank = ld.get_blank_image(100, 100, dtype=np.int32)
    # horizontal
    y = 2
    (x1, x2) = (2, 10)
    ld.draw_line(x1, y, x2, y, blank)
    assert blank[y][x1 - 1] == 0
    assert blank[y][x1    ] == 1
    assert blank[y][x2 - 1] == 1
    assert blank[y][x2    ] == 1
    assert blank[y][x2 + 1] == 0
    assert blank.sum() == 9

    # vertical
    x = 15
    (y1, y2) = (12, 20)
    ld.draw_line(x, y1, x, y2, blank)
    assert blank[y1 - 1][x] == 0
    assert blank[y1    ][x] == 1
    assert blank[y2 - 1][x] == 1
    assert blank[y2    ][x] == 1
    assert blank[y2 + 1][x] == 0
    assert blank.sum() == 18

    # line
    (x1, y1) = (30, 30)
    (x2, y2) = (50, 50)
    ld.draw_line(x1, y1, x2, y2, blank)
    assert blank[y1 - 1][x1 - 1] == 0
    assert blank[y1    ][x1    ] == 1
    assert blank[y1 + 1][x1 + 1] == 1
    assert blank[y2 - 1][x2 - 1] == 1
    assert blank[y2    ][x2    ] == 1
    assert blank[y2 + 1][x2 + 1] == 0
    assert blank.sum() == 39
#endregion

#endregion




