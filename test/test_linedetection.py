"""Tests of IO"""
from pathlib import Path
from unittest import TestCase
from datetime import datetime
import numpy as np
import pytest

from linedetection import (
    LinearMolecule,
)

BASE_DIR = Path.cwd()

def test_create_from_labelled_image(mol_1) -> None:
    assert mol_1.width == 1
    assert mol_1.height == 4
    assert mol_1.mol_idx == 2

def test_create_from_yx_array(mol_from_yx) -> None:
    assert mol_from_yx.width == 1
    assert mol_from_yx.height == 4
    assert mol_from_yx.mol_idx == 3

def test_init(mol_2:LinearMolecule) -> None:
    assert mol_2.count() == 5
    assert len(mol_2.y) == mol_2.count()
    assert mol_2.width == 5
    assert mol_2.height == 3

def test_count(mol_2:LinearMolecule) -> None:
    assert mol_2.count() == 5

def test_orig_x(mol_from_yx:LinearMolecule):
    assert mol_from_yx.orig_x()[0] == 3
    assert mol_from_yx.orig_x()[3] == 3

def test_orig_y(mol_from_yx:LinearMolecule):
    assert mol_from_yx.orig_y()[0] == 0
    assert mol_from_yx.orig_y()[3] == 3

def test_values(mol_from_yx:LinearMolecule):
    assert mol_from_yx.values()[0] == 0.1

def test_get_yx(mol_from_yx:LinearMolecule):
    assert mol_from_yx.get_yx(0) == (0,0)

def test_get_orig_yx(mol_from_yx:LinearMolecule):
    assert mol_from_yx.get_orig_yx(0) == (0.0,3.0)

def test_get_vector(mol_from_yx:LinearMolecule):
    (yx1,yx2) = mol_from_yx.get_vector(0,3)
    assert yx1[0] == 0
    assert yx2[1] == 0

def test_get_displacement(mol_from_yx:LinearMolecule):
    disp = mol_from_yx.get_displacement(0,3)
    assert disp[0] == -3.0
    assert disp[1] == 0.0

def test_x(mol_1:LinearMolecule):
    assert mol_1.x[0] == 0
    assert mol_1.x[3] == 0

def test_y(mol_1:LinearMolecule):
    assert mol_1.y[0] == 0
    assert mol_1.y[3] == 3

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
    print(f"matrix.sum() is {np.sum(matrix,axis=2)}")

