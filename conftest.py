import linedetection
import pytest
import numpy as np

from linedetection import LineDetection, Molecule

@pytest.fixture
def mol_1() -> Molecule:
    labelled_img = np.array([
        [0,0,0,2,1],
        [0,0,1,2,0],
        [3,0,0,2,0],
        [0,0,0,2,0]])
    mol = Molecule.create_from_labelled_image(labelled_img, mol_idx=2)
    return mol

@pytest.fixture
def mol_2() -> Molecule:
    labelled_img = np.array([
        [0,0,0,2,1],
        [0,0,1,1,0],
        [1,1,0,2,0],
        [0,0,0,2,0]])
    mol = Molecule.create_from_labelled_image(labelled_img)
    return mol

@pytest.fixture
def mol_from_yx() -> Molecule:
    x_ar = [10,20,30,40]
    y_ar = [0,1,2,3]
    mol = Molecule.create_from_yx_array(y_ar,x_ar,mol_idx=3)
    return mol

