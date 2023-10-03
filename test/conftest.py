import linedetection
import pytest
import numpy as np

from linedetection import LinearMolecule

@pytest.fixture
def mol_1() -> LinearMolecule:
    labelled_img = np.array([
        [0,0,0,2,1],
        [0,0,1,2,0],
        [3,0,0,2,0],
        [0,0,0,2,0]])
    mol = LinearMolecule.create_from_labelled_image(labelled_img, mol_idx=2)
    return mol

@pytest.fixture
def mol_2() -> LinearMolecule:
    labelled_img = np.array([
        [0,0,0,2,1],
        [0,0,1,1,0],
        [1,1,0,2,0],
        [0,0,0,2,0]])
    mol = LinearMolecule.create_from_labelled_image(labelled_img)
    return mol

@pytest.fixture
def mol_from_yx() -> LinearMolecule:
    y_ar = [0,1,2,3]
    x_ar = [3,3,3,3]
    mol = LinearMolecule.create_from_yx_array(y_ar,x_ar,value=0.1,mol_idx=3)
    return mol
