import pytest
import numpy as np
from typing import List
from linemol import Molecule, LineDetection

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

@pytest.fixture
def mol_list() -> List[Molecule]:
    labelled_img = np.array([
        [0,0,0,0,1,1,0,0],
        [0,0,1,1,0,0,0,0],
        [1,1,0,2,2,0,2,2],
        [0,0,0,0,0,2,0,0]])
    source_img = np.array([
        [0,0,0,0,3,4,0,0],
        [0,0,5,2,0,0,0,0],
        [8,8,0,4,5,0,4,3],
        [0,0,0,0,0,4,0,0]])
    mol_list = Molecule.create_all_from_labelled_image(labelled_img, source_img)
    return mol_list

@pytest.fixture
def ld0() -> LineDetection:
    labelled_img = np.array([
        [0,0,0,0,1,1,0,0],
        [0,0,1,1,0,0,0,0],
        [1,1,0,2,2,0,2,2],
        [0,0,0,0,0,2,0,0]])
    source_img = np.array([
        [0,0,0,0,3,4,0,0],
        [0,0,5,2,0,0,0,0],
        [8,8,0,4,5,0,4,3],
        [0,0,0,0,0,4,0,0]])

    config = {"min_len":2,"max_len":100,"allowed_empty":0}
    ld = LineDetection(labelled_img, source_img, **config)
    return ld
