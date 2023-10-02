"""Tests of IO"""
from pathlib import Path
from unittest import TestCase
from datetime import datetime
import numpy as np
import pytest

from topostats.line_detection import (
    LinearMolecule,
)

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

def test_create_from_labelled_image() -> None:
    labelled_ar = get_labelled_img_1()
    mol = LinearMolecule.create_from_labelled_image(labelled_ar,mol_idx=2)
    assert mol.width == 1
    assert mol.height == 4


def get_labelled_img_1() -> np.ndarray:
    labelled_img = np.array([
        0,0,0,2,1,
        0,0,1,2,0,
        3,0,0,2,0,
        0,0,0,2,0])
    return labelled_img



