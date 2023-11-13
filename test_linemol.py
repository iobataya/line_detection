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
    Line,
)
BASE_DIR = Path.cwd()
(YAXIS, XAXIS) = (0,1)

class TestLine:
    def test_init(self) -> None:
        # vertical
        hl = Line(0, 5)
        assert len(hl.yxT) == 6
        assert [5, 0] in hl.yxT
        assert [0, 0] in hl.yxT
        # horizontal
        hh = Line(-3, 0)
        assert len(hh.yxT) == 4
        assert [0, -3] in hh.yxT
        assert [0, 0] in hh.yxT

        # line
        line = Line(5,5)
        assert len(line.yxT) == 6
        assert line.yxT[5][YAXIS] == 5
        line2 = Line(-4,3)
        assert len(line2.yxT) == 5
        assert [3, -4] in line2.yxT

    def test_create_from_yx(self):
        yx1 = np.array([12, -5])
        yx2 = np.array([8, -9])
        line = Line.create_from_yx(yx1, yx2)
        assert [0,0] in line.yxT
        shifted = line.shifted_yxT()
        assert [12, -5] in shifted
        assert [8, -9] in shifted

    def test_create_from_pos(self):
        line = Line.create_from_pos(-5, 12, -9, 8)
        assert [0,0] in line.yxT
        shifted = line.shifted_yxT()
        assert [12, -5] in shifted
        assert [8, -9] in shifted


    def test_create_from_array(self):
        line = Line.create_from_pos(-5, 12, -9, 8)
        yxT = line.yxT
        from_ar = Line.create_from_array(yxT, offset_x=-8, offset_y=9)
        assert [0, 0] in from_ar.yxT
        assert [0, 0] in from_ar.shifted_yxT()
        assert line.yxT is not from_ar.yxT

    def test_get_x1y1x2y2(self):
        line = Line.create_from_pos(-5, 12, -9, 8)
        (x1, y1, x2, y2) = line.get_x1y1x2y2()
        assert (x1, y1, x2, y2) == (-5, 12, -9, 8)

    def test_get_mask(self) -> None:
        line = Line(3, -4, offset_x=0, offset_y=4)
        mask = line.get_mask(10 ,10)
        assert mask[4][0] == True
        assert mask[0][3] == True

    def test_str(self) -> None:
        line = Line(10,10)
        assert str(line) == "Line (10,10)-(0,0) displacement:(10,10)"
        line = Line(-5,-4)
        assert str(line) == "Line (-5,-4)-(0,0) displacement:(-5,-4)"

class TestMolecule:
    def test_init(self, mol_2:Molecule, mol_from_yx:Molecule) -> None:
        assert mol_2.count() == 5
        assert len(mol_2.y) == mol_2.count()
        assert mol_2.width == 5
        assert mol_2.height == 3
        assert mol_2.yxT.dtype == np.int32
        assert mol_from_yx.yxT[0][YAXIS] == 0
        assert mol_from_yx.yxT[0][XAXIS] == 0
        with pytest.raises(IndexError):
            y_ar = [0,500]
            x_ar = [10,120]
            mol = Molecule(np.array([y_ar,x_ar]))

    def test_get_yx(self, mol_from_yx:Molecule):
        assert mol_from_yx.get_yx(0) == (0,0)

    def test_get_vector(self, mol_from_yx:Molecule):
        # idx 1: yxB (0,0)
        vec = mol_from_yx.get_vector(1)
        assert vec.dtype == np.int32
        assert vec[XAXIS] == 10
        assert vec[YAXIS] == 1

    def test_get_displacement_vector(self, mol_from_yx:Molecule):
        # idx 0: (0,0)
        # idx 3: (30,3)
        # disp:  (-30,-3)
        disp_vec = mol_from_yx.get_displacement_vector(0,3)
        assert disp_vec.dtype == np.int32
        assert disp_vec[XAXIS] == -30
        assert disp_vec[YAXIS] == -3

    def test_count(self, mol_2:Molecule, mol_from_yx:Molecule) -> None:
        assert mol_2.count() == 5
        assert mol_from_yx.count() == 4

    def test_create_from_labelled_image(self, mol_1: Molecule) -> None:
        assert mol_1.width == 1
        assert mol_1.height == 4
        assert mol_1.mol_idx == 2

    def test_create_from_yx_array(self, mol_from_yx: Molecule) -> None:
        assert mol_from_yx.width == 31
        assert mol_from_yx.height == 4
        assert mol_from_yx.mol_idx == 3

    def test_create_all_from_labelled_image(self, mol_list: List[Molecule]) -> None:
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


    def test_get_displacement_matrix(self, mol_2: Molecule) -> None:
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

    def test_get_displacement_vectors(self, mol_1:Molecule, mol_2:Molecule):
        vecs1 = mol_1.get_displacement_vectors()
        assert len(vecs1) == 4 * 3 // 2  # combinations n(n-1)/2
        assert vecs1.shape == (len(vecs1), 4)  # [[id0, id1, y,x],]

        vecs2 = mol_2.get_displacement_vectors()
        assert len(vecs2) == 5 * 4 // 2
        assert vecs2.shape == (len(vecs2), 4)

    def test_get_mask(self, mol_2:Molecule) -> None:
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

    def test_set_source_image(self, mol_list:Molecule) -> None:
        """ source image should be croppted to shape like
            [0,0,0,0,3,4],
            [0,0,5,2,0,0],
            [8,8,0,4,5,0],
        """
        mol = mol_list[0]
        assert mol.src_img.shape == (3, 6)
        sum = 3 + 4 + 2 + 5 + 8 + 8 + 4 + 5
        assert mol.src_img.sum().sum() == sum

    def test_str(self, mol_1:Molecule):
        expected =  "Molecule:2:4 pixels, 1 x 4"
        assert mol_1.__str__() == expected

    def test_repr(self, mol_1:Molecule):
        expected = "2(4 px) 1 x 4"
        assert mol_1.__repr__() == expected

class TestLineDetection:
    def test_init(self, ld0:ld):
        assert ld0.mol_count() == 3
        assert ld0.config["min_len"] == 2
        assert ld0.config["allowed_empty"] == 0

    def test_filter_by_length(self, ld0:ld):
        mol = ld0.molecules[0]
        mol_idx = mol.mol_idx
        filtered = ld0.filter_by_length(mol)
        assert len(filtered) == 10
        len_filtered = ld0.stat_df.loc[ld0.stat_df["mol_idx"]==mol_idx]["len_filtered"].tolist()
        assert len_filtered[0] == 10


    def test_add_len_filter_stat(self, ld0:ld):
        mol = ld0.molecules[0]
        mol_idx = mol.mol_idx
        filtered = ld0.filter_by_length(ld0.molecules[0])
        ld0._add_len_filter_stat(ld0.molecules[0],100,10)
        total = ld0.stat_df.loc[ld0.stat_df["mol_idx"]==mol_idx]["total_vecs"].tolist()
        assert 100 in total

    def test_score_lines(self, ld0:ld):
        mol = ld0.molecules[0]
        filtered = ld0.filter_by_length(mol)
        result_count = ld0.score_lines(mol,filtered)
        assert result_count > 0
        sorted = ld0.score_df.sort_values("score",ascending=False)
        high_score = sorted["score"].iloc[0]
        assert high_score == 30

    def test_score_line(self, ld0:ld):
        mol = ld0.molecules[0]
        ld0.config["allowed_empty"] = 10
        (score, _) = ld0.score_line(mol, 0, 2, 1, 2)
        assert score == 16
        (score, empty) = ld0.score_line(mol, 0, 0, 5, 0)
        assert score == 7
        assert empty == 4
        # allowed empty check
        ld0.config["allowed_empty"] = 3
        (score, empty) = ld0.score_line(mol, 0, 0, 5, 0)
        assert score == 0
        assert empty == -4

    def test_get_blank_image(self) -> None:
        blank = ld.get_blank_image(200, 100, dtype=np.int32)
        assert blank.shape[0] == 200
        assert blank.shape[1] == 100

    def test_get_emphasized(self):
        pass

    def test_filter_by_quadrant(self):
        pass

    def test_overlay_lines(self, ld0:ld):
        config = {"min_len":10, "max_len":141,"allowed_empty":1}
        count = 0
        for i in range(len(ld0.molecules)):
            mol = ld0.molecules[i]
            filtered = ld0.filter_by_length(mol)
            if filtered is not None:
            #ol_filtered = ld.filter_by_overlapping(mol, filtered)
                lines_cnt = ld0.score_lines(mol, filtered)
                if lines_cnt > 0:
                    count += 1
        assert len(ld0.stat_df) == 3
        ld0.overlay_lines()
