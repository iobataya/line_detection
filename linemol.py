"""Functions for reading and writing data."""
import os
import logging
from datetime import datetime
import io
import struct
from pathlib import Path
import re
import matplotlib.pyplot as plt

import numpy as np
from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.main import round_trip_load as yaml_load, round_trip_dump as yaml_dump
import yaml
import pandas as pd

class Molecule:
    """ Molecule class for detection of linear parts
    Holds positions and pixels of molecule, providing methods to analyze by vectors
    Each [y, x] coordinate is converted into a vector from the origin.
    Original coordinate in a source image is stored as orig_variables.
    It is accessible by index of pixel (pix_idx) in the ndarray.

    Arguments:
        yx_positions(ndarray): [[y,], [x,]]
        mol_idx: index of this molecule
    """
    def __init__(self, yx_positions:np.ndarray, mol_idx:int=1, source_image:np.ndarray=None):
        self.mol_idx = mol_idx
        self.orig_yx = yx_positions
        (self.orig_left, self.orig_right) = (self.orig_x().min(), self.orig_x().max())
        (self.orig_top, self.orig_bottom) = (self.orig_y().min(), self.orig_y().max())
        self.width = self.orig_right - self.orig_left + 1
        self.height = self.orig_bottom - self.orig_top + 1
        if self.width > 127 or self.height > 127:
            raise IndexError(f"Grain is too large over 127.({self.width},{self.height})")
        (vec_x, vec_y) = (self.orig_x() - self.orig_left, self.orig_y() - self.orig_top)
        self.x = vec_x.astype(np.int32)
        self.y = vec_y.astype(np.int32)
        self.yx = np.array([self.y,self.x], dtype=np.int32)
        self.yxT = self.yx.T
        self.origin_mask = self.get_image()
        if source_image is not None:
            self.set_source_image(source_image)

    def orig_x(self):
        return self.orig_yx[1]

    def orig_y(self):
        return self.orig_yx[0]

    def get_yx(self, pix_idx:int):
        return (self.y[pix_idx],self.x[pix_idx])

    def get_orig_yx(self, pix_idx:int):
        return (self.orig_y()[pix_idx], self.orig_x()[pix_idx])

    def get_vector(self, pix_idx:int):
        return self.yxT[pix_idx]

    def get_displacement_vector(self, pix_idx0:int, pix_idx1:int):
        return self.yxT[pix_idx0] - self.yxT[pix_idx1]

    def count(self):
        """ Returns count of pixels """
        return len(self.orig_x())

    @staticmethod
    def create_from_labelled_image(labelled_image:np.ndarray, mol_idx:int=1):
        l = np.where(labelled_image==mol_idx)
        # prepare nparray of y, x from the resulting tuple
        (y_ar, x_ar) = (l[0], l[1])
        return Molecule(np.array([y_ar,x_ar], dtype=np.int32),mol_idx=mol_idx)

    @staticmethod
    def create_from_yx_array(y_array, x_array, mol_idx=1):
        """ create a molecule from y,x list """
        mol = Molecule(np.array([y_array,x_array], dtype=np.int32),mol_idx=mol_idx)
        return mol

    def get_displacement_matrix(self) -> np.ndarray:
        """ Get a displacement vector matrix """
        # Generate matrix from two identical transposed yxTs.
        m1 = self.yxT.reshape((1,self.count(),2))
        m2 = self.yxT.reshape((self.count(),1,2))
        # Convert to all possible displacement vectors
        mv = m1 - m2
        return mv

    def filter_by_quadrant(self) -> np.ndarray:
        """ Generate all possible pixel pairs and filter by quadrants
        If y is negative, the vector direct in 3 or 4 quadrant.
        """
        vectors = self.get_displacement_matrix()
        result = np.where(vectors < 0)  # find all negative x and y, yielding [[pix_idx],[pix_idx],[idxY_or_idxX],]
        negative_y = np.where(result[2]==0)  # When idxY_or_idxX == 0, the element is positin of Y.
        for pix_idx in negative_y[0]:
            idx0 = result[0][pix_idx]
            idx1 = result[1][pix_idx]
            vectors[idx0][idx1] = [0,0] # set vector in 3/4 quad to zero, supposed to be filtered by length
        return vectors

    def filter_by_length(self, min_length=2, max_length=40):
        """ Filter all possible pixel pairs by length between them
        """
        result_dict = {}
        # pick up displacement vector in 1+2 quadrants
        vectors = self.filter_by_quadrant()
        result_dict["source vectors"] = len(vectors[0])
        # filter by length
        msq = np.sum(np.square(vectors),axis=2) # square y and x, then sum them up
        md = np.sqrt(msq) # get root of it
        length_filtered_mask = np.where((md >= min_length) & (md <= max_length), 1, 0)
        pix_pairs = np.where(length_filtered_mask > 0)
        result_dict["length filtered vectors"] = len(pix_pairs[0])
        return (pix_pairs, result_dict)

    def get_image(self):
        """ Generates a binary image of this molecule"""
        image = self.get_blank_image()
        for i in range(0,len(self.x)):
            y = int(self.y[i])
            x = int(self.x[i])
            image[y][x] = 1
        return image

    def get_blank_image(self, dtype=np.float64):
        """ Generates a blank image ndarray for this molecule """
        return np.zeros((self.height,self.width),dtype=dtype)

    def set_source_image(self, src_img):
        """ Set a source image of this molecule cropped from src_img"""
        self.src_img = src_img[self.orig_top:self.orig_bottom + 1, self.orig_left:self.orig_right + 1]

    @staticmethod
    def plot_image(image, aspect = 1.0):
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.imshow(image, cmap="afmhot", aspect=aspect)
        plt.show()

    def __str__(self):
        return f"ID:{self.mol_idx}:{self.count()} pixels, (x,y,w,h)=({self.orig_left},{self.orig_top},{self.width},{self.height})"

    def __repr__(self):
        return f"ID:{self.mol_idx}({self.count()}),({self.orig_left},{self.orig_top},{self.width},{self.height})"

    def get_pix_id_pair(self, vector_ar, vec_idx):
        (pix_id1,pix_id2) = (vector_ar[0],vector_ar[1])
        (id1, id2) = (pix_id1[vec_idx], pix_id2[vec_idx])
        return (id1, id2)

    def score_line(self, vector_ar, vec_idx, allowed_empty=0) -> np.float64:
        """ Scores by height signal along the line

        Returns:
            float: sum of height signal or -1 when empty pixels exceeds allowed count
        """
        (id1, id2) = self.get_pix_id_pair(vector_ar, vec_idx)

        line_mask = self.get_line_mask(id1, id2)
        line_pixels = line_mask.sum()
        masked = self.origin_mask * line_mask
        overlapped_pixels = masked.sum()
        empty = line_pixels - overlapped_pixels
        if empty <= allowed_empty:
            score_img = line_mask * self.src_img
            score = score_img.sum()
            del masked, score_img
            return score
        else:
            return -1

    def get_emphasized(self, mask, factor=2.0, src_img=None):
        if src_img is None:
            src_img = self.src_img()
        return src_img + (mask * src_img * (factor - 1.0))

    def get_line_mask(self, pix_id1, pix_id2):
        yx1 = self.get_yx(pix_id1)
        yx2 = self.get_yx(pix_id2)
        (x1, y1) = (yx1[1], yx1[0])
        (x2, y2) = (yx2[1], yx2[0])
        mask = self.get_blank_image(dtype=np.int32)
        self.draw_line(x1, y1, x2, y2, mask)
        return mask

    def draw_line(self, x1, y1, x2, y2, np_area):
        """ Returns ndarray drawin the line by 1
        Args:
            (x1,y1,x2,y2) (int): starting and end point of line
            np_area (np.nd_array): nd_array to draw the line as 1
        Returns:
            (np.nd_array): area drawn
        """
        def _set_pixel(x, y, np_area, value=1):
            np_area[y][x] = value

        def _vertical_line(y1, y2, x, np_area):
            (i, j) = (min(y1, y2), max(y1, y2) + 1)
            for y in range(i, j):
                _set_pixel(x, y, np_area)
            return np_area

        def _horizontal_line(x1, x2, y, np_area):
            (i, j) = (min(x1, x2), max(x1, x2) + 1)
            for x in range(i, j):
                _set_pixel(x, y, np_area)
            return np_area

        def _bresenham(x1, y1, x2, y2, np_area):
            """modified from https://github.com/encukou/bresenham/blob/master/bresenham.py """
            (dx, dy) = (x2 - x1, y2 - y1)
            xsign = 1 if dx > 0 else -1
            ysign = 1 if dy > 0 else -1
            (dx, dy) = (abs(dx), abs(dy))
            if dx > dy:
                (xx, xy, yx, yy) = (xsign, 0, 0, ysign)
            else:
                (dx, dy) = (dy, dx)
                (xx, xy, yx, yy) = (0, ysign, xsign, 0)
            D = 2*dy - dx
            y = 0
            for x in range(dx + 1):
                _set_pixel(x1 + x * xx + y * yx, y1 + x * xy + y * yy, np_area)
                if D >= 0:
                    y += 1
                    D -= 2*dx
                D += 2*dy
            return np_area

        if x1 == x2 and y1 == y2: # a point
            _set_pixel(x1, y1, np_area)
            return np_area

        if x1 == x2 :  # vertical line
            ret_ndarray = _vertical_line(y1, y2, x1, np_area)
        elif y1 == y2:  # horizontal line
            ret_ndarray = _horizontal_line(x1, x2, y1, np_area)
        else:
            ret_ndarray = _bresenham(x1, y1, x2, y2, np_area)
        return ret_ndarray


class LineDetection:
    """ Line detection of molecules

    linedet_config:
        min_len: minimum length of a pixel pair
        max_len: maximum length of a pixel pair
        allowed_empty: allowed pixel number of empty
    """
    def __init__(self, molecule_list, source_img, **linedet_config):
        if len(molecule_list)==0:
            raise ValueError()
        if not isinstance(molecule_list[0], Molecule):
            raise ValueError()
        self.molecules = molecule_list
        self.source_img = source_img
        self.config = linedet_config
        self.result_df = None
        self.statistics = {}
        self.statistics["molecule count"] =  self.mol_count()
        if linedet_config == None:
            self.config["min_len"] = 10
            self.config["max_len"] = 100
            self.config["allowed_empty"] = 0
        else:
            self.config = linedet_config
        self.result_df = pd.DataFrame(columns=[
            "mol_idx","score","pix1","pix2","angle","overlapped"
        ])
        self.stat_df = pd.DataFrame(columns=[
            "mol idx", "pixels", "vector total", "length filtered", "empty filtered"])



    def mol_count(self):
        return len(self.molecules)

    def get_result_df(self):
        for mol in self.molecules:
            self.add_result_of(mol)
        return self.result_df

    def add_result_of(self, mol:Molecule):
        (vector_ar, stat) = mol.filter_by_length(
            min_length=self.config["min_len"],
            max_length=self.config["max_len"])
        if "length filter" not in self.statistics:
            self.statistics["length filter"] = {}
        self.statistics["length filter"][mol.mol_idx] = stat

        score_list = []
        pix_id1s = []
        pix_id2s = []
        angles = []
        vec_count = len(vector_ar[0])

        for i in range(vec_count):
            score = mol.score_line(vector_ar, i, allowed_empty=self.config["allowed_empty"])
            if score == None:
                continue
            score_list.append(score)
            (pid1, pid2) = mol.get_pix_id_pair(vector_ar, i)
            pix_id1s.append(pid1)
            pix_id2s.append(pid2)
            (dy, dx) = mol.get_displacement_vector(pid1, pid2)
            if dy < 0:
                (dy, dx) = (-dy, -dx)
            angle = np.degrees(np.arctan2(dy,dx))
            angles.append(angle)
        if "scoring" not in self.statistics:
            self.statistics["empty filtered"] = {}

        # TODO: Make statistics also DataFrame


        data = {"score":score_list,"pix1":pix_id1s,"pix2":pix_id2s,"angle":angles, "overlapped":False}
        df = pd.DataFrame(data)
        df.sort_values("score",ascending=False, inplace=True) # sorting score in place
        df.reset_index(inplace=True, drop=True) # re-indexing in place
        df["r_angle"] = df["angle"] - (df["angle"] % 5.0)
        self.result_df.append(df)

    #TODO
    # score line - it uses many of Molecule instance. score in the class.
    #

    def get_angle_dist(self, mol_idx):
        pass

    def run_line_detection(self, mol_idx):
        pass
