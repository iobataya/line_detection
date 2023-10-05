"""Functions for reading and writing data."""
import os
import logging
from datetime import datetime
import io
import struct
from pathlib import Path
import re

import numpy as np
from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.main import round_trip_load as yaml_load, round_trip_dump as yaml_dump
import yaml

class LineDetection:
    pass


class Molecule:
    """
    Holds pixels by [[y positions],[x positions]], dtype=np.int32
    """
    def __init__(self,yx_positions, mol_idx=1):
        self.mol_idx = mol_idx
        self.orig_yx = yx_positions
        (self.orig_left, self.orig_right) = (self.orig_x().min(), self.orig_x().max())
        (self.orig_top, self.orig_bottom) = (self.orig_y().min(), self.orig_y().max())
        self.width = self.orig_right - self.orig_left + 1
        self.height = self.orig_bottom - self.orig_top + 1
        if self.width > 127 or self.height > 127:
            raise IndexError(f"Grain is too large over 127.({self.width},{self.height})")
        (vec_x, vec_y) = (self.orig_x() - self.orig_left, self.orig_y() - self.orig_top)
        self.x = vec_x.astype(np.byte)
        self.y = vec_y.astype(np.byte)
        self.yxB = np.array([self.y,self.x], dtype=np.byte)
        self.yxBT = self.yxB.T

    def orig_x(self):
        return self.orig_yx[1]

    def orig_y(self):
        return self.orig_yx[0]

    def get_yx(self, pix_idx):
        return (self.y[pix_idx],self.x[pix_idx])

    def get_orig_yx(self,pix_idx):
        return (self.orig_y()[pix_idx], self.orig_x()[pix_idx])

    def get_vector(self, pix_idx):
        return self.yxBT[pix_idx]

    def get_displacement_vector(self, pix_idx0, pix_idx1):
        return self.yxBT[pix_idx0] - self.yxBT[pix_idx1]

    def count(self):
        """ Returns count of pixels """
        return len(self.orig_x())

    @staticmethod
    def create_from_labelled_image(labelled_image, mol_idx=1):
        l = np.where(labelled_image==mol_idx)
        # prepare nparray of y, x from the resulting tuple
        (y_ar, x_ar) = (l[0], l[1])
        return Molecule(np.array([y_ar,x_ar], dtype=np.int32),mol_idx=mol_idx)

    @staticmethod
    def create_from_yx_array(y_array,x_array,mol_idx=1):
        """ create a molecule from y,x list """
        mol = Molecule(np.array([y_array,x_array], dtype=np.int32),mol_idx=mol_idx)
        return mol

    def get_displacement_matrix(self):
        """ Get a displacement matrix """
        # [y座標配列, x座標配列]から転置した[y,x]配列、yxTを使う。
        m1 = self.yxBT.reshape((1,self.count(),2))
        m2 = self.yxBT.reshape((self.count(),1,2))
        mv = m1 - m2  # 2点間の全ベクトル
        return mv

    def filter_by_quadrant(self):
        vectors = self.get_displacement_matrix()
        result = np.where(vectors < 0) # 負の値の要素をすべて探す。-> 返り値は[[pix_idx],[pix_idx],[idxY or idxX],]
        negative_y = np.where(result[2]==0) # idxYがゼロのときその要素はY座標。-> 返り値は[[idxY],]
        for pix_idx in negative_y[0]:
            idx0 = result[0][pix_idx]
            idx1 = result[1][pix_idx]
            vectors[idx0][idx1] = [0,0]
        return vectors

    def filter_by_length(self, min_length=0, filter_quad=True):
        if filter_quad:
            vectors = self.filter_by_quadrant()
        else:
            vectors = self.get_displacement_matrix()
        msq = np.sum(np.square(vectors),axis=2) # y,xを二乗してsum -> 長さ2乗
        md = np.sqrt(msq) # 平方根を取って距離に。
        length_filtered_mask = np.where(md>=min_length, 1, 0)  # 最小の長さを超えていれば距離, if not 0
        pix_pairs = np.where(length_filtered_mask > 0)
        return pix_pairs,length_filtered_mask

    def get_image(self):
        """ get binary image of this molecule"""
        image = self.get_blank_image()
        for i in range(0,len(self.x)):
            y = int(self.y[i])
            x = int(self.x[i])
            image[y][x] = 1
        return image

    def get_blank_image(self, dtype=np.float64):
        return np.zeros((self.height,self.width),dtype=dtype)

    @staticmethod
    def plot_image(image, aspect = 1.0):
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.imshow(image, cmap="afmhot", aspect=aspect)
        plt.show()

    def __str__(self):
        return f"ID:{self.mol_idx}:{self.count()} pixels, (x,y,w,h)=({self.orig_left},{self.orig_top},{self.width},{self.height})"

    def __repr__(self):
        return f"ID:{self.mol_idx}({self.count()}),({self.orig_left},{self.orig_top},{self.width},{self.height})"

    def get_line_mask(self, pix_id1, pix_id2):
        yx1 = self.get_yx(pix_id1)
        yx2 = self.get_yx(pix_id2)
        (x1, y1) = (yx1[1], yx1[0])
        (x2, y2) = (yx2[1], yx2[0])
        mask = self.get_blank_image(dtype=np.byte)
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
            """ https://github.com/encukou/bresenham/blob/master/bresenham.py """
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