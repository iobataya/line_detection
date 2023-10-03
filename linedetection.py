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

class LinearMolecule:
    """
    Numpy配列 [y座標配列, x座標配列, シグナル値配列]をインスタンスとして持つ。-> yx_values
    ラベルされたImage、xy配列、から生成する。メソッド間のやり取りには、内部の処理はこのyx_values配列を使う。
    MatPlotLibでimshowするときだけ展開する。

    ※座標配列 - 入出力、描画
    [y][x] : yx_pos, shape(N,2)
    [y,x]:   yx_pos.T, shape(2,N)

    ※ベクター配列 - 距離計算, 象限判別
    [[yx_pos_idx],[yx_pos_idx],[yx_pos]], shape(N,N,2)
    """
    def __init__(self,yx_values, mol_idx=1):
        self.mol_idx = mol_idx
        self.orig_yx_values = yx_values
        self.orig_left = self.orig_x().min()
        self.orig_top = self.orig_y().min()
        self.width = int(self.orig_x().max() - self.orig_left + 1)
        self.height = int(self.orig_y().max() - self.orig_top + 1)
        self.x = self.orig_x() - self.orig_left
        self.y = self.orig_y() - self.orig_top
        self.orig_yxT = np.array([self.orig_y(),self.orig_x()]).T
        self.yxT = np.array([self.y,self.x]).T

    def orig_x(self):
        return self.orig_yx_values[1]

    def orig_y(self):
        return self.orig_yx_values[0]

    def values(self):
        return self.orig_yx_values[2]

    def get_yx(self, pix_idx):
        return (self.y[pix_idx],self.x[pix_idx])

    def get_orig_yx(self,pix_idx):
        return (self.orig_y()[pix_idx], self.orig_x()[pix_idx])

    def get_vector(self, pix_idx0, pix_idx1):
        return (self.yxT[pix_idx0], self.yxT[pix_idx1])

    def get_displacement(self, pix_idx0, pix_idx1):
        return self.yxT[pix_idx0] - self.yxT[pix_idx1]

    def get_orig_vector(self,pix_idx0, pix_idx1):
        return (self.orig_yxT[pix_idx0], self.orig_yxT[pix_idx1])

    def count(self):
        return len(self.x)

    @staticmethod
    def create_from_labelled_image(labelled_image, mol_idx=1):
        l = np.where(labelled_image==mol_idx)
        # prepare nparray of y, x
        (y_ar, x_ar) = (l[0], l[1])
        values = np.zeros((len(x_ar),))
        for i in range(0,len(y_ar)):
            (y, x) = (y_ar[i], x_ar[i])
            values[i] = labelled_image[y][x]
        return LinearMolecule(np.array([y_ar,x_ar,values]),mol_idx=mol_idx)

    @staticmethod
    def create_from_yx_array(y_array,x_array,value=1.0,mol_idx=1):
        """ create a molecule from y,x list """
        val_array = np.full(len(y_array),value)
        mol = LinearMolecule(np.array([y_array,x_array,val_array]),mol_idx=mol_idx)
        return mol

    def get_displacement_matrix(self):
        """ Get a displacement matrix """
        # [y座標配列, x座標配列]から転置した[y,x]配列、yxTを使う。
        m1 = self.yxT.reshape((1,self.count(),2))
        m2 = self.yxT.reshape((self.count(),1,2))
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

    def get_blank_image(self):
        return np.zeros((self.height,self.width))

    @staticmethod
    def plot_image(image, aspect = 1.0):
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.imshow(image, cmap="afmhot", aspect=aspect)
        plt.show()

    def __str__(self):
        return f"ID:{self.mol_idx}:{self.count()} pixels, (x,y,w,h)=({self.orig_left},{self.orig_top},{self.width},{self.height})"

    def __repr__(self):
        return f"ID:{self.mol_idx}({self.count()}),({self.orig_left},{self.orig_top},{self.width},{self.height})"

    def get_line_mask(self,pix_id1,pix_id2):
        yx1 = self.get_yx(pix_id1)
        yx2 = self.get_yx(pix_id2)
        (x1, y1) = (int(yx1[1]), int(yx1[0]))
        (x2, y2) = (int(yx2[1]), int(yx2[0]))
        mask = self.get_blank_image()
        self.draw_line(x1,y1,x2,y2,mask)
        return mask

    def draw_line(self,x0,y0,x1,y1,np_area):
        """ Returns ndarray drawin the line by 1
        Args:
            (x0,y0,x1,y1) (int): starting and end point of line
            np_area (np.nd_array): nd_array to draw the line as 1
        Returns:
            (np.nd_array): area drawn
        """
        def _set_pixel(x,y,np_area,value=1):
            np_area[y][x] = 1

        def _horizontal_line(x0,y0,x1,y1,np_area):
            dx = abs(x1 - x0)
            step = 1
            if x0 > x1:
                step = -1
            (x, y) = (x0, y0)
            for i in range(dx):
                _set_pixel(x,y,np_area)
                x += step

            _set_pixel(x1,y1,np_area)
            return np_area

        def _vertical_line(x0,y0,x1,y1,np_area):
            dy = abs(y1 - y0)
            step = 1
            if y0 > y1:
                step = -1
            (x, y) = (x0, y0)
            for i in range(dy):
                _set_pixel(x,y,np_area)
                y += step
            _set_pixel(x1,y1,np_area)
            return np_area

        def _bresenham(x0,y0,x1,y1,np_area):
            """
            based on
            https://github.com/encukou/bresenham/blob/master/bresenham.py
            """
            (dx, dy) = (x1 - x0, y1 - y0)
            xsign = 1 if dx > 0 else -1
            ysign = 1 if dy > 0 else -1
            (dx, dy) = (abs(dx),abs(dy))
            if dx > dy:
                (xx, xy, yx, yy) = (xsign, 0, 0, ysign)
            else:
                (dx, dy) = (dy, dx)
                (xx, xy, yx, yy) = (0, ysign, xsign, 0)
            D = 2*dy - dx
            y = 0
            for x in range(dx + 1):
                _set_pixel(x0 + x*xx + y*yx, y0 + x*xy + y*yy, np_area)
                if D >= 0:
                    y += 1
                    D -= 2*dx
                D += 2*dy
            return np_area

        if x0 == x1 and y0 == y1:
            return np_area
        if x0 == x1 :  # vertical
            ret_ndarray = _vertical_line(x0,y0,x1,y1,np_area)
        elif y0 == y1:  # horizontal
            ret_ndarray = _horizontal_line(x0,y0,x1,y1,np_area)
        else:
            ret_ndarray = _bresenham(x0,y0,x1,y1,np_area)

        return ret_ndarray