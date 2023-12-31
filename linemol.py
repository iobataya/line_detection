"""Classes for detection of linear part of molecule"""
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.main import round_trip_load as yaml_load, round_trip_dump as yaml_dump
import yaml
import pandas as pd
import math
import tqdm

class Molecule:
    """ Molecule class holds positions and pixels of molecule,
    providing methods to analyze by vectors
    Each [y, x] coordinate is converted into a vector from the origin.
    Original coordinate in a source image is stored as src_variables.
    It is accessible by index of pixel (pix_idx) in the ndarray.

    Arguments:
        yx_positions(ndarray): [[y,], [x,]]
        mol_idx: index of this molecule
    """
    def __init__(self, yx_positions:np.ndarray, mol_idx:int=1, source_image:np.ndarray=None):
        (XAXIS, YAXIS) = (1, 0)
        self.mol_idx = mol_idx
        self.src_yx = yx_positions
        self.src_left = self.src_yx[XAXIS].min()
        self.src_top = self.src_yx[YAXIS].min()
        src_right = self.src_yx[XAXIS].max()
        src_bottom = self.src_yx[YAXIS].max()
        self.width = src_right - self.src_left + 1
        self.height = src_bottom - self.src_top + 1
        if self.width > 255 or self.height > 255:
            raise IndexError(f"Grain is too large over 255.({self.width},{self.height})")

        vec_x = self.src_yx[XAXIS] - self.src_left
        vec_y = self.src_yx[YAXIS] - self.src_top
        (self.x, self.y) = (vec_x.astype(np.int32), vec_y.astype(np.int32))
        self.yxT = np.array([self.y, self.x], dtype=np.int32).T

        self.mask = self.get_mask()
        if source_image is not None:
            self.set_source_image(source_image)

    def get_yx(self, pix_idx:int):
        return (self.y[pix_idx],self.x[pix_idx])

    def get_xy12(self, px_idx1:int, px_idx2:int):
        (y1, x1) = self.get_yx(px_idx1)
        (y2, x2) = self.get_yx(px_idx2)
        return (x1, y1, x2, y2)

    def get_vector(self, pix_idx:int):
        return self.yxT[pix_idx]

    def get_displacement_vector(self, pix_idx0:int, pix_idx1:int):
        dvec = self.yxT[pix_idx0] - self.yxT[pix_idx1]
        return dvec

    def count(self):
        """ Returns count of pixels """
        return len(self.x)

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

    @staticmethod
    def create_all_from_labelled_image(labelled_image:np.ndarray, source_image:np.ndarray) -> list:
        """ create molecule list from labelled image """
        count = labelled_image.max()
        mol_list = []
        for i in range(1,count + 1):
            mol = Molecule.create_from_labelled_image(
                labelled_image,
                mol_idx=i,
            )
            mol.set_source_image(source_image)
            mol_list.append(mol)
        return mol_list

    def get_displacement_matrix(self) -> np.ndarray:
        """ Get a displacement vector matrix """
        # Generate matrix from two identical transposed yxTs.
        m1 = self.yxT.reshape((1,self.count(),2))
        m2 = self.yxT.reshape((self.count(),1,2))
        # Convert to all possible displacement vectors
        dvecs_matrix = m1 - m2
        return dvecs_matrix

    def get_displacement_vectors(self) -> np.ndarray:
        """ All displacement vectors (pair of pixel) [idx0, idx1]
          directing toward in 1/2 quadrant

        Returns:
            ndarray of [[idx0, idx1, y, x],]
        """
        matrix = self.get_displacement_matrix()
        (count,_,_) = matrix.shape
        #combinations = count * (count - 1) / 2
        ret_array = []
        for row in range(0, count):
            for col in range(row + 1, count):
                if matrix[row][col][0] >= 0:  # vector in 1/2 quadrant
                    (idx0, idx1) = (row, col)
                else:
                    (idx0, idx1) = (col, row)  # vector from 3/4 quadrant to be swapped
                dy = matrix[idx0][idx1][0]
                dx = matrix[idx0][idx1][1]
                ret_array.append([idx0, idx1, dy, dx])
        return np.array(ret_array, dtype=np.int32)

    def get_cross_product_of_line(self, px_id1, px_id2, px_id3, px_id4):
        (y1, x1) = self.get_yx(px_id1)
        (y2, x2) = self.get_yx(px_id2)
        (y3, x3) = self.get_yx(px_id3)
        (y4, x4) = self.get_yx(px_id4)
        (dx1, dy1) = (x2 - x1, y2 - y1)
        (dx2, dy2) = (x4 - x3, y4 - y3)
        return dx1 * dy2 - dx2 * dy1

    def rect_overlaps(self, px_id1, px_id2, px_id3, px_id4):
        (y1, x1) = self.get_yx(px_id1)
        (y2, x2) = self.get_yx(px_id2)
        (y3, x3) = self.get_yx(px_id3)
        (y4, x4) = self.get_yx(px_id4)
        return (max(x1,x3) <= min(x2, x4)) and (max(y1, y3) < min(y2,y4))


    def get_blank(self, dtype=np.int32):
        return np.zeros((self.height, self.width), dtype=dtype)

    def get_mask(self):
        """ Generates a binary image of this molecule"""
        image = self.get_blank()
        for i in range(0, len(self.x)):
            y = int(self.y[i])
            x = int(self.x[i])
            image[y][x] = 1
        return image

    def get_source_image(self):
        mask = self.get_mask()
        return mask * self.src_img

    def set_source_image(self, src_img):
        """ Set a source image of this molecule cropped from src_img"""
        cropped = src_img[self.src_top:self.src_top + self.height,
                                 self.src_left:self.src_left + self.width]
        self.src_img = cropped

    def __str__(self):
        return f"Molecule:{self.mol_idx}:{self.count()} pixels, {self.width} x {self.height}"

    def __repr__(self):
        return f"{self.mol_idx}({self.count()} px) {self.width} x {self.height}"

class Line:
    """ Line class
        Holds displacement line starting at [0, 0]  as yxT.
    """
    (XAXIS,YAXIS) = (1, 0)
    def __init__(self, dx, dy, offset_x=0, offset_y=0):
        (self.dx, self.dy) = (dx, dy)
        self.yxT = Line.draw_line(0, 0, dx, dy)
        self.offset_yx = np.array([offset_y, offset_x], dtype=np.int32)

    @staticmethod
    def create_from_yx(yx1:np.ndarray, yx2:np.ndarray):
        dyx = yx1 - yx2
        return Line(dyx[Line.XAXIS],
                    dyx[Line.YAXIS],
                    offset_x=yx2[Line.XAXIS],
                    offset_y=yx2[Line.YAXIS])

    @staticmethod
    def create_from_pos(x1, y1, x2, y2):
        (dx, dy) = (x1 - x2, y1 - y2)
        line = Line(dx, dy, offset_x=x2, offset_y=y2)
        return line

    @staticmethod
    def create_from_array(yxT:np.ndarray, offset_x=0, offset_y=0):
        yx = yxT.T
        (y_ar, x_ar) = (yx[0], yx[1])
        line = Line.create_from_pos(
            x_ar.min(),
            y_ar.min(),
            x_ar.max(),
            y_ar.max()
        )
        return line

    @staticmethod
    def create_from_score_df(score_df, mol_idx:int, line_idx:int):
        row = score_df.loc[(score_df["mol_idx"]==mol_idx) & (score_df["line_idx"]==line_idx)]
        if len(row) > 0:
            return Line.create_from_pos(row["x1"], row["y1"], row["x2"], row["y2"])
        return None

    def shifted_yxT(self):
        return self.yxT + self.offset_yx

    def get_x1y1x2y2(self):
        (x2, y2) = (self.offset_yx[Line.XAXIS], self.offset_yx[Line.YAXIS])
        (x1, y1) = (x2 + self.dx, y2 + self.dy)
        return (x1, y1, x2, y2)

    def count(self):
        return len(self.yxT)

    def get_mask(self, height, width):
        """ Get mask of this line. dtype is bool """
        blank = np.zeros((height, width), dtype=bool)
        yxT_shifted = self.yxT + self.offset_yx
        for i in range(len(yxT_shifted)):
            (y, x) = (yxT_shifted[i][0], yxT_shifted[i][1])
            if y < 0 or x < 0:
                continue
            blank[y][x] = True
        return blank

    def get_mask_at(self, height, width, origin_x, origin_y):
        """ Get mask of displacement line whose origin is at origin_x, origin_y (x2, y2)
        This mask is binary, (dtype=bool)
        """
        blank = np.zeros((height, width), dtype=bool)
        offset_yx = np.array([origin_y, origin_x], dtype=np.int32)
        yxT_shifted = self.yxT + offset_yx
        for i in range(len(yxT_shifted)):
            (y, x) = (yxT_shifted[i][0], yxT_shifted[i][1])
            if y < 0 or x < 0:
                continue
            blank[y][x] = True
        return blank

    def __str__(self):
        (x1, y1, x2, y2) = self.get_x1y1x2y2()
        return f"Line ({x1},{y1})-({x2},{y2}) displacement:({self.dx},{self.dy})"

    @staticmethod
    def draw_line(x1, y1, x2, y2) -> np.ndarray:
        """ Returns binary image of line in a defined size
        Args:
            (x1,y1,x2,y2) (int): starting and end point of line
        Returns:
            (np.nd_array): [[dy,], [dx,]]
        """
        (lx, ly) = ([], [])

        def _set_pixel(x, y):
            lx.append(x)
            ly.append(y)

        def _vertical_line(y1, y2, x):
            (i, j) = (min(y1, y2), max(y1, y2) + 1)
            for y in range(i, j):
                _set_pixel(x, y)
            return np.array([ly, lx])

        def _horizontal_line(x1, x2, y):
            (i, j) = (min(x1, x2), max(x1, x2) + 1)
            for x in range(i, j):
                _set_pixel(x, y)
            return np.array([ly, lx])

        def _bresenham(x1, y1, x2, y2):
            # modified from https://github.com/encukou/bresenham/blob/master/bresenham.py
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
                _set_pixel(x1 + x * xx + y * yx, y1 + x * xy + y * yy)
                if D >= 0:
                    y += 1
                    D -= 2*dx
                D += 2*dy
            return np.array([ly, lx])

        if x1 == x2 and y1 == y2: # a point
            _set_pixel(x1, y1)
            return np.array([[y1,],[x1,]])

        if x1 == x2 :  # vertical line
            ret_ndarray = _vertical_line(y1, y2, x1)
        elif y1 == y2:  # horizontal line
            ret_ndarray = _horizontal_line(x1, x2, y1)
        else:
            ret_ndarray = _bresenham(x1, y1, x2, y2)
        return ret_ndarray.T

class LineDetection:
    """ Line detection of molecules

    Arguments:
        labelled_img(np.ndarray): image of labelled02 from TopoStats find_grains result
        source_img(np.ndarray): source image of signal (ex. z-height)

    linedet_config:
        min_len: minimum length of a pixel pair (5)
        max_len: maximum length of a pixel pair (50)
        max_pix: maximum pixesl count (500)
        allowed_empty: allowed pixel number of empty (1)
        record_neg_empty: record score even if empty count>allowed empty (False)
        score_cutoff: cutoff limit of normalized score(1.0)
    """
    def __init__(self, labelled_img:np.ndarray, source_img:np.ndarray, **linedet_config):
        self.labelled_img = labelled_img
        self.source_img = source_img
        if labelled_img.shape != source_img.shape:
            raise TypeError("Shape of labelled_img and source_image are different.")
        self.molecules = Molecule.create_all_from_labelled_image(labelled_img, source_img)

        self.height = labelled_img.shape[0]
        self.width = labelled_img.shape[1]
        self.line_cache = {}
        self.line_cache_hit = (0,0)
        self.mask_cache = {}

        self.config = linedet_config
        self.config.setdefault("min_len", 5)
        self.config.setdefault("max_len", 50)
        self.config.setdefault("max_pix", 500)
        self.config.setdefault("allowed_empty", 1)
        self.config.setdefault("record_neg_empty", False)
        self.config.setdefault("score_cutoff", 1.0)
        self.config.setdefault("use_cache", False)

        self.score_columns = ["mol_idx","score","empty","line_idx","x1","y1","x2","y2","length","angle"]
        self.score_df = pd.DataFrame(columns=self.score_columns)
        self.stat_columns = ["mol_idx", "pixels", "total_vecs", "x0","y0","len_filtered","total_lines", "min_len", "max_len", "max_pix","score_cutoff"]
        self.stat_df = pd.DataFrame(columns=self.stat_columns)


    def filter_by_length(self, mol:Molecule) -> np.ndarray:
        """ Filter all possible pixel pairs by length between them

            Returns:
                Array of pair of pixel ID in the molecule and displacement vector as dy, dx
                [idx1, idx2, dy, dx]
                If none of returning vectors, returns None
        """
        if mol.count() > self.config["max_pix"]:
            comb = mol.count() * (mol.count() - 1) // 2
            self._add_len_filter_stat(mol, comb, 0)
            return None

        # calculate length using dy, dx
        min_len = self.config["min_len"]
        max_len = self.config["max_len"]
        dvectors = mol.get_displacement_vectors()  # all vectors
        dyx = dvectors.T[2:4]  # ndarray of [[dy,], [dx,]]
        dyx_sq = np.square(dyx).sum(axis=0)
        lengths = np.sqrt(dyx_sq)
        filtered_idx = np.where((lengths>=min_len) & (lengths<=max_len))[0]
        if len(filtered_idx) == 0:
            return None
        filtered = dvectors.take(filtered_idx, axis=0)
        self._add_len_filter_stat(mol, len(dyx[0]), len(filtered))

        return filtered

    def _add_len_filter_stat(self, mol:Molecule, total_vecs:int, len_filtered:int):
        stat = pd.DataFrame([{
            "mol_idx":mol.mol_idx,
            "pixels":mol.count(),
            "min_len":self.config["min_len"],
            "max_len":self.config["max_len"],
            "max_pix":self.config["max_pix"],
            "total_vecs":total_vecs,
            "x0":mol.src_left,
            "y0":mol.src_top,
            "len_filtered":len_filtered
        }])
        self.stat_df= pd.concat([self.stat_df,stat], axis=0, ignore_index=True)



    def score_lines(self, mol:Molecule, filtered_vecs):
        """ Score all lines of the molecule

        Arguments:
            mol(Molecule): molecule
            vectors(ndarray): array of pair of pixel index of the molecule
        """
        if mol.count() > self.config["max_pix"]:
            return 0
        if filtered_vecs.shape[0] == 0 or filtered_vecs.shape[1] == 0:
            return 0
        # ["mol_idx","score","empty","x1","y1","x2","y2","length","angle"]
        results = {key: [] for key in self.score_columns}

        vecs = filtered_vecs  # vecs[0]: idx1, vecs[1]: idx2
        line_count = len(vecs.T[0])
        line_idx = 0
        for i in range(line_count):
            yx1 = vecs[i][0]
            yx2 = vecs[i][1]
            (dy, dx)  = (vecs[i][2], vecs[i][3])
            (x1, y1) = (mol.yxT[yx1][1], mol.yxT[yx1][0])
            (x2, y2) = (mol.yxT[yx2][1], mol.yxT[yx2][0])
            (score, empty) = self.score_line(mol, x1, y1, x2, y2)
            if self.config["record_neg_empty"] == False and empty < 0:
                continue  # skip record negative empty pixel count

            length = math.sqrt(dx*dx + dy*dy)
            angle = math.degrees(math.atan2(dy, dx))

            results["mol_idx"].append(mol.mol_idx)
            results["score"].append(score)
            results["empty"].append(empty)
            results["line_idx"].append(line_idx)
            results["x1"].append(x1)
            results["y1"].append(y1)
            results["x2"].append(x2)
            results["y2"].append(y2)
            results["length"].append(length)
            results["angle"].append(angle)
            line_idx += 1
        df = pd.DataFrame.from_dict(results)  # the fastest way to add rows
        # normalized score for each molecule
        df['norm_score'] = df['score'].apply(lambda x: (x - df['score'].mean()) / df['score'].std())
        df.sort_values("norm_score", ascending=False, inplace=True)
        cutoff_df = df.loc[df["norm_score"] > self.config["score_cutoff"]]
        self.score_df = pd.concat([self.score_df,cutoff_df], axis=0, ignore_index=True)
        return line_count

    def score_line(self, mol:Molecule, x1, y1, x2, y2) -> np.float64:
        """ Scores by height signal along the line

        Arguments:
            mol(Molecule): molecule
            vectors(ndarray): [y,x]

        Returns:
            float:  sum of height signal
            int:    empty pixels, negative value if not allowed.
        """
        allowed_empty = self.config["allowed_empty"]
        line_mask = self._get_line_mask_cache(mol.height,mol.width, x1, y1, x2, y2)
        line_pixels = line_mask.sum()
        masked = mol.mask * line_mask
        overlapped_pixels = masked.sum()
        empty = line_pixels - overlapped_pixels
        if empty <= allowed_empty:
            score_img = masked * mol.src_img
            score = score_img.sum()
            del masked, score_img
            return (score, empty)
        else:
            return (0, -empty)

    def prep_overlap_elim(self):
        score_cols = self.score_df.columns
        if not 'overlapped' in score_cols:
            self.score_df["overlapped"] = np.zeros(len(self.score_df), dtype=bool)
        if not 'overlap_by' in score_cols:
            self.score_df["overlap_by"] = np.zeros(len(self.score_df), dtype=str)
        stat_cols = self.stat_df.columns
        if not 'overlap_checked' in stat_cols:
            self.stat_df["overlap_checked"] = np.zeros(len(self.stat_df), dtype=bool)

    def eliminate_overlap(self, mol_idx:int, diff_pix = 2):
        if self.stat_df.loc[self.stat_df["mol_idx"]==mol_idx, "overlap_checked"].any():  # already checked
            return
        rows = self.score_df.loc[self.score_df["mol_idx"]==mol_idx]
        line_idx = rows["line_idx"].to_numpy()
        line_cnt = len(line_idx)
        if line_cnt == 0:
            self.stat_df.loc[self.stat_df["mol_idx"]==mol_idx, "overlap_checked"] = True
            return
        (x1s, y1s, x2s, y2s) = (rows["x1"].to_numpy(), rows["y1"].to_numpy(), rows["x2"].to_numpy(), rows["y2"].to_numpy())
        (xs, ys) = (np.concatenate([x1s, x2s]), np.concatenate([y1s, y2s]))
        (height, width) = (ys.max() + 1, xs.max() + 1)
        covered = set()  # pool of overlapped line indecies
        for i in range(line_cnt):
            if line_idx[i] in covered:
                continue
            (x1, y1, x2, y2) = (x1s[i],y1s[i],x2s[i],y2s[i])
            line_i = self._get_line_cache(x1, y1, x2, y2)
            count_i = len(line_i.yxT)
            for j in range(i+1, line_cnt):
                if line_idx[j] in covered:
                    continue
                (x1, y1, x2, y2) = (x1s[j],y1s[j],x2s[j],y2s[j])
                line_j = self._get_line_cache(x1, y1, x2, y2)
                count_j = len(line_j.yxT)
                shorter = min(count_i, count_j)
                (mask_i, mask_j) = (line_i.get_mask(height, width), line_j.get_mask(height,width))
                self._get_line_mask_cache(height, width, x1, y1, x2, y2)

                if (mask_i & mask_j).sum() >= (shorter - diff_pix):
                    if count_i > count_j:  # line_j is covered with line_i
                        covered_idx = line_idx[j]
                        covers_str = str(line_i)
                    else:
                        covered_idx = line_idx[i]
                        covers_str = str(line_j)
                    self.score_df.loc[
                        (self.score_df["mol_idx"]==mol_idx) & (self.score_df["line_idx"]==covered_idx),
                        ["overlapped","overlap_by"]
                        ] = [True, covers_str]
                    covered.add(covered_idx)
        self.stat_df.loc[self.stat_df["mol_idx"]==mol_idx, "overlap_checked"] = True

    def get_lines_mask(self, mol_idx, line_idx_list, height, width):
        """ Get mask of specified lines in molecule """
        mask = np.zeros((height,width), dtype=bool)
        for line_idx in line_idx_list:
            row = self.score_df.loc[(self.score_df["mol_idx"]==mol_idx) & (self.score_df["line_idx"]==line_idx),["x1","y1","x2","y2"]].values[0]
            line = Line.create_from_pos(row[0],row[1],row[2],row[3])
            mask = mask + line.get_mask(height,width)
        return mask

    def save_score_pkl(self, filename, with_config=False):
        """ Save score DataFrame
        Arguments:
            filename(str): filename to save
                            extension of .pkl or compressed types like .pkl.gz are acceptable
            with_config(bool): if True, filtering configs are added
        """
        path = Path(filename)
        if not with_config:
            savename = str(path)
        else:
            postfix = f'_{self.config["min_len"]}_{self.config["max_len"]}_{self.config["allowed_empty"]}'
            savename = str(path.stem + postfix + ".pkl")
        self.score_df.to_pickle(savename)

    def overlay_lines(self, num_lines=1, factor=1.2):
        """ Image of all lines of highest score of a molecule """
        # TODO: Return only line image !
        # TODO: Split into overlay_line and overlay_lines
        src = self.source_img
        overlay = LineDetection.get_blank_image(self.height,self.width)
        if len(self.score_df)==0:
            raise ValueError("No scores estimated.")
        max_height = self.source_img.max()
        mol_count = 0
        for mol in self.molecules:
            mol_idx = mol.mol_idx
            df = self.score_df.loc[self.score_df["mol_idx"]==mol_idx].sort_values("norm_score", ascending=False)
            if len(df[:num_lines]) == 0:
                continue
            mol_pos = [mol.src_left, mol.src_top, mol.src_left, mol.src_top]  # positions of Molecule
            lines = LineDetection.get_blank_image(self.height,self.width)
            line_count = 0
            for i in range(len(df[:num_lines])):
                p = df.iloc[i][4:8] + mol_pos
                l = Line.create_from_pos(p["x1"], p["y1"], p["x2"], p["y2"])
                line = l.get_mask(self.height,self.width)
                line.astype(dtype=np.float64)
                lines = lines + line
                line_count += 1
            overlay = overlay + (lines / line_count)
            mol_count += 1
        return (overlay * max_height * factor / mol_count) + src

    def _get_line_mask_cache(self, height, width, x1, y1, x2, y2):
        """ Get line mask binary image from in cache. If none, generate it.
        """
        (hit, total) = self.line_cache_hit
        total += 1
        if self.config["use_cache"] and (x1, y1, x2, y2) in self.line_cache:
            line = self.line_cache[(x1, y1, x2, y2)]
            hit += 1
            self.line_cache_hit = (hit, total)
            return line.get_mask_at(height, width,origin_x=x2, origin_y=y2)
        else:
            line = Line.create_from_pos(x1, y1, x2, y2)
            self.line_cache[(x1, y1, x2, y2)] = line
            self.line_cache_hit = (hit, total)
            return line.get_mask(height, width)

    def _get_line_cache(self, x1, y1, x2, y2):
        """ Get line instance from in cache. If none, generate it.
        """
        (hit, total) = self.line_cache_hit
        total += 1
        if self.config["use_cache"] and (x1, y1, x2, y2) in self.line_cache:
            line = self.line_cache[(x1, y1, x2, y2)]
            hit += 1
            self.line_cache_hit = (hit, total)
            return line
        else:
            line = Line.create_from_pos(x1, y1, x2, y2)
            self.line_cache[(x1, y1, x2, y2)] = line
            self.line_cache_hit = (hit, total)
            return line


    def __str__(self):
        mol_count = len(self.molecules)
        count = f"Line detection object: {mol_count} molecules."
        score_rows = len(self.score_df)
        row =  f"Calculated score rows: {score_rows}"
        return count + row

# region static methods

    @staticmethod
    def _matrix_table(matrix, row_title=[], col_title=[]) -> str:
        """ Returns matrix table as readable format in str """
        (r, c, _) = matrix.shape
        if len(row_title) != r:
            row_title = list(map(str, [*range(r)]))
        else:
            row_title = [str(rt) for rt in row_title]
        if len(col_title) != c:
            col_title = list(map(str, [*range(c)]))
        else:
            col_title = [str(ct) for ct in col_title]
        s = [[str(e) for e in row] for row in matrix]  # convert to str
        [r.insert(0,t+"|") for t,r in zip(row_title,s)] # insert a header column
        lens = [max(map(len, col)) for col in zip(*s)]  # get max length for columns
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)  # format of columns
        rows = [fmt.format(*row) for row in s] # combine columns by format
        table = '\n'.join(rows) # combine rows
        hc_w = max(map(len, row_title))
        t_w = max(map(len, rows))+4
        h1 = str(' '*hc_w) + '|\t' + str('\t'.join(col_title)) + '\n'
        h2 = str("-"*hc_w) + "+" + str("-"*t_w) + '\n'
        return h1 + h2 + table



    @staticmethod
    def get_blank_image(height, width, dtype=np.float64) -> np.ndarray:
        """ Generates a blank image ndarray  """
        return np.zeros((height,width), dtype=dtype)

    @staticmethod
    def draw_line_by_id(mol:Molecule, pidx1:int, pidx2:int, height:int, width) -> np.ndarray:
        (y1, x1) = mol.get_yx(pidx1)
        (y2, x2) = mol.get_yx(pidx2)
        return LineDetection.draw_line(x1, y1, x2, y2, height, width)

    @staticmethod
    def draw_line_pos(x1, y1, x2, y2) -> np.ndarray:
        pass

    @staticmethod
    def get_emphasized(mol, mask, factor=2.0, use_max=False, src_img=None):
        if src_img is None:
            src_img = mol.src_img.copy()
        if use_max:
            max_height = (mask * src_img).max()
            mask = mask.astype(dtype=bool)
            src_img[mask] = max_height * factor
            return src_img
        else:
            return src_img + (mask * src_img * (factor - 1.0))

    @staticmethod
    def get_pix_id_pair(vector_ar, vec_idx):
        (pix_id1, pix_id2) = (vector_ar[0],vector_ar[1])
        (id1, id2) = (pix_id1[vec_idx], pix_id2[vec_idx])
        return (id1, id2)
#endregion

    def mol_count(self):
        return len(self.molecules)

class SpmPlot:
    """ Helper class for plotting after TopoStats data processing """

    @staticmethod
    def image(image_data, figsize=(4,4), **plot_configs):
        """ Plot image of TopoStats np.ndarray """
        if image_data is None:
            raise ValueError("No image data")
        if not isinstance(image_data, np.ndarray):
            raise TypeError("The data is not Numpy.ndarray")
        fig, ax = plt.subplots(figsize=figsize)
        plt.imshow(image_data, **plot_configs)
        plt.show()

    @staticmethod
    def tile_images(image_dict, image_keys=[], figsize=(6,6), **plot_configs):
        """ Tile images """
        if len(image_dict) == 0:
            raise ValueError("No images specified.")
        if len(image_keys) == 0:  # if not specified, try to show all
            image_keys = list(image_dict.keys())
        cols = len(image_keys)
        if cols > 4:
            raise ValueError("Sorry, 4 images at maximum.")
        fig, ax = plt.subplots(1, cols, figsize=figsize)
        for col in range(cols):
            ax[col].set_title(image_keys[col])
            ax[col].imshow(image_dict[image_keys[col]], **plot_configs)
        fig.tight_layout()
        plt.show()

