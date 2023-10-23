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
        if self.width > 127 or self.height > 127:
            raise IndexError(f"Grain is too large over 127.({self.width},{self.height})")

        vec_x = self.src_yx[XAXIS] - self.src_left
        vec_y = self.src_yx[YAXIS] - self.src_top
        (self.x, self.y) = (vec_x.astype(np.int32), vec_y.astype(np.int32))
        self.yxT = np.array([self.y, self.x], dtype=np.int32).T

        self.mask = self.get_mask()
        if source_image is not None:
            self.set_source_image(source_image)

    def get_yx(self, pix_idx:int):
        return (self.y[pix_idx],self.x[pix_idx])

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

    def get_blank(self):
        return np.zeros((self.height, self.width), dtype=np.int32)

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

        self.config = linedet_config
        self.config.setdefault("min_len", 5)
        self.config.setdefault("max_len", 50)
        self.config.setdefault("max_pix", 500)
        self.config.setdefault("allowed_empty", 1)
        self.config.setdefault("record_neg_empty", False)
        self.config.setdefault("score_cutoff", 1.0)

        self.score_columns = ["mol_idx","score","empty","x1","y1","x2","y2","length","angle"]
        self.score_df = pd.DataFrame(columns=self.score_columns)
        self.stat_columns = ["mol_idx", "pixels", "total_vecs", "min_len", "max_len", "max_pix","len_filtered","score_cutoff"]
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
        min_len = self.config["min_len"]
        max_len = self.config["max_len"]

        dvectors = mol.get_displacement_vectors()  # all vectors
        dyx = dvectors.T[2:4]  # ndarray of [[dy,], [dx,]]
        dyx_sq = np.square(dyx).sum(axis=0)
        lengths = np.sqrt(dyx_sq)
        filtered_idx = np.where((lengths>=min_len) & (lengths<=max_len))[0]
        filtered = dvectors.take(filtered_idx, axis=0)
        self._add_len_filter_stat(mol, len(dyx[0]), len(filtered))

        return filtered

    def _add_len_filter_stat(self, mol:Molecule, total_vecs:np.int32, len_filtered:np.int32):
        stat = pd.DataFrame([{
            "mol_idx":mol.mol_idx,
            "pixels":mol.count(),
            "min_len":self.config["min_len"],
            "max_len":self.config["max_len"],
            "max_pix":self.config["max_pix"],
            "total_vecs":total_vecs,
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
            results["x1"].append(x1)
            results["y1"].append(y1)
            results["x2"].append(x2)
            results["y2"].append(y2)
            results["length"].append(length)
            results["angle"].append(angle)
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
        line_mask = LineDetection.draw_line(x1, y1, x2, y2, mol.height, mol.width)
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
        src = self.source_img
        overlay = LineDetection.get_blank_image(self.height,self.width)
        if len(self.score_df)==0:
            raise ValueError("No scores estimated.")
        for mol in self.molecules:
            mol_idx = mol.mol_idx
            df = self.score_df.loc[self.score_df["mol_idx"]==mol_idx].sort_values("norm_score", ascending=False)
            if len(df[:num_lines]) == 0:
                continue
            mol_pos = [mol.src_left, mol.src_top, mol.src_left, mol.src_top]  # positions of Molecule
            mol_mask = mol.get_mask()
            max_height = (mol.src_img * mol_mask).max()
            lines = LineDetection.get_blank_image(self.height,self.width)
            for i in range(len(df[:num_lines])):
                p = df.iloc[i][3:7] + mol_pos
                line = self.draw_line(p["x1"], p["y1"], p["x2"], p["y2"],self.height,self.width)
                lines = lines + (line * max_height * factor)
            overlay = overlay + (lines / num_lines)
        return overlay + src


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
    def plot_image(image, aspect = 1.0):
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.imshow(image, cmap="afmhot", aspect=aspect)
        plt.show()

    @staticmethod
    def get_blank_image(height, width, dtype=np.float64) -> np.ndarray:
        """ Generates a blank image ndarray  """
        return np.zeros((height,width), dtype=dtype)


    @staticmethod
    def draw_line(x1, y1, x2, y2, height, width) -> np.ndarray:
        """ Returns binary image of line in a defined size
        Args:
            (x1,y1,x2,y2) (int): starting and end point of line
            np_area (np.nd_array): nd_array to draw the line as 1
        Returns:
            (np.nd_array): area drawn
        """
        np_area = np.zeros((height,width), dtype=np.int32)
        def _set_pixel(x, y, value=1):
            np_area[y][x] = value

        def _vertical_line(y1, y2, x):
            (i, j) = (min(y1, y2), max(y1, y2) + 1)
            for y in range(i, j):
                _set_pixel(x, y)
            return np_area

        def _horizontal_line(x1, x2, y):
            (i, j) = (min(x1, x2), max(x1, x2) + 1)
            for x in range(i, j):
                _set_pixel(x, y)
            return np_area

        def _bresenham(x1, y1, x2, y2):
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
                _set_pixel(x1 + x * xx + y * yx, y1 + x * xy + y * yy)
                if D >= 0:
                    y += 1
                    D -= 2*dx
                D += 2*dy
            return np_area

        if x1 == x2 and y1 == y2: # a point
            _set_pixel(x1, y1)
            return np_area

        if x1 == x2 :  # vertical line
            ret_ndarray = _vertical_line(y1, y2, x1)
        elif y1 == y2:  # horizontal line
            ret_ndarray = _horizontal_line(x1, x2, y1)
        else:
            ret_ndarray = _bresenham(x1, y1, x2, y2)
        return ret_ndarray

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


