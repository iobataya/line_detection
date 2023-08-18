# coding=utf-8
import math,os
import random
from vectors import V,VV,VMap,VMapList,VVCache
from decimal import *

class LineDetection:
    """
    Class to evaluate linear component in a shape from vector map
    Holds conditions for filtering
    """

    def __init__(self, vmap, min_length=2, max_length=200, allowed_empty=2, enable_cache=True):
        """
        Args:
            cutoff_angle (float): 0-180, cutoff angle difference from reference vector by degree
            ref_angle (float): initial reference angle, default: 0
        """
        self.allowed_empty_pixels = allowed_empty
        self.vvcache = VVCache(enable_cache=enable_cache)
        self.vvcache.register_cache('line', VV.calc_line_pixels)
        self.results = {}
        self.src_vmap = vmap
        self.vv_list = vmap.generate_all_pairs(min_length=min_length, max_length=max_length)
        self.vv_total_count = vmap.get_combinations_count()
        self.vv_count = len(self.vv_list)

    def sum_along_line(self, target_vmap_int, line_vv):
        """ Sum signal along a line VV in this map
        If there are empty pixels more than allowed_empty, returns zero.
        Args:
            target_vmap(VMap): VMap contains signals
            line_vv(VV):    A line for summing up signals
        Returns:
            float:  accumulated signal along a line
        """
        trace_line = self.vvcache.get_value('line', line_vv)
        signal_sum = 0
        none_count = self.allowed_empty_pixels
        for trace_pixel in trace_line:
            if trace_pixel in target_vmap_int:
                signal_sum += target_vmap_int[trace_pixel]
            else:
                none_count = none_count - 1
                if none_count == 0:
                    signal_sum = 0
                    break  # pixels were empty along the line !
        return signal_sum

    @staticmethod
    def dist2tsv(dic):
        keys = list(dic.keys())
        keys.sort()
        s = ''
        for key in keys:
            s = s + '{}\t{}\n'.format(key, dic[key])
        return s

class VFilter:
    """ Filter class provides filtering
        by_ methods proceed filtering one by one from source
        in order to track progress from external user interface
        Filtered items are in .filtered as list
    """
    def __init__(self, source, min_value, max_value=10000):
        self._current_idx = 0
        self.source = source
        self.filter_count = len(self.source)
        self.filtered = []
        self.min_value = min_value
        self.max_value = max_value

    def by_count(self):
        """ Filter an item and add to list if it is in range
        Args:
            source: list(VMap), list(VVMap)
            min_count (int): minimum count
            max_count (int): maximum count
        Returns:
            bool: True if source has more items to be filtered
            list: List of filtered items
        """
        if self._current_idx == self.filter_count:
            return False
        (min_count, max_count) = (self.min_value, self.max_value)
        items = self.source[self._current_idx]
        cnt = len(items)
        if cnt >= min_count and cnt <= max_count:
            self.filtered.append(items)
        self._current_idx += 1
        return True

    def __str__(self):
        s = "Source: {}, {}\n".format(type(self.source), len(self.source))
        s += "(min, max) : ({}, {})\n".format(self.min_value, self.max_value)
        s += "Current idx: {}\n".format(self._current_idx)
        return s

class GridTiler:
    def __init__(self, list_vmap, grid_cols=25):
        self.tiled_vmap = VMap()
        self.tiling_idx = 0
        # set sort type and get max size
        (max_w, max_h) = (0, 0)
        for vmap in list_vmap:
            if len(vmap) == 0:
                continue
            diagonal_v = vmap.get_diagonal()
            max_w = max(max_w, diagonal_v.x_int())
            max_h = max(max_h, diagonal_v.y_int())
        (self.grid_w,self.grid_h) = (max_w + 1, max_h + 1)
        self.to_be_tiled = VMapList(list_vmap)
        self.grid_cols = grid_cols

    def sort_by_count(self,reverse=False):
        self.to_be_tiled.sort_by_count(reverse=reverse)

    def sort_by_angle(self,reverse=False):
        self.to_be_tiled.sort_by_angle(reverse=reverse)

    def sort_by_max_value(self,reverse=False):
        self.to_be_tiled.sort_by_max_value(reverse=reverse)

    def tiling_add(self):
        if len(self.to_be_tiled) == 0:
            return False
        left_top = self._tiling_xy()
        vmap = self.to_be_tiled[self.tiling_idx]
        vmap = vmap.get_zero_offset()
        self.tiled_vmap.add_vmap(vmap.get_translated(left_top))
        self.tiling_idx += 1

    def _tiling_xy(self):
        idx = self.tiling_idx
        col_idx = idx % self.grid_cols
        row_idx = (idx - col_idx) / self.grid_cols
        return V(col_idx * self.grid_w, row_idx * self.grid_h)

    def __str__(self):
        setting = "Tiling VMaps {}, grid_cols: {},  grid:({} x {})".format(len(self.to_be_tiled), self.grid_cols, self.grid_w, self.grid_h)
        current = ", current idx:{} next position: {}".format(self.tiling_idx, self._tiling_xy())
        return setting + current
