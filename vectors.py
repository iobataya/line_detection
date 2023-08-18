# coding=utf-8
import os
import math
from decimal import Decimal, ROUND_HALF_UP

class V:
    """Vector class holds x and y as float value
    """
    def __init__(self, x, y):
        self.xy = (float(x), float(y))

    def x(self):
        """x value"""
        return self.xy[0]

    def x_int(self):
        """ x value as int rounded by round_half_up """
        x = Decimal(self.xy[0])
        return int(x.quantize(Decimal('1.'), rounding=ROUND_HALF_UP))

    def y(self):
        """y value """
        return self.xy[1]

    def y_int(self):
        """ y value as int rounded by round_half_up """
        y = Decimal(self.xy[1])
        return int(y.quantize(Decimal('1.'), rounding=ROUND_HALF_UP))

    # region primitive calcs
    def area(self):
        """area"""
        return self.xy[0] * self.xy[1]

    def cross(self, other):
        """ Returns cross product
        Args:
            other (V): other vector
        Returns:
            float: this vector X other vector
        """
        (x1, y1) = self.xy
        (x2, y2) = other.xy
        return x1 * y2 - x2 * y1

    def magnitude(self):
        """magnitude (length) """
        (x, y) = self.xy
        return (x * x + y * y) ** 0.5

    def max(self, other):
        (x1, y1) = self.xy
        (x2, y2) = other.xy
        return V(max(x1, x2), max(y1, y2))

    def min(self, other):
        (x1, y1) = self.xy
        (x2, y2) = other.xy
        return V(min(x1, x2), min(y1, y2))

    def neighborings(self, step=1.0):
        """ Returns neighboring 8 vectors
        Args:
            step (float): size of neighboring
        Returns:
            list: list of 8 vectors as V
        """
        (x, y) = self.xy
        ret = [V(x - step, y - step),
               V(x,        y - step),
               V(x + step, y - step),
               V(x - step, y),
               V(x + step, y),
               V(x - step, y + step),
               V(x,        y + step),
               V(x + step, y + step)]
        return ret

    def rotate(self, radian):
        """Returns rotated vector
        Args:
            radian (float): angle for rotation as radian
        Returns:
            V: rotated vector
        """
        (x, y) = self.xy
        cos = math.cos(radian)
        sin = math.sin(radian)
        rx = x * cos - y * sin
        ry = x * sin + y * cos
        return V(rx, ry)

    def round(self):
        """Returns rounded by half_up
        Note that values are float
        """
        return V(self.x_int(), self.y_int())

    def to_quad12(self):
        """Rotate if this vector is in III or IV quadrant"""
        if self.xy[1] >= 0:
            return self
        else:
            return V(-self.xy[0], -self.xy[1])
    # endregion

    # region calcs
    def angle(self, quadrant12=False):
        """ Returns angle of this vector as degree
        Args:
            quadrant12 (bool):forcing angle to 1st/2nd quadrant or not
        Returns:
            float: angle as degree
        """
        rad = self.radian(quadrant12)
        return 180 * rad / math.pi

    def normalized(self):
        """normalized vector """
        m = self.magnitude()
        return V(self.xy[0] / m, self.xy[1] / m)

    def cross_norm(self, other):
        """Returns normalized cross product
        Args:
            other (V): other vector
        Returns:
            V: normalized this vector X normalized other vector
        """
        return self.normalized().cross(other.normalized())

    def radian(self, quadrant12=False):
        """ Returns angle of this vector as radian
        Args:
            quadrant12 (bool):forcing angle to 1st/2nd quadrant or not
        Returns:
            float: angle as radian
        """
        v = self
        if quadrant12:
            v = v.to_quad12()
        return math.atan2(v.y(), v.x())

    def serialize(self):
        (x, y) = self.xy
        if x % 1.0 == 0:
            sx = str(int(x))
        else:
            sx = str(x)
        if y % 1.0 == 0:
            sy = str(int(y))
        else:
            sy = str(y)
        return sx + ',' + sy

    @staticmethod
    def deserialize(s):
        xy = s.split(',')
        if len(xy) != 2:
            raise ValueError("cannot deserialized {}".format(s))
        return V(float(xy[0]),float(xy[1]))

    # endregion

    def _cross_norm(v_tuple):
        """ Returns absolute cross product
        Provided as a class method, calc_cross
        Args:
            v1: 1st vector
            v2: Normalized 2nd vector (reference)
        Returns:
            float: this v1 X v2
        """
        (v1, v2) = v_tuple
        v1 = v1.to_quad12().normalized()
        return abs(v1.cross(v2))
    calc_cross = staticmethod(_cross_norm)

    # region  operators
    def __add__(self, other):
        (x1, y1) = self.xy
        (x2, y2) = other.xy
        return V(x1 + x2, y1 + y2)

    def __sub__(self, other):
        (x1, y1) = self.xy
        (x2, y2) = other.xy
        return V(x1 - x2, y1 - y2)

    def __mul__(self, other):
        (x1, y1) = self.xy
        (x2, y2) = other.xy
        return V(x1 * x2, y1 * y2)

    def __neg__(self):
        (x, y) = self.xy
        return V(-x, -y)

    """ for python 2.7 """
    def __div__(self, other):
        (x1, y1) = self.xy
        (x2, y2) = other.xy
        return V(x1 / x2, y1 / y2)

    def __truediv__(self, other):
        (x1, y1) = self.xy
        (x2, y2) = other.xy
        return V(x1 / x2, y1 / y2)

    def __str__(self):
        (x, y) = self.xy
        return 'V({}, {})'.format(x, y)

    def __repr__(self):
        (x, y) = self.xy
        return 'V({}, {})'.format(x, y)

    def __eq__(self, other):
        if not isinstance(other, V):
            return False
        return self.__hash__() == other.__hash__()

    def __lt__(self, other):
        if self.x() == other.x():
            return self.y() < other.y()
        return self.x() < other.x()

    def __hash__(self):
        return hash(self.xy)
    # endregion


class VV:
    """ Vector pair of V
    Args:
        v1(V): vector 1
        v2(V): vector 2
    """

    def __init__(self, v1, v2):
        self.vv = (v1, v2)
        if not isinstance(v1, V) or not isinstance(v2, V):
            raise ValueError("aruguments should be class V")

    # region vars
    def v1(self):
        return self.vv[0]

    def x1(self):
        return self.vv[0].x()

    def y1(self):
        return self.vv[0].y()

    def v2(self):
        return self.vv[1]

    def x2(self):
        return self.vv[1].x()

    def y2(self):
        return self.vv[1].y()
    # endregion

    def displacement(self):
        """ Returns displacement vector v1 - v2
        Returns:
            V: displacement vector
        """
        return self.vv[0] - self.vv[1]

    def distance(self):
        """ Returns distance between two vectors """
        return self.displacement().magnitude()

    def get_pixels_along(self):
        """ Returns positions between vec pair
        List of positions rounded V along displacement vector
        Returns:
            list: [V,..]
        """
        if self.distance() == 0:
            return []
        vec_pair = self.displacement()
        (end_v, start_v) = (self.v1(), self.v2())
        if vec_pair.x_int() == 0:  # vertical
            return self._vertical_line(start_v, end_v)
        elif vec_pair.y_int() == 0:  # horizontal
            return self._horizontal_line(start_v, end_v)
        else:
            return self._bresenham(start_v, end_v)

    def _horizontal_line(self, start, end):
        (x0, y0) = (start.x_int(), start.y_int())
        (x1, y1) = (end.x_int(), end.y_int())
        dx = abs(x1 - x0)
        line_pixel = []
        step = 1
        if x0 > x1:
            step = -1
        (x, y) = (x0, y0)
        for i in range(dx):
            line_pixel.append(V(x, y))
            x += step
        line_pixel.append(V(x1, y1))
        return line_pixel

    def _vertical_line(self, start, end):
        (x0, y0) = (start.x_int(), start.y_int())
        (x1, y1) = (end.x_int(), end.y_int())
        dy = abs(y1 - y0)
        line_pixel = []
        step = 1
        if y0 > y1:
            step = -1
        (x, y) = (x0, y0)
        for i in range(dy):
            line_pixel.append(V(x, y))
            y += step
        line_pixel.append(V(x1, y1))
        return line_pixel

    def _bresenham(self, start, end):
        """
        based on
        https://github.com/encukou/bresenham/blob/master/bresenham.py
        """
        line_pixel = []
        (x0, y0) = (start.x_int(), start.y_int())
        (x1, y1) = (end.x_int(), end.y_int())

        dx = x1 - x0
        dy = y1 - y0

        xsign = 1 if dx > 0 else -1
        ysign = 1 if dy > 0 else -1

        dx = abs(dx)
        dy = abs(dy)

        if dx > dy:
            xx, xy, yx, yy = xsign, 0, 0, ysign
        else:
            dx, dy = dy, dx
            xx, xy, yx, yy = 0, ysign, xsign, 0

        D = 2*dy - dx
        y = 0

        for x in range(dx + 1):
            line_pixel.append(V(x0 + x*xx + y*yx, y0 + x*xy + y*yy))
            if D >= 0:
                y += 1
                D -= 2*dx
            D += 2*dy
        return line_pixel

    def swap(self):
        return -self

    def swap_to_outward(self):
        """ Swap vectors so that it directs to outward
        """
        if self.vv[0].magnitude() < self.vv[1].magnitude():
            return self
        else:
            return -self

    def round(self):
        """Returns round half up vector.
        The values are rounded, but type is float.
        """
        return VV(self.vv[0].round(), self.vv[1].round())

    def _distance(vv):
        """ Returns distance between vector pair for cache
        Provided as a class method, calc_distance
        Args:
            vv: vector pair
        Returns:
            float: distance
        """
        return vv.distance()
    calc_distance = staticmethod(_distance)

    def _get_pixels_along(vv):
        """ Returns list of pixels along a line of VV
        Provided as as class method, calc_line_pixels
        Args:
            vv: vector pair
        Returns:
            list: [V,...]
        """
        return vv.get_pixels_along()
    calc_line_pixels = staticmethod(_get_pixels_along)


    # region builtin methods
    def __neg__(self):
        return VV(self.v2(), self.v1())

    def __eq__(self, other):
        if not isinstance(other, VV):
            raise ValueError("VV is not compared to other class.")
        return self.__hash__() == other.__hash__()

    def __str__(self):
        return 'VV({}, {})'.format(self.vv[0], self.vv[1])

    def __repr__(self):
        (v1, v2) = self.vv
        return 'VV({}, {})'.format(self.vv[0], self.vv[1])

    def __hash__(self):
        return hash(self.vv)
    # endregion



class VMap(dict):
    """Vector map
    Bears vectors as keys for a certain value at each vectors.
    Key for this dict should be class V
    """
    # region constructor and parameters
    SORT_BY_MAX_VALUE = 0
    SORT_BY_COUNT = 1
    SORT_BY_ANGLE = 2

    def __init__(self, list_vecs=[], fill_value=0, list_vecvalues=[]):
        """ Instantiate VMap
        Args:
            list_vecs(list):vectors list [V,V,]
            fill_value(any):filling value
        """
        super(dict, self).__init__()
        self.sort_type = self.SORT_BY_MAX_VALUE
        if len(list_vecs) > 0:
            if isinstance(list_vecs, list):
                if not isinstance(list_vecs[0], V):
                    raise KeyError("Value key should be V")
                for vec in list_vecs:
                    self.set_value(vec, fill_value)
            elif isinstance(list_vecs, dict):
                vec_keys = list(list_vecs)
                if not isinstance(vec_keys[0], V):
                    raise KeyError("Vector key should be V")
                for key in vec_keys:
                    self[key] = list_vecs[key]
        elif len(list_vecvalues) > 0:
            if not isinstance(list_vecvalues[0], VecValue):
                raise KeyError("items in list_vecvalues should be VecValue")
            for vval in list_vecvalues:
                self[vval.vector] = vval.value

    def set_sort_type(self,sort_type):
        self.sort_type = sort_type


    def copy(self):
        """Copy
        Returns:
            VMap: a clone by deep copy
        """
        vecs = self.vec_list()
        ret = VMap()
        if len(vecs) == 0:
            return ret
        for vec in vecs:
            ret[vec] = self[vec]
        return ret

    def first_vec(self):
        """ Returns the first vector in map.
        If none, returns none.
        Returns:
            V: the first vector. None if none in the map.
        """
        vec_list = self.vec_list()
        if len(vec_list) == 0:
            return None
        else:
            return vec_list[0]

    def get_value(self, key):
        return self.get(key)

    def set_value(self, key, value):
        """ Set value by a vector key
        Raises:
            KeyError: when the key is not V
        """
        if not isinstance(key, V):
            raise KeyError("key aruguments should be V")
        self[key] = value

    def round(self):
        """ Get clone with half-up-round vector keys for pixelation
        Returns:
            VMap:   copy of VMap with half-up-rounded keys
        """
        half_up = VMap()
        for key in self.keys():
            half_up[key.round()] = self[key]
        return half_up

    def vec_list(self):
        """Vectors as a list
        Returns:
            list: vectors in map
        """
        return list(self.keys())

    def vec_keys(self):
        """Vectors as a set
        Returns:
            dict.keys(): vectors in map
        """
        return self.keys()
    # endregion

    # region calcs
    def get_min_max_vectors(self):
        """Min/Max vector pair
        Returns:
            tuple: (V, V) pair of min/max vector
        """
        vectors = self.vec_list()
        if len(vectors) == 0:
            raise ValueError("no vector")
        first_vec = vectors[0]
        vmin = first_vec
        vmax = first_vec
        for i in range(1, len(vectors)):
            vec = vectors[i]
            vmax = vmax.max(vec)
            vmin = vmin.min(vec)
        return (vmin, vmax)

    def get_min_max_values(self):
        """Min/max values and its vector key
        Returns:
            tuple: (VValue,VValue), pair of vector and value of min / max
        """
        vectors = self.vec_list()
        if len(vectors) == 0:
            raise ValueError("no vector")
        first_vec = vectors[0]
        vmin = first_vec
        vmax = first_vec
        (min_value, max_value) = (self[vmin], self[vmax])
        for i in range(1, len(vectors)):
            vec = vectors[i]
            if self[vec] < min_value:
                vmin = vec
            if self[vec] > max_value:
                vmax = vec
            (min_value, max_value) = (self[vmin], self[vmax])
        (min_vvalue, max_vvalue) = (
            VecValue(vmin, self[vmin]), VecValue(vmax, self[vmax]))
        return (min_vvalue, max_vvalue)

    def _get_neighboring_from(self, first_vec):
        """Get a vmap neighboring the vec
        Find and generate a VMap of 8 x neiboring vectors starting at the vector.
        Picked-up vectors are removed from this instance!
        Args:
            V: starting vector
        Returns:
            VMap: extracted vector map from this instance.
        """
        found_vmap = VMap()
        needs_check = VMap()
        needs_check[first_vec] = self[first_vec]
        while(len(needs_check) > 0):
            # a vector checking in this turn.
            checking_vec = needs_check.first_vec()
            del needs_check[checking_vec]
            # move the vector to found_vmap from this instance
            found_vmap[checking_vec] = self[checking_vec]
            del self[checking_vec]
            # neighboring vectors around checking_vec
            neighbors = checking_vec.neighborings()
            for neighbor in neighbors:
                if neighbor in self:
                    # neighbor found. Add it to needs_check and found_vmap
                    # these are to be checked in later next turn
                    needs_check[neighbor] = self[neighbor]
                    found_vmap[neighbor] = self[neighbor]
        return found_vmap

    def get_neighboring_vmaps(self, min_vectors=2, max_vmaps=100):
        """ Returns list of VMaps having neighboring pixels
        Args:
            min_vectors(int): minimum vector number to collect
            max_vmaps(int):   maximum VMaps
        Returns:
            list: VMap list
        """
        vmap_copy = self.copy()
        found_vmaps = []
        while(len(found_vmaps) < max_vmaps):
            if len(vmap_copy) == 0:
                break
            first_vec = vmap_copy.first_vec()
            found_vmap = vmap_copy._get_neighboring_from(first_vec)
            if len(found_vmap) < min_vectors:
                continue
            found_vmaps.append(found_vmap)
        return found_vmaps

    def get_offset(self):
        """ Get left-top vector (vmin)
        """
        (vmin, _) = self.get_min_max_vectors()
        return vmin

    def get_diagonal(self):
        """ Get diagonal vector of whole vectors
        """
        (vmin, vmax) = self.get_min_max_vectors()
        diagonal = vmax - vmin
        return diagonal

    def get_zero_offset(self, round_vec=True):
        """ Get Vmap translated to top-left is at (0,0)
        Args:
            round_vec (bool): convert to int for pixelation if True
        Returns:
            VMap: translated VMap with left-top is at (0,0)
        """
        offset = self.get_offset()
        return self.get_translated(-offset, round_vec)

    def get_translated(self, translation, round_vec=True):
        """ Get translated VMap
        Args:
            shift (V):translation vector
            round_vec (bool): convert to int for pixelation if True
        """
        if not isinstance(translation, V):
            raise TypeError("Set translation as V.")
        ret_vmap = VMap()
        for vec in self.vec_list():
            new_vec = vec + translation
            if round_vec:
                new_vec = new_vec.round()
            ret_vmap[new_vec] = self[vec]
        return ret_vmap

    def get_rotated(self, radian, center=V(0, 0)):
        """Get rotated VMap
        Args:
            radian(float): rotating angle by radian
            center(V): rotating center
        Returns:
            VMap: a copy of rotated VMap
        """
        ret_vmap = VMap()
        vecs = self.vec_list()
        cos = math.cos(radian)
        sin = math.sin(radian)
        for i in range(0, len(vecs)):
            disp_vec = vecs[i] - center
            rot_disp_vec = disp_vec.rotate(radian)
            ret_vmap[rot_disp_vec + center] = self[vecs[i]]
        return ret_vmap

    def multiply_vmap(self, multi_vmap):
        """ multiply by value of multi_vmap.
        """
        if multi_vmap == None or len(multi_vmap) == 0:
            return
        if not isinstance(multi_vmap, VMap):
            raise TypeError()
        for new_key in multi_vmap.vec_list():
            if new_key in self:
                self[new_key] *= multi_vmap[new_key]

    def add_vmap(self, add_vmap):
        """ add value from add_vmap
        """
        if add_vmap == None or len(add_vmap) == 0:
            return
        if not isinstance(add_vmap, VMap):
            raise TypeError()
        for new_key in add_vmap.vec_list():
            if new_key in self:
                self[new_key] += add_vmap[new_key]
            else:
                self[new_key] = add_vmap[new_key]

    def add_line(self, vec_pair, value=1.0):
        """Add a line
        Args:
            vec_pair(VV):   start and end vector
            value(float):   filling value
        """
        if vec_pair.v1() == vec_pair.v2():
            return
        pixels = vec_pair.get_pixels_along()
        for pix in pixels:
            self[pix] = value

    def add_rect(self, vec_pair, value=1.0):
        """Add a rect
        Args:
            vec_pair(VV): vector pair of corners
            value(float): filling value
        """
        (v1, v2) = vec_pair.vv
        (left, right) = (v1.min(v2).x(), v1.max(v2).x())
        (top, bottom) = (v1.min(v2).y(), v1.max(v2).y())
        self.add_line(VV(V(left, top), V(right, top)),  value=value)
        self.add_line(VV(V(right, top), V(right, bottom)),  value=value)
        self.add_line(VV(V(right, bottom), V(left, bottom)), value=value)
        self.add_line(VV(V(left, bottom), V(left, top)), value=value)

    # endregion
    def get_combinations_count(self):
        vecs_count = len(self.vec_list())
        return VMap._comb(vecs_count, 2)

    @staticmethod
    def _pert(n):
        if n > 1:
            n = n - 1
            return (n + 1) * VMap._pert(n)
        else:
            return n
    @staticmethod
    def _comb(n, r):
        return VMap._pert(n) / r / VMap._pert(n - r)

    def generate_all_pairs(self, min_length=0, max_length=200):
        """ Generate all vector pairs from vmap
        Returns:
            list: list of VV
        """
        ret_vv = []
        vecs = self.vec_list()
        n = len(vecs)
        for i in range(0, n):
            v1 = vecs[i]
            for j in range(i + 1, n):
                v2 = vecs[j]
                vv = VV(v1, v2)
                distance = vv.distance()
                if min_length <= distance and distance <= max_length:
                    ret_vv.append(vv)
        return ret_vv

    def str_list(self):
        """Return list of VMap as tab separated str."""
        keys = list(self.keys())
        keys.sort()
        ret = ''
        for i in range(0, len(keys)):
            ret += "{}\t{}\t{}\n".format(keys[i].x(),
                                         keys[i].y(), self[keys[i]])
        return ret

    def __eq__(self, other):
        """ Note that this check takes long time.
        """
        if not isinstance(other, VMap):
            return False
        for key in self.vec_list():
            if not key in other:
                return False
            if self[key] != other[key]:
                return False
        return True


    def __lt__(self, other):
        if len(self) == 0:
            return True
        if len(other) == 0:
            return False
        if self.sort_type == self.SORT_BY_ANGLE:
            disp1 = self.get_diagonal()
            disp2 = other.get_diagonal()
            return disp1.angle(quadrant12=True) < disp2.angle(quadrant12=True)
        elif self.sort_type == self.SORT_BY_COUNT:
            return len(self) < len(other)
        else:
            (_, val_max1) = self.get_min_max_values()
            (_, val_max2) = other.get_min_max_values()
            return val_max1.value < val_max2.value

    def get_vecvalue_list(self):
        ret = []
        for vec in self.vec_list():
            vecvalue = VecValue(vec,self[vec])
            ret.append(vecvalue)
        return ret

    def serialize(self):
        vecvalue_list = self.get_vecvalue_list()
        line = ''
        for vval in vecvalue_list:
            line = line + vval.serialize() + ';'
        return line[:-1]

    @staticmethod
    def deserialize(s):
        items = s.split(';')
        vmap = VMap()
        if s!='' and len(items)>0:
            for item in items:
                vecval = VecValue.deserialize(item)
                vmap[vecval.vector] = vecval.value
        return vmap

    def __str__(self):
        if len(self) == 0:
            return "Vmap: NO vector"
        (min_vec, max_vec) = self.get_min_max_vectors()
        vec_range = "vector {}-{}".format(min_vec, max_vec)
        (min_vval, max_vval) = self.get_min_max_values()
        val_range = "values {}-{}".format(min_vval.value, max_vval.value)
        return "VMap: {} vectors, {}, {}".format(len(self), vec_range, val_range)

class VMapList(list):
    def __init__(self,*args):
        super(list, self).__init__()
        if args==None or len(args)==0:
            return
        if len(args)==1 and isinstance(args[0],list):
            for arg in args[0]:
                if not isinstance(arg, VMap):
                    raise TypeError()
                self.append(arg)
        elif len(args)>1:
            for arg in args:
                if not isinstance(arg, VMap):
                    raise TypeError()
                self.append(arg)
        else:
            raise TypeError()

    def sort_by_count(self, reverse=False):
        for vmap in self:
            vmap.set_sort_type(VMap.SORT_BY_COUNT)
        self.sort(reverse=reverse)

    def sort_by_angle(self, reverse=False):
        for vmap in self:
            vmap.set_sort_type(VMap.SORT_BY_ANGLE)
        self.sort(reverse=reverse)

    def sort_by_max_value(self, reverse=False):
        for vmap in self:
            vmap.set_sort_type(VMap.SORT_BY_MAX_VALUE)
        self.sort(reverse=reverse)

    @staticmethod
    def isVMapList(obj):
        """ Check if the obj is list of VMap
        Args:
            obj(ANY): to be checked
        Returns:
            bool: True if obj is list containing VMap
        """
        if obj == None or not isinstance(obj, list):
            return False
        if len(obj) == 0:
            return False
        if not isinstance(obj[0],VMap):
            return False
        return True

class VecValue:
    """ Vector and value
    """

    def __init__(self, vector, value):
        """ Vector and value
        Args:
            vector(V):  vector as V
            value(any): value
        """
        if not isinstance(vector, V):
            raise ValueError("vector should be V.")
        self.vector = vector
        self.value = value

    def serialize(self):
        vec_str = self.vector.serialize()
        val_str = str(self.value)
        return vec_str + ':' + val_str

    @staticmethod
    def deserialize(s):
        v_val = s.split(':')
        if len(v_val) != 2:
            raise ValueError("cannot be deserialized {}.".format(s))
        vec = V.deserialize(v_val[0])
        val = float(v_val[1])
        return VecValue(vec,val)

    def __eq__(self, other):
        return self.vector == other.vector and self.value == other.value

    def __str__(self):
        return 'Vec {}, Value {}'.format(self.vector, self.value)

    def __repr__(self):
        return '{}:{}'.format(self.vector, self.value)

class VVCache(dict):
    """ Cache class with VV keys
    """

    def __init__(self, enable_cache=True):
        super(dict, self).__init__()
        self.read_cnt = {}
        self.cache_hit_cnt = {}
        self.func = {}
        self.enable_cache = enable_cache

    def register_cache(self, name, func):
        """ Register cache name and function to calculate
        Args:
            name(str): name of cache
            func(method): static method that can calculate method from VV
        """
        self[name] = {}
        self.func[name] = func
        self.read_cnt[name] = 0
        self.cache_hit_cnt[name] = 0

    def get_value(self, name, key):
        if not self.enable_cache:
            return self.func[name](key)

        self.read_cnt[name] += 1
        if key in self[name]:
            pass
            self.cache_hit_cnt[name] += 1
        else:
            v = self.func[name](key)
            self[name][key] = v
        return self[name][key]

    def reset_counts(self):
        for cache_name in self.keys():
            self.read_cnt[cache_name] = 0
            self.cache_hit_cnt[cache_name] = 0

    def __str__(self):
        ret_str = "VVCache:{"
        if self.enable_cache:
            for cache_name in self.keys():
                total = len(self[cache_name])
                read_cnt = self.read_cnt[cache_name]
                hit_cnt = self.cache_hit_cnt[cache_name]
                if read_cnt > 0:
                    hit_ratio = "{}%".format(
                        round(100.0 * hit_cnt / read_cnt, 1))
                else:
                    hit_ratio = "not read"
                ret_str += "{}:total {}, read {}, hits {}({}), ".format(
                    cache_name, total, read_cnt, hit_cnt, hit_ratio)
        else:
            ret_str += " - cache disabled - "
        ret_str += "}"
        return ret_str

class VMapIO:
    def __init__(self):
        self.head_frmt = "# {}\n"
        self.data_frmt = "{}\n"
        self.file_handle = None
        self.source = None
        self.vmap_list = []
        self._current_idx = 0
        self._list_count = 0
        self._file_size = 0
        self.mode = ''

    def open(self, filepath, mode, source = None):
        self.filepath = filepath
        if mode == 'w':
            if source == None:
                raise ValueError("Source is none.")
            if isinstance(source, VMap):
                self.vmap_list = VMapList()
                self.vmap_list.append(source)
            elif isinstance(source, VMapList):
                self.vmap_list = source
            elif isinstance(source, list):
                self.vmap_list = VMapList()
                for vmap in source:
                    if isinstance(vmap, VMap):
                        self.vmap_list.append(vmap)
            else:
                raise TypeError("Source should be VMap or VMapList.")
            self.file_handle = open(self.filepath, mode='w')
            self._current_idx = 0
            self._list_count = len(self.vmap_list)
            self.mode = 'w'
        elif mode == 'r':
            self.vmap_list = VMapList()
            self._file_size = os.path.getsize(self.filepath)
            self.file_handle = open(self.filepath, mode='r')
            self.mode = 'r'
        else:
            raise ValueError("mode should be r or w")

    def close(self):
        if not self.file_handle.closed:
            self.file_handle.close()

    def write_vmap(self):
        if self.file_handle.closed or self.mode=='r':
            return False
        if self._current_idx >= self._list_count:
            self.file_handle.close()
            return False
        vmap = self.vmap_list[self._current_idx]
        # comment line
        self.file_handle.write(self.head_frmt.format(vmap.__str__()))
        # data line of vmap
        self.file_handle.write(self.data_frmt.format(vmap.serialize()))
        self._current_idx += 1
        return True

    def read_vmap(self):
        if self.file_handle.closed or self.mode=='w':
            return False
        line = self.file_handle.readline()
        if line == '': # EOF is empty
            return False
        while(line[0] == '#' or line[0] == '\n'):
            line = self.file_handle.readline()
        line = line.rstrip('\n')
        vmap = VMap.deserialize(line)
        self.vmap_list.append(vmap)
        return True

    def progress(self):
        if self.file_handle == None or self.file_handle.closed:
            return 1.0
        if self.mode == 'w':
            return (self._current_idx + 1.0) / self._list_count
        else:
            return (self.file_handle.tell() + 1.0) / self._file_size

    @staticmethod
    def _tsv_format_str(items):
        s = ""
        for item in items:
            s += item + "\t"
        return s[:-1] + "\n"
