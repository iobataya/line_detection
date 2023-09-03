# coding=utf-8
import os,time
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
        x = x1 if x1 > x2 else x2
        y = y1 if y1 > y2 else y2
        return V(x, y)

    def min(self, other):
        (x1, y1) = self.xy
        (x2, y2) = other.xy
        x = x1 if x1 < x2 else x2
        y = y1 if y1 < y2 else y2
        return V(x, y)

    def covers(self, other):
        (x1, y1) = (self.x(), self.y())
        (x2, y2) = (other.x(), other.y())
        if (x1 * x2 < 0) or (y1 * y2 < 0): # different quadrant
            return False
        if (x1 * y2 - x2 * y1) != 0: # cross product should be 0
            return False
        if (x1 >= 0):
            sx = (x1 >= x2)
        else:
            sx = (x1 <= x2)
        if (y1 >= 0):
            sy = (y1 >= y2)
        else:
            sy = (y1 <= y2)
        return sx and sy

    def neighborings(self, step=1.0):
        """ Returns neighboring 8 vectors
        Args:
            step (float): size of neighboring
        Returns:
            set: set of 8 vectors as V
        """
        (x, y) = self.xy
        ret = {V(x - step, y - step),
               V(x,        y - step),
               V(x + step, y - step),
               V(x - step, y),
               V(x + step, y),
               V(x - step, y + step),
               V(x,        y + step),
               V(x + step, y + step)}
        return ret

    def dialation_mask(self, step=1.0):
        """ Returns neighboring 5 vectors for dialation

        Args:
            step (float): size of neighboring
        Returns:
            set: set of 5 vectors as V
         """
        (x, y) = self.xy
        ret = {V(x,        y - step),
               V(x - step, y),
               V(x,        y),
               V(x + step, y),
               V(x,        y + step)}
        return ret

    @staticmethod
    def get_dialated_set(vector_set, dialation=0):
        dialated_set = vector_set
        for i in range(0,dialation):
            for v in vector_set:
                dialated_set = dialated_set | v.dialation_mask()
        return dialated_set

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

    def cross_product(self):
        """ Returns cross product of displacement vector v1 - v2
        Returns:
            float: cross product value
        """
        return self.v1().cross(self.v2())

    def cross_product_norm(self):
        """ Returns normalized cross product of v1 - v2
        value equals to sinThita
        Returns:
            float: normalized cross product (sinThita)
        """
        if self.displacement().magnitude() == 0:
            return 0
        return self.cross_product()/self.displacement().magnitude()

    def distance(self):
        """ Returns distance between two vectors """
        return self.displacement().magnitude()

    def get_pixels_along(self, dialation=0):
        """ Returns positions between vec pair
        Get set of V along displacement vector

        Args:
            dialation(int): Dialation to thicken line
        Returns:
            Set: (V,..)
        """
        ret_set = set()
        if self.distance() == 0:
            return ret_set
        vec_pair = self.displacement()
        (end_v, start_v) = (self.v1(), self.v2())
        if vec_pair.x_int() == 0:  # vertical
            ret_set = self._vertical_line(start_v, end_v)
        elif vec_pair.y_int() == 0:  # horizontal
            ret_set = self._horizontal_line(start_v, end_v)
        else:
            ret_set = self._bresenham(start_v, end_v)
        ret_set = V.get_dialated_set(ret_set)
        return ret_set


    def _horizontal_line(self, start, end):
        (x0, y0) = (start.x_int(), start.y_int())
        (x1, y1) = (end.x_int(), end.y_int())
        dx = abs(x1 - x0)
        line_pixel = set()
        step = 1
        if x0 > x1:
            step = -1
        (x, y) = (x0, y0)
        for i in range(dx):
            line_pixel.add(V(x, y))
            x += step
        line_pixel.add(V(x1, y1))
        return line_pixel

    def _vertical_line(self, start, end):
        (x0, y0) = (start.x_int(), start.y_int())
        (x1, y1) = (end.x_int(), end.y_int())
        dy = abs(y1 - y0)
        line_pixel = set()
        step = 1
        if y0 > y1:
            step = -1
        (x, y) = (x0, y0)
        for i in range(dy):
            line_pixel.add(V(x, y))
            y += step
        line_pixel.add(V(x1, y1))
        return line_pixel

    def _bresenham(self, start, end):
        """
        based on
        https://github.com/encukou/bresenham/blob/master/bresenham.py
        """
        line_pixel = set()
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
            line_pixel.add(V(x0 + x*xx + y*yx, y0 + x*xy + y*yy))
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

    def covers(self, vv):
        """ Check if this vv covers argument VV

        Arguments:
            vv(VV): VV to be checked
        Returns:
            bool: True if this covers VV
        """
        p = self
        pp = p.v1() - p.v2()
        vv = vv
        q1 = vv.v1() - p.v2()
        q2 = vv.v2() - p.v2()
        return pp.covers(q1) and pp.covers(q2)

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

    # TODO: How to handle caching of line !?
    def _get_pixels_along(vv):
        """ Returns list of pixels along a line of VV
        Provided as as class method, calc_line_pixels
        Args:
            vv: vector pair
        Returns:
            set: {V,...}
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


#TODO: Convert set of positions of V for line to Line class !

class Line:
    """ Line class holds vector pair, dialation and set of positions

    Attributes:
        vv(VV): vector pair of this line
        dialation(int): repetition of dialation to thicken
    """
    def __init__(self, vec_pair, dialation=0):
        """ Line class
        Args:
            vec_pair(VV): vector pair of line
            dialation(int): repetitin of dialation
        """
        self.vv = vec_pair
        self.dialation = dialation
        self.vec_set = self._get_pixels_along()
        self.params_dic = {}

    def _get_pixels_along(self):
        ret_set = set()
        if self.vv.distance() == 0:
            return ret_set
        vec_pair = self.vv.displacement()
        (end_v, start_v) = (self.vv.v1(), self.vv.v2())
        if vec_pair.x_int() == 0:  # vertical
            ret_set = Line._vertical_line(start_v, end_v)
        elif vec_pair.y_int() == 0:  # horizontal
            ret_set = Line._horizontal_line(start_v, end_v)
        else:
            ret_set = Line._bresenham(start_v, end_v)
        ret_set = V.get_dialated_set(ret_set,self.dialation)
        return ret_set

    def __eq__(self, other):
        return (self.vv == other.vv) and (self.dialation == other.dialation)

    def __hash__(self):
        return hash((self.vv,self.dialation))

    @staticmethod
    def _horizontal_line(start, end):
        (x0, y0) = (start.x_int(), start.y_int())
        (x1, y1) = (end.x_int(), end.y_int())
        dx = abs(x1 - x0)
        line_pixel = set()
        step = 1
        if x0 > x1:
            step = -1
        (x, y) = (x0, y0)
        for i in range(dx):
            line_pixel.add(V(x, y))
            x += step
        line_pixel.add(V(x1, y1))
        return line_pixel

    @staticmethod
    def _vertical_line(start, end):
        (x0, y0) = (start.x_int(), start.y_int())
        (x1, y1) = (end.x_int(), end.y_int())
        dy = abs(y1 - y0)
        line_pixel = set()
        step = 1
        if y0 > y1:
            step = -1
        (x, y) = (x0, y0)
        for i in range(dy):
            line_pixel.add(V(x, y))
            y += step
        line_pixel.add(V(x1, y1))
        return line_pixel

    @staticmethod
    def _bresenham(start, end):
        """
        based on
        https://github.com/encukou/bresenham/blob/master/bresenham.py
        """
        line_pixel = set()
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
            line_pixel.add(V(x0 + x*xx + y*yx, y0 + x*xy + y*yy))
            if D >= 0:
                y += 1
                D -= 2*dx
            D += 2*dy
        return line_pixel

class VMap(dict):
    """Vector map
    Bears vectors as keys for a certain value at each vectors.
    Key for this dict should be class V
    """
    # region constructor and parameters
    SORT_BY_MAX_VALUE = 0
    SORT_BY_COUNT = 1
    SORT_BY_ANGLE = 2
    SORT_BY_X     = 3
    SORT_BY_Y     = 4

    def __init__(self, list_vecs=[], fill_value=0, list_vecvalues=[]):
        """ Instantiate VMap
        Args:
            list_vecs(list):vectors list [V,V,]
            fill_value(any):filling value
        """
        super(VMap, self).__init__()
        self.sort_type = self.SORT_BY_MAX_VALUE
        self.min_vec = None
        self.max_vec = None
        self.min_vecvalue = None
        self.max_vecvalue = None
        self.diagonal = None

        if len(list_vecs) > 0:
            if isinstance(list_vecs, list):
                if not isinstance(list_vecs[0], V):
                    raise KeyError("Value key should be V")
                for vec in list_vecs:
                    self[vec] = fill_value
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

    def update(self,other,add_value=False):
        """
        Overrides update method with add/overwrite switching
        Args:
            other(VMap) : Other VMap to be added
            add_value(bool) : If True, value is added. If False, value is overwritten
        Returns:
            None
        """
        if not isinstance(other, VMap):
            raise TypeError("Argument should be VMap.")
        if add_value:
            for key in other.keys():
                if key in self:
                    self[key] = self[key] + other[key]
                else:
                    self[key] = other[key]
        else:
            for key in other.keys():
                self[key] = other[key]

    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        return self[key]

    def __setitem__(self,key,value):
        if not isinstance(key, V):
            raise KeyError("key aruguments should be V")
        super(VMap, self).__setitem__(key, value)
        if self.min_vec == None:
            self.min_vec = key
            self.max_vec = key
            self.min_vecvalue = VecValue(key, value)
            self.max_vecvalue = VecValue(key, value)
            self.diagonal = key
            return
        self.min_vec = self.min_vec.min(key)
        self.max_vec = self.max_vec.max(key)
        self.diagonal = self.max_vec - self.min_vec
        if value < self.min_vecvalue.value:
            self.min_vecvalue = VecValue(key, value)
        if value > self.max_vecvalue.value:
            self.max_vecvalue = VecValue(key, value)


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

    def vec_set(self):
        """ Vectors as a set
        Resturns:
            set: vectors as a set
        """
        return set(self.keys())

    # endregion

    # region calcs
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
            # store vectors in the intersection
            for neighbor in neighbors:
                if neighbor in self:
                    needs_check[neighbor] = self[neighbor]
                    found_vmap[neighbor] = self[neighbor]
        return found_vmap

    def extract_by_set(self, vset):
        """ Generate extracted VMap from vset and source
        Args:
            vset(set) : V keys
        Returns:
            VMap: extracted VMap
        """
        if not isinstance(vset, set):
            raise TypeError("vset should be set.")
        ret_vmap = VMap()
        for v in vset:
            ret_vmap[v] = self[v]
        return ret_vmap

    def get_offset(self):
        """ Get left-top vector (vmin)
        """
        return self.min_vec

    def get_zero_offset(self, round_vec=True):
        """ Get Vmap translated to top-left is at (0,0)
        Args:
            round_vec (bool): convert to int for pixelation if True
        Returns:
            VMap: translated VMap with left-top is at (0,0)
        """
        return self.get_translated(-self.min_vec, round_vec)

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

    def add_vmap(self, add_vmap, add_value=True):
        self.update(add_vmap, add_value=add_value)

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
            if pix in self:
                self[pix] += value
            else:
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
        return int(VMap._comb(vecs_count, 2))

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

    # TODO: we don't have to calculate all pairs !!
    # TODO: generate 'lines' that doesn't overlap each other.

    def generate_all_lines(self,min_length=0, max_length=200, allowed_dialation=0):
        """ Generate possible lines having thickness

        Returns:
            list: list of Line
        """
        ret_lines = []
        # get filtered pairs
        all_pairs = self.generate_all_pairs(min_length=min_length,max_length=max_length)

        for d in range(0, allowed_dialation):
            for vv in all_pairs:
                line = Line(vv,dialation=d)

        return ret_lines


    @staticmethod
    def _check_if_covered(checking_vv_list, vv_set):
        """ Eliminate smaller vector-pair covered by a longer vector-pair
        Note that checking_vv_set loses the popped vv (the last VV)

        Args:
            checking_vv_list (list): VV list for check. The last vv is popped and checked.
            vv_set (set): Resulting set of VV.
        Returns:
            (list, set): remaining VV list and set of VV
        """
        if len(checking_vv_list) == 0:
            return ([], vv_set, 0)
        vv1 = checking_vv_list.pop()
        #print("\nSTART: vv1:{}".format(vv1))
        if not vv1 in vv_set:
            #print("\tSKIP: {} is covered by other vv".format(vv1))
            return (checking_vv_list, vv_set)

        for vv2 in checking_vv_list:
            if not vv2 in vv_set:
                #print("\tSKIP: {} is not in vv_set.".format(vv2))
                continue
            if vv1 == vv2:
                #print("\tSKIP: vv1 == vv2")
                continue
            if vv1.covers(vv2):
                #print("\tREMOVE vv2: {} covers {}".format(vv1,vv2))
                vv_set.remove(vv2)
            elif vv2.covers(vv1):
                #print("\tREMOVE vv1: {} covers {}".format(vv2,vv1))
                if vv1 in vv_set: # sometimes vv1 no longer exists by previous check
                    vv_set.remove(vv1)
        return (checking_vv_list, vv_set)

    def generate_all_pairs(self, min_length=1.4, max_length=200):
        """ Generate all vector pairs from vmap
        Returns:
            list: list of VV
        """
        ret_vv = set()
        vecs = self.vec_list()
        n = len(vecs)
        for i in range(0, n):
            v1 = vecs[i]
            if v1 == None:
                continue
            for j in range(i + 1, n):
                v2 = vecs[j]
                vv1 = VV(v1, v2).swap_to_outward()
                distance = vv1.distance()
                if distance < min_length or max_length <  distance:
                    continue
                ret_vv.add(vv1)
        return list(ret_vv)

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
            disp1 = self.diagonal
            disp2 = other.diagonal
            return disp1.angle(quadrant12=True) < disp2.angle(quadrant12=True)
        elif self.sort_type == self.SORT_BY_COUNT:
            return len(self) < len(other)
        elif self.sort_type == self.SORT_BY_MAX_VALUE:
            return self.max_vecvalue.value < other.max_vecvalue.value
        elif self.sort_type == self.SORT_BY_X:
            return self.min_vec.x() < other.min_vec.x()
        elif self.sort_type == self.SORT_BY_Y:
            return self.min_vec.y() < other.min_vec.y()
        else:
            return True


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
        vec_range = "vector {}-{}".format(self.min_vec, self.max_vec)
        val_range = "values {}-{}".format(self.min_vecvalue.value, self.max_vecvalue.value)
        return "VMap: {} vectors, {}, {}".format(len(self), vec_range, val_range)

class VMapList(list):
    def __init__(self, *args):
        super(VMapList, self).__init__()
        (self.min_vec, self.max_vec) = (None, None)
        (self.min_vecvalue, self.max_vecvalue) = (None, None)
        self.max_diagonal = None
        if args==None or len(args)==0:
            return
        if len(args)==1:
            if isinstance(args[0],list):
                self.extend(args[0])
            else:
                self.append(args[0])
        elif len(args)>1:
            for arg in args:
                self.append(arg)
        else:
            raise TypeError()

    def append(self, vmap):
        if isVMap(vmap):
            super(VMapList,self).append(vmap)
            self._update_min_max(vmap)
        else:
            raise TypeError()

    def extend(self, vmap_list):
        if isVMapList(vmap_list):
            for vmap in vmap_list:
                self.append(vmap)
        else:
            raise TypeError()

    def _update_min_max(self, app_vmap):
        if self.min_vec == None:
            (self.min_vec, self.max_vec) = (app_vmap.min_vec, app_vmap.max_vec)
            (self.min_vecvalue, self.max_vecvalue) = (app_vmap.min_vecvalue, app_vmap.max_vecvalue)
            self.max_diagonal = app_vmap.diagonal
            return
        self.min_vec = self.min_vec.min(app_vmap.min_vec)
        self.max_vec = self.max_vec.max(app_vmap.max_vec)
        self.max_diagonal = self.max_diagonal.max(app_vmap.diagonal)
        if app_vmap.min_vecvalue.value < self.min_vecvalue.value:
            self.min_vecvalue = app_vmap.min_vecvalue
        if app_vmap.max_vecvalue.value > self.max_vecvalue.value:
            self.max_vecvalue = app_vmap.max_vecvalue

    def sort_by_count(self, reverse=False):
        self.sort_by(VMap.SORT_BY_COUNT,reverse=reverse)

    def sort_by_angle(self, reverse=False):
        self.sort_by(VMap.SORT_BY_ANGLE,reverse=reverse)

    def sort_by_max_value(self, reverse=False):
        self.sort_by(VMap.SORT_BY_MAX_VALUE,reverse=reverse)

    def sort_by(self, sort_type, reverse=False):
        for vmap in self:
            vmap.set_sort_type(sort_type)
        self.sort(reverse=reverse)

    def __str__(self):
        cnt = len(self)
        return "{} VMaps, min {} - max {}, val {} - {}, max diagonal {}".format(
            cnt,self.min_vec,self.max_vec, self.min_vecvalue.value, self.max_vecvalue.value, self.max_diagonal)

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
        super(VVCache, self).__init__()
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

class Progress:
    def __init__(self, total=0):
        self.iter_left = 0
        self.set_total(total)
    def set_total(self, total):
        self.iter_total = total
        self.iter_left = total
    def fraction(self):
        return (self.iter_total - self.iter_left) / self.iter_total
    def done(self,done_count=0,left_count=0):
        if done_count > 0:
            self.iter_left -= done_count
        elif left_count > 0:
            self.iter_left = left_count
        else:
            self.iter_left -= 1
    def has_next(self):
        return self.iter_left > 0
    def __str__(self):
        p = 100 * self.fraction()
        return ("Progress {:.2f}% ({} left in {})".format(p,self.iter_left,self.iter_total))

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

    @staticmethod
    def _tsv_format_str(items):
        s = ""
        for item in items:
            s += item + "\t"
        return s[:-1] + "\n"

class LineDetResult:
    def __init__(self, vv, sum_z, angle_step=3):
        self.vv = vv
        self.sum_z = sum_z
        self.angle_step = angle_step
        self.angle = self.vv.displacement().angle(quadrant12=True)
        self.angle_bin = self.get_angle_bin(self.angle)

    def get_angle_bin(self, angle):
        return angle - (angle % self.angle_step)

    def __str__(self):
        return "{}\t{}\t{}\t{}".format(self.vv,self.sum_z,self.angle,self.angle_bin)

class LineDetResults(list):
    def __init__(self):
        super(LineDetResults, self).__init__()
        self.max_sum_z = None
        self.avr_sum_z = None
        self.max_sum_vv = None
        self.max_sum_angle_bin = None
        self.sum_at_angles = {}
        self.cnt_at_angles = {}
        self.vv_sum_max_at_angles = {}

    def append(self, line_result):
        if isLineResult(line_result):
            super(LineDetResults,self).append(line_result)
            self._statistics(line_result)
        else:
            raise TypeError()

    def extend(self, obj):
        raise NotImplementedError()

    def _statistics(self, line_result):
        if self.max_sum_z == None:
            self.avr_sum_z = line_result.sum_z
            self.max_sum_z = line_result.sum_z
            self.max_sum_vv = line_result.vv
            self.max_sum_angle_bin = line_result.angle_bin
        else:
            self.avr_sum_z = (self.avr_sum_z + line_result.sum_z) / 2
            if self.max_sum_z < line_result.sum_z:
                self.max_sum_z = line_result.sum_z
                self.max_sum_vv = line_result.vv
                self.max_sum_angle_bin = line_result.angle_bin
        if not line_result.angle_bin in self.sum_at_angles:
            self.sum_at_angles[line_result.angle_bin] = line_result.sum_z
            self.cnt_at_angles[line_result.angle_bin] = 1
            self.vv_sum_max_at_angles[line_result.angle_bin] = line_result.vv
        else:
            self.sum_at_angles[line_result.angle_bin] += line_result.sum_z
            self.cnt_at_angles[line_result.angle_bin] += 1
            if self.vv_sum_max_at_angles[line_result.angle_bin] < line_result.sum_z:
                self.vv_sum_max_at_angles[line_result.angle_bin] = line_result.vv

    def get_csv_data_lines(self, step_angle=3):
        if len(self) == 0:
            return ''
        data_frmt = "{},{},{},{},{}\n"
        s = ''
        for i in range(0,180,step_angle):
            angle = float(i)
            if angle in self.sum_at_angles:
                s += data_frmt.format(self._vv_str(angle),angle,self.sum_at_angles[angle],
                                      self.cnt_at_angles[angle],self.avr_sum_z)
        return s

    def _vv_str(self,angle_bin):
        line = self.vv_sum_max_at_angles[angle_bin]
        (v1, v2) = line.vv
        return "{}:{}-{}:{}".format(v1.x_int(),v1.y_int(),v2.x_int(),v2.y_int())

class DialatedLineDetection:
    def __init__(self, vmap, min_length=2, max_length=200, allowed_empty=2, allowed_dialation=1):
        """
        Args:
            cutoff_angle (float): 0-180, cutoff angle difference from reference vector by degree
            ref_angle (float): initial reference angle, default: 0
        """
        self.results = {}
        self.src_vmap = vmap
        self.lines_list = vmap.generate_all_lines(min_length=min_length, max_length=max_length)
        self.vv_list = vmap.generate_all_pairs(min_length=min_length, max_length=max_length)
        self.vv_total_count = vmap.get_combinations_count()
        self.vv_count = len(self.vv_list)

    def sum_along_line(self, target_vmap_int, line_vv, dialation_to=2):
        """ Sum signal along a line VV in this map
        If there are empty pixels more than allowed_empty, returns zero.
        Args:
            target_vmap(VMap): VMap contains signals
            line_vv(VV):    A line for summing up signals
            dialation_to(int): Maximum trials of dialation repetitions.
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
        del trace_line
        del line_vv

        return signal_sum


class LineDetection:
    """
    Class to evaluate linear component in a shape from vector map
    Holds conditions for filtering
    """

    def __init__(self, vmap, min_length=2, max_length=200, allowed_empty=2, vvcache=None,exclude_covered=False):
        """
        Args:
            vmap
        """
        self.allowed_empty_pixels = allowed_empty
        if vvcache == None:
            self.vvcache = VVCache(enable_cache=False)
        else:
            self.vvcache = vvcache
        self.vvcache.register_cache('line', VV.calc_line_pixels)
        self.results = {}
        self.src_vmap = vmap
        self.vv_list = vmap.generate_all_pairs(min_length=min_length, max_length=max_length)
        self.vv_total_count = vmap.get_combinations_count()
        self.vv_count = len(self.vv_list)

    def sum_along_line(self, target_vmap_int, line_vv, dialation_to=2):
        """ Sum signal along a line VV in this map
        If there are empty pixels more than allowed_empty, returns zero.
        Args:
            target_vmap(VMap): VMap contains signals
            line_vv(VV):    A line for summing up signals
            dialation_to(int): Maximum trials of dialation repetitions.
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
        del trace_line
        del line_vv

        return signal_sum

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
    def __init__(self, vmap_list, grid_cols=25):
        self.tiled_vmap = VMap()
        self.tiling_idx = 0
        # set sort type and get max size
        self.to_be_tiled = vmap_list
        diagonal_v = self.to_be_tiled.max_diagonal
        max_w = diagonal_v.x_int() + 1
        max_h = diagonal_v.y_int() + 1
        (self.grid_w, self.grid_h) = (max_w, max_h)
        self.grid_cols = grid_cols

    def sort_by_count(self,reverse=False):
        self.to_be_tiled.sort_by_count(reverse=reverse)

    def sort_by_angle(self,reverse=False):
        self.to_be_tiled.sort_by_angle(reverse=reverse)

    def sort_by_max_value(self,reverse=False):
        self.to_be_tiled.sort_by_max_value(reverse=reverse)

    def sort_by(self,sort_type_list, reverse_type_list):
        for i in range(0,len(sort_type_list)):
            sort_type = sort_type_list[i]
            reverse_type = reverse_type_list[i]
            self.to_be_tiled.sort_by(sort_type,reverse_type)

    def tiling_add(self):
        if len(self.to_be_tiled) == 0 or self.tiling_idx>=len(self.to_be_tiled):
            return False
        left_top = self._tiling_xy()
        vmap = self.to_be_tiled[self.tiling_idx]
        vmap = vmap.get_zero_offset()
        self.tiled_vmap.add_vmap(vmap.get_translated(left_top))
        self.tiling_idx += 1
        return True

    def _tiling_xy(self):
        idx = self.tiling_idx
        col_idx = idx % self.grid_cols
        row_idx = (idx - col_idx) / self.grid_cols
        return V(col_idx * self.grid_w, row_idx * self.grid_h)

    def __str__(self):
        setting = "Tiling VMaps {}, grid_cols: {},  grid:({} x {})".format(len(self.to_be_tiled), self.grid_cols, self.grid_w, self.grid_h)
        current = ", current idx:{} next position: {}".format(self.tiling_idx, self._tiling_xy())
        return setting + current



#region static methods for this module
def isVMap(obj):
    return hasattr(obj, 'diagonal')

def isVMapList(obj):
    return hasattr(obj, 'max_diagonal')

def isLineResult(obj):
    return hasattr(obj, 'get_angle_bin')
#endregion