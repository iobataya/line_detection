import math, random
import importlib
import os
import sys
import unittest
import time

PATH_TO_TEST = os.getenv("LINE_DET")
sys.path.append(PATH_TO_TEST)
from vectors import V, VV, VMap, VMapList, VecValue, VVCache, VMapIO

class TestV(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestV, self).__init__(*args, **kwargs)
        self.v1 = V(2, 3)
        self.v2 = V(4, 5.0)
        self.v3 = V(5.0000000001, 10.0000000005)
        self.v4 = V(3, -5)
        self.v5 = V(-4, -6)
        self.v6 = V(math.sqrt(3)/2, 0.5)

    def test_constructor(self):
        self.assertTrue(isinstance(self.v1.x(), float))
        self.assertTrue(isinstance(self.v1.y(), float))

    # region primitive calcs
    def test_area(self):
        area2 = 4.0 * 5.0
        self.assertEqual(self.v2.area(), area2)

    def test_cross(self):
        cp = 2.0 * 5.0 - 3.0 * 4.0
        self.assertEqual(self.v1.cross(self.v2), cp)

    def test_magnitude(self):
        self.assertEqual(self.v1.magnitude(), math.sqrt(2.0 * 2.0 + 3.0 * 3.0))
        self.assertEqual(self.v2.magnitude(), math.sqrt(4.0 * 4.0 + 5.0 * 5.0))

    def test_max_min(self):
        self.assertEqual(self.v1.max(self.v5), V(2.0, 3.0))
        self.assertEqual(self.v4.min(self.v5), V(-4.0, -6.0))

    def test_neighborings(self):
        expected = [V(1.0, 2.0), V(2.0, 2.0), V(3.0, 2.0),
                    V(1.0, 3.0),            V(3.0, 3.0),
                    V(1.0, 4.0), V(2.0, 4.0), V(3.0, 4.0)]
        self.assertEqual(self.v1.neighborings(), expected)

    def test_rotate(self):
        self.assertEqual(self.v1.rotate(math.pi / 2), V(-3, 2))

    def test_to_int(self):
        self.assertTrue(isinstance(self.v2.x_int(), int))
        self.assertTrue(isinstance(self.v2.y_int(), int))
        v_frac = V(0.4, 0.5)
        self.assertEqual(v_frac.round().x(), 0)
        self.assertEqual(v_frac.round().y(), 1)

    def test_to_quad(self):
        self.assertEqual(self.v1.to_quad12(), V(2.0, 3.0))
        self.assertEqual(self.v4.to_quad12(), V(-3.0, 5.0))

    def test_normalized(self):
        mx1 = 2.0 / math.sqrt(2.0 * 2.0 + 3.0 * 3.0)
        my1 = 3.0 / math.sqrt(2.0 * 2.0 + 3.0 * 3.0)
        self.assertEqual(self.v1.normalized(), V(mx1, my1))

    def test_cross_norm(self):
        v1_norm = self.v1.normalized()
        v2_norm = self.v2.normalized()
        self.assertEqual(v1_norm.cross(v2_norm))
    # endregion

    # region calcs
    def test_angle(self):
        self.assertEqual(V(1, 0).angle(), 0)
        self.assertEqual(V(0, 1).angle(), 90)
        self.assertEqual(V(-1, 1).angle(), 135)
        self.assertEqual(V(-1, -1).angle(quadrant12=True), 45)

    def test_normalized(self):
        m = self.v2.magnitude()
        self.assertEqual(self.v2.normalized(), V(4.0 / m, 5.0 / m))

    def test_cross_norm(self):
        m1 = self.v1.magnitude()
        norm1 = V(2.0 / m1, 3.0 / m1)
        m5 = self.v5.magnitude()
        norm5 = V(-4.0 / m5, -6.0 / m5)
        self.assertEqual(self.v1.cross_norm(self.v5), norm1.cross(norm5))

    def test_radian(self):
        rad30 = round(math.pi / 6.0, 10)
        self.assertEqual(round(self.v6.radian(), 10), rad30)
        quad34 = self.v6.rotate(math.pi)
        self.assertEqual(round(quad34.radian(quadrant12=True), 10), rad30)
    # endregion

    def test_operators(self):
        self.assertEqual(self.v1 + self.v2, V(6, 8))
        self.assertEqual(self.v1 - self.v2, V(-2, -2))
        self.assertEqual(self.v1 * self.v2, V(8, 15))
        self.assertEqual(self.v1 / self.v2, V(2.0/4.0, 3.0/5.0))
        vecs = [self.v1, self.v2]
        self.assertEqual(str(self.v1), "V(2.0, 3.0)")
        self.assertEqual(str(vecs), "[V(2.0, 3.0), V(4.0, 5.0)]")
        self.assertEqual(-self.v1, V(-2,-3))

    def test_serialize(self):
        v1 = V(1.2,2.0)
        self.assertEqual(v1.serialize(), "1.2,2")
        v2 = V(-45,-2.2)
        self.assertEqual(v2.serialize(), "-45,-2.2")
        v3 = V.deserialize("-20.2, 3")
        self.assertEqual(v3, V(-20.2,3.0))


class TestVV(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestVV, self).__init__(*args, **kwargs)
        self.vv1 = VV(V(10, 20), V(20, 30))
        self.vv_h = VV(V(20, 3), V(-5, 3))
        self.vv_v = VV(V(15, -10), V(15, 5))

    def test_constructor(self):
        self.assertEqual(self.vv1.v1(), V(10, 20))
        self.assertEqual(self.vv1.x1(), 10.0)
        self.assertEqual(self.vv1.y1(), 20.0)
        self.assertEqual(self.vv1.v2(), V(20, 30))
        self.assertEqual(self.vv1.x2(), 20.0)
        self.assertEqual(self.vv1.y2(), 30.0)
        minus = -self.vv1
        self.assertEqual(minus.v1(), V(20,30))
        with self.assertRaises(ValueError):
            tuples = VV((10, 20), (20, 30))

    def test_displacement(self):
        self.assertEqual(self.vv1.displacement(), V(-10.0, -10.0))

    def test_distance(self):
        d = math.sqrt((20.0 - 10.0) ** 2 + (30.0 - 20.0) ** 2)
        self.assertEqual(self.vv1.distance(), d)

    def test_get_pixels_along(self):
        PRINT = False
        vertical = self.vv_v.get_pixels_along()
        self.assertTrue(V(15, -10).round() in vertical)
        self.assertTrue(V(15,  0).round() in vertical)
        self.assertTrue(V(15,  5).round() in vertical)
        horizontal = self.vv_h.get_pixels_along()
        self.assertTrue(V(20, 3).round() in horizontal)
        self.assertTrue(V(0, 3).round() in horizontal)
        self.assertTrue(V(-5, 3).round() in horizontal)
        diagonal = self.vv1.get_pixels_along()
        for x in range(10, 21):
            self.assertTrue(V(x, x + 10).round() in diagonal)
        if PRINT:
            horizontal.sort()
            vertical.sort()
            diagonal.sort()
            print("vertical: {},{},{}".format(
                self.vv_v, len(vertical), vertical))
            print("horizontal: {},{},{}".format(
                self.vv_h, len(horizontal), horizontal))
            print("diagonal: {},{},{}".format(
                self.vv1, len(diagonal), diagonal))

    def test_get_pixels_along_slope(self):
        PRINT = False
        line_h = VV(V(5, 5), V(8, 21))
        line_v = VV(V(5, 5), V(21, 8))
        pixels1 = line_h.get_pixels_along()
        pixels2 = line_v.get_pixels_along()
        self.assertEqual(len(pixels1), 21 - 5 + 1)
        self.assertEqual(len(pixels2), 21 - 5 + 1)
        if PRINT:
            pixels1.sort()
            pixels2.sort()
            print(line_h, len(pixels1), pixels1)
            print(line_v, len(pixels2), pixels2)

    def test_swap(self):
        self.assertEqual(self.vv1.swap().v2(), self.vv1.v1())

    def test_swap_to_outward(self):
        self.assertEqual(self.vv1.swap_to_outward(), self.vv1)
        swapped = self.vv1.swap()
        self.assertEqual(swapped.swap_to_outward(), self.vv1)


class TestVecValue(unittest.TestCase):
    def test_vecvalue(self):
        vvalue1 = VecValue(V(3, 4), 0.5)
        self.assertEqual(vvalue1.vector.x(), 3.0)
        self.assertEqual(vvalue1.value, 0.5)
        with self.assertRaises(ValueError):
            non_vec = VecValue((20, 20), 1.5)

    def test_serialize(self):
        vval = VecValue(V(1.0,4),-5e-3)
        self.assertEqual(vval.serialize(), "1,4:-0.005")
        s = " 103 ,-256.256 : 10.2"
        self.assertEqual(VecValue.deserialize(s), VecValue(V(103,-256.256),10.2))

class TestVMap(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestVMap, self).__init__(*args, **kwargs)
        v1 = V(10,20)
        v2 = V(20, 5)
        v3 = V(14,19)
        vval1 = VecValue(v1, 0.5)
        vval2 = VecValue(v2, 0.8)
        vval3 = VecValue(v3, 0.7)
        self.vvals = [v1, v2, v3]
        self.vecvals = [vval1, vval2, vval3]
        self.vecvals2 = [vval3, vval1, vval2]
        self.dict = VMap()
        self.dict[v1] = 0.5
        self.dict[v2] = 0.8
        self.dict[v3] = 0.7
        self.vmap1 = self.dict

    #region constructor and parameters
    def test_constructor(self):
        vmap_from_list = VMap(list_vecs = self.vvals, fill_value = 0.5)
        self.assertEqual(vmap_from_list[V(10,20)], 0.5)

        vmap_from_vecval = VMap(list_vecvalues = self.vecvals)
        self.assertEqual(vmap_from_vecval[V(10,20)], 0.5)

        self.assertEqual(self.dict[V(14,19)], 0.7)

    def test_sort(self):
        sort1 = VMap(list_vecvalues=[VecValue(V(1,2),0.1),
                                     VecValue(V(10,1),0.2),
                                     VecValue(V(10,5),0.5),
                                     VecValue(V(2,8),0.3)])
        sort2 = VMap(list_vecvalues=[VecValue(V(5,0),0.8),
                                     VecValue(V(10,5),0.3),
                                     VecValue(V(10,20),0.1)])
        sort3 = VMap(list_vecvalues=[VecValue(V(0,1),0.6),
                                     VecValue(V(10,0),0.1)])
        sort_lst = [sort1,sort2,sort3]
        sort_lst.sort(reverse=True) # should be sorted by max value
        (_, val_max) = sort_lst[0].get_min_max_values()
        self.assertEqual(val_max.value, 0.8)

        sort_lst = [sort1,sort2,sort3]
        for lst in sort_lst:
            lst.set_sort_type(VMap.SORT_BY_COUNT)
        sort_lst.sort() # should be sorted by count
        self.assertEqual(len(sort_lst[0]),2)

        sort_lst = [sort1,sort2,sort3]
        for lst in sort_lst:
            lst.set_sort_type(VMap.SORT_BY_ANGLE)
        sort_lst.sort() # should be sorted by angle in quadrant12
        self.assertTrue(V(10,0) in sort_lst[0])

    def test_copy(self):
        copied = self.vmap1.copy()
        self.assertFalse(self.vmap1 is copied)
        self.assertEqual(len(copied), 3)
        self.assertEqual(copied[V(20,5)], 0.8)

    def test_get_value(self):
        self.assertEqual(self.vmap1.get_value(V(14,19)), 0.7)

    def test_set_value(self):
        vmap2 = self.vmap1.copy()
        vmap2.set_value(V(10,20), 1.0)
        self.assertEqual(vmap2[V(10,20)], 1.0)
        with self.assertRaises(KeyError):
            vmap2.set_value((10,20), 1.0)

    def test_vectors(self):
        self.assertEqual(len(self.vmap1.vec_list()), 3)

    def test_round(self):
        vmap3 = self.vmap1.copy()
        vmap3.set_value(V(0.1, 0.5), 0.2)
        self.assertEqual(vmap3[V(0.1, 0.5)], 0.2)
        vmap3_int = vmap3.round()
        self.assertEqual(vmap3_int[V(0, 1)], 0.2)
    #endregion

    def test_min_max_vectors(self):
        vmin_max = self.vmap1.get_min_max_vectors()
        (min_vec, max_vec) = vmin_max
        self.assertEqual(min_vec, V(10,5))
        self.assertEqual(max_vec, V(20,20))

    def test_min_max_values(self):
        vmin_max_val = self.vmap1.get_min_max_values()
        (min_vval, max_vval) = vmin_max_val
        self.assertEqual(min_vval.vector, V(10,20))
        self.assertEqual(min_vval.value, 0.5)
        self.assertEqual(max_vval.vector, V(20,5))
        self.assertEqual(max_vval.value, 0.8)

    def test_get_neighboring_vmap_iteration(self):
        PRINT_PROGRESS = False
        xys     = [V(10,10),V(11,10),V(12,11),V(13,10),V(9,9),V(20,20),V(3,9),V(100,100),V(101,100),V(102,100),]
        vmap = VMap(list_vecs = xys, fill_value = 1.0)

        initial_count = len(vmap)
        current_count = initial_count
        while(current_count > 0):
            found_vmap = vmap._get_neighboring_from(vmap.first_vec())
            self.assertGreaterEqual(len(found_vmap), 1) # a vector should exist
            found_v0 = found_vmap.first_vec()
            self.assertFalse(found_v0 in vmap) # checked pixel should not exist in vmap
            current_count = len(vmap)
            if PRINT_PROGRESS:
                progress = 100 - 100 * current_count / initial_count
                print("Progress: {}%, {} found.".format(progress, len(found_vmap)))

    def test_get_neighboring_vmaps(self):
        xys     = [V(10,10),V(11,10),V(12,11),V(13,10),V(9,9),V(20,20),V(3,9),V(100,100),V(101,100),V(102,100),]
        exp_xys = [V(10,10),V(11,10),V(12,11),V(13,10),V(9,9)]

        vmap = VMap(list_vecs = xys, fill_value = 1.0)
        self.assertEqual(vmap[V(10,10)], 1.0)

        ex_vmap = VMap(list_vecs = exp_xys, fill_value = 1.0)
        result = vmap._get_neighboring_from(V(10,10))
        for vec in result.keys():
            self.assertTrue(vec in ex_vmap)
        self.assertFalse(V(20,20) in result)
        self.assertFalse(V(3,9) in result)
        result = vmap._get_neighboring_from(V(100,100))
        self.assertEqual(len(result),3)

        vmap = VMap(list_vecs = xys, fill_value = 1.0)
        neighboring = vmap.get_neighboring_vmaps(min_vectors = 2)
        self.assertEqual(len(neighboring),2)

    def test_get_offset(self):
        line = VMap()
        line.add_line(VV(V(10,200),V(20,100)))
        offset = line.get_offset()
        self.assertEqual(offset.x(), 10)
        self.assertEqual(offset.y(), 100)

    def test_get_diagonal(self):
        line = VMap()
        line.add_line(VV(V(10,200),V(20,100)))
        diagonal = line.get_diagonal()
        self.assertEqual(diagonal.x(), 10)
        self.assertEqual(diagonal.y(), 100)


    def test_get_translated(self):
        line = VMap()
        line.add_line(VV(V(9.9,20.1),V(20.1,9.9)), value = 1.0)
        offset = line.get_offset()
        line_origined = line.get_translated(V(-offset.x(),-offset.y()), round_vec = True)
        self.assertEqual(line_origined[V(0,10)], 1.0)

    def test_get_rotated(self):
        rotated90 = self.vmap1.get_rotated(math.pi/2)
        rotated90_int = rotated90.round()
        self.assertTrue(V(-20,10) in rotated90_int)
        self.assertTrue(V(-5,20) in rotated90_int)

        vmap = VMap()
        vmap.add_rect(VV(V(0, 0),V(10, 10)))
        rot60 = vmap.get_rotated(math.pi / 3, center = V(0, 0))
        top_right_V = V(10, 0).rotate(math.pi / 3)
        self.assertTrue(top_right_V in rot60)

    def test_serialize(self):
        vval1 = VecValue(V(1.0,-2.5),-0.6)
        vval2 = VecValue(V(10,20),0.2)
        vval3 = VecValue(V(-5,-6),0)
        vmap = VMap(list_vecvalues = [vval1,vval2,vval3])
        expected_str = "1,-2.5:-0.6;10,20:0.2;-5,-6:0"
        self.assertEqual(vmap.serialize(), expected_str)

        input_str = "20.2 , -5:1e-3 ; -5,-6:10 "
        vmap = VMap.deserialize(input_str)
        self.assertEqual(vmap[V(20.2,-5)], 0.001)
        self.assertEqual(vmap[V(-5,-6)], 10)

    def test_eq(self):
        vmap1 = VMap(list_vecvalues=self.vecvals)
        vmap2 = VMap(list_vecvalues=self.vecvals2)
        self.assertTrue(vmap1 == vmap2)

    #region Drawing
    @staticmethod
    def add_circle(vmap, radius, origin=V(0, 0), value=1.0):
        """Add a circle
        Args:
            radius(float):  radius of circle
            origin(V):      center of circle
            value(float):   filling value
        """
        length = 2 * math.pi * radius
        angle_step = 1 / length
        vec = V(radius, 0)
        angle = 0
        while(angle < 2 * math.pi):
            vec_key = vec + origin
            vec_key = vec_key.round()
            vmap[vec_key] = value
            vec = vec.rotate(angle)
            angle += angle_step
    @staticmethod
    def add_triangle(vmap, length, origin=V(0, 0), value=1.0):
        """Add a triangle
        Args:
            length(float):  length of sides
            origin(V):      first position
            value(float):   filling value
        """
        pt1 = V(0, 0)
        pt2 = V(length, 0)
        pt3 = V(length / 2, length * math.sin(math.pi/3))
        vv1 = VV(pt1 + origin, pt2 + origin)
        vv2 = VV(pt2 + origin, pt3 + origin)
        vv3 = VV(pt3 + origin, pt1 + origin)
        vmap.add_line(vv1, value)
        vmap.add_line(vv2, value)
        vmap.add_line(vv3, value)

    @staticmethod
    def add_sinuidal(vmap, amplitude, width, offset=V(0, 0), value=1.0):
        step = math.pi * width / 16.0
        for i in range(0, 15):
            x1 = width * i / 16.0
            y1 = amplitude * math.sin(step * i)
            x2 = width * (i + 1.0) / 16.0
            y2 = amplitude * math.sin(step * (i + 1.0))
            vmap.add_line(VV(V(x1, y1), V(x2, y2)))
    #endregion

class TestVMapList(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestVMapList, self).__init__(*args, **kwargs)
        vval1 = VecValue(V(10,20), 0.5)
        vval2 = VecValue(V(20, 5), 0.8)
        vval3 = VecValue(V(14,19), 0.7)
        vval4 = VecValue(V( 3, 5), 1.0)
        self.vmap1 = VMap(list_vecvalues = [vval1,vval2])
        self.vmap2 = VMap(list_vecvalues = [vval2,vval3,vval4])
        self.vmap3 = VMap(list_vecvalues = [vval1,vval2,vval3,vval4])


    def test_constructor(self):
        lst = VMapList([self.vmap1, self.vmap2, self.vmap3])
        with self.assertRaises(TypeError):
            hoge = VMapList(['hoge','hage'])

    def test_sort_by_count(self):
        self.vmaplist = VMapList(self.vmap3,self.vmap1,self.vmap2)
        self.vmaplist.sort_by_count()
        self.assertEqual(len(self.vmaplist[0]), 2)
        self.assertEqual(len(self.vmaplist[2]), 4)

    def test_sort_by_angle(self):
        vval1 = VecValue(V(0,0),1.0)
        vval2 = VecValue(V(1,1),1.0)
        vval3 = VecValue(V(0.5,2.0),1.0)
        vmap_landscape = VMap(list_vecvalues = [vval1,vval2]) # 45 deg
        vmap_portrait = VMap(list_vecvalues = [vval1,vval3]) # 76 deg
        #d_landscape = vmap_landscape.get_diagonal()
        #angle_landscape = d_landscape.angle(quadrant12=True)
        #d_portrait = vmap_portrait.get_diagonal()
        #angle_portrait = d_portrait.angle(quadrant12=True)
        #print(d_landscape, angle_landscape)
        #print(d_portrait, angle_portrait)
        lst = VMapList([vmap_portrait,vmap_landscape])
        #print(lst[0], lst[1])
        lst.sort_by_angle()
        #print(lst[0], lst[1])
        self.assertTrue(V(1,1) in lst[0])


    def test_sort_by_max_value(self):
        self.vmaplist = VMapList([self.vmap3,self.vmap1,self.vmap2])
        self.vmaplist.sort_by_max_value()
        self.assertTrue(V(3,5) in self.vmaplist[2])

class TestVMapIO(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestVMapIO, self).__init__(*args, **kwargs)
        self.folder = os.path.join(os.getenv("LINE_DET"),'test')
        vmap1 = VMap(list_vecs = [V(0,11),V(13,0)], fill_value = 0.5)
        vmap2 = VMap(list_vecs = [V(-10,0),V(-10,-10),V(-5,-5)], fill_value = 0.2)
        vmap3 = VMap(list_vecs = [V(2,3),V(3,4)], fill_value = 1.1)
        self.vmap_list = VMapList()
        self.vmap_list.append(vmap1)
        self.vmap_list.append(vmap2)
        self.vmap_list.append(vmap3)
        self.vmap_std_list = [vmap1,vmap2,vmap3]

    def test_constructor(self):
        self.assertEqual(self.vmap_list[0][V(0,11)], 0.5)

    def test_write(self):
        io = VMapIO()
        path = os.path.join(self.folder,'test_VMapList_write.vmap')
        io.open(path,'w',self.vmap_list)
        self.assertTrue(io.write_vmap())
        self.assertTrue(io.write_vmap())
        self.assertTrue(io.write_vmap())
        self.assertFalse(io.write_vmap())
        io.close()
        path = os.path.join(self.folder,'test_list_write.vmap')
        io.open(path,'w',self.vmap_std_list)
        self.assertTrue(io.write_vmap())
        self.assertTrue(io.write_vmap())
        self.assertTrue(io.write_vmap())
        self.assertFalse(io.write_vmap())
        io.close()


    def test_read(self):
        io = VMapIO()
        path = os.path.join(self.folder,'test_VMapList_write.vmap')
        io.open(path,'r')
        self.assertTrue(io.read_vmap())
        self.assertTrue(io.read_vmap())
        self.assertTrue(io.read_vmap())
        self.assertFalse(io.read_vmap())
        io.close()
        result_vmap = io.vmap_list
        self.assertEqual(result_vmap[1][V(-10,-10)], 0.2)
        self.assertEqual(result_vmap[2][V(2,3)], 1.1)

if __name__ == '__main__':
    unittest.main()
