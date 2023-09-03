import math, random
import importlib
import os
import sys
import unittest
import time

PATH_TO_TEST = os.getenv("LINE_DET")
sys.path.append(PATH_TO_TEST)
from vectors import V, VV, Line, VMap, VMapList, VecValue, VVCache
from vectors import VMapIO, LineDetection, GridTiler, VFilter, Progress

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
        expected = {V(1.0, 2.0), V(2.0, 2.0), V(3.0, 2.0),
                    V(1.0, 3.0),            V(3.0, 3.0),
                    V(1.0, 4.0), V(2.0, 4.0), V(3.0, 4.0)}
        self.assertEqual(self.v1.neighborings(), expected)

    def test_dialation_mask(self):
        expected = {V(9,10),V(11,10),V(10,9),V(10,11),V(10,10)}
        self.assertEqual(V(10,10).dialation_mask(), expected)

    def test_get_dialated_set(self):
        vec_set = {V(0,0),V(0,1),V(0,2)}
        expected_1 = {V(0,0),V(-1,0),V(1,0),V(0,-1),
                      V(0,1),V(-1,1),V(1,1),
                      V(0,2),V(-1,2),V(1,2),V(0,3)
                    }
        self.assertEqual(V.get_dialated_set(vec_set,dialation=1),expected_1)

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

    def test_covers(self):
        v1 = V(5,5)
        v2 = V(3,3)
        v3 = V(4,3)
        v4 = V(10,10)
        self.assertTrue(v1.covers(v2))
        self.assertFalse(v1.covers(v3))
        self.assertFalse(v1.covers(v4))
        vh_1 = V(10,0)
        vh_2 = V(15,0)
        vh_3 = V(-3,0)
        self.assertTrue(vh_2.covers(vh_1))
        self.assertFalse(vh_1.covers(vh_3))
        vv_1 = V(0,-5)
        vv_2 = V(0,-8)
        vv_3 = V(0,2)
        self.assertTrue(vv_2.covers(vv_1))
        self.assertFalse(vv_2.covers(vv_3))

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
            print("vertical: {},{},{}".format(
                self.vv_v, len(vertical), vertical))
            print("horizontal: {},{},{}".format(
                self.vv_h, len(horizontal), horizontal))
            print("diagonal: {},{},{}".format(
                self.vv1, len(diagonal), diagonal))

    def test_get_pixels_along_slope(self):
        line_h = VV(V(5, 5), V(8, 21))
        line_v = VV(V(5, 5), V(21, 8))
        pixels1 = line_h.get_pixels_along()
        pixels2 = line_v.get_pixels_along()
        self.assertEqual(len(pixels1), 21 - 5 + 1)
        self.assertEqual(len(pixels2), 21 - 5 + 1)
        if PRINT:
            print(line_h, len(pixels1), pixels1)
            print(line_v, len(pixels2), pixels2)

    def test_swap(self):
        self.assertEqual(self.vv1.swap().v2(), self.vv1.v1())

    def test_swap_to_outward(self):
        self.assertEqual(self.vv1.swap_to_outward(), self.vv1)
        swapped = self.vv1.swap()
        self.assertEqual(swapped.swap_to_outward(), self.vv1)

    def test_covers(self):
        vv1 = VV(V(20,20),V(10,10))
        vv2 = VV(V(13,13), V(18,18))
        self.assertTrue(vv1.covers(vv2))
        self.assertFalse(vv2.covers(vv1))
        vv3 = VV(V(0,0),V(5,5))
        self.assertFalse(vv1.covers(vv3))
        vv4 = VV(V(-30,-20),V(-40,-30))
        vv5 = VV(V(-32,-22),V(-38,-28))
        self.assertTrue(vv4.covers(vv5))
        vv6 = VV(V(-5,0),V(0,0))
        vv7 = VV(V(-3,0),V(-4,0))
        self.assertTrue(vv6.covers(vv7))

        vvlist =  [VV(V(4.0, 4.0), V(7.0, 2.0)), VV(V(3.0, 2.0), V(6.0, 0.0)), VV(V(1.0, 4.0), V(7.0, 0.0))]
        for i in range(0,len(vvlist)):
            for j in range(i,len(vvlist)):
                vv1 = vvlist[i]
                vv2 = vvlist[j]
                vv1_cov = vv1.covers(vv2)
                vv2_cov = vv2.covers(vv1)
                if PRINT:
                    frmt = "{},{} - covering:{}"
                    print(frmt.format(vv1,vv2,vv1_cov,vv2_cov))



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


# TODO: Convert set of positions of line to Line class !
class TestLine(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLine, self).__init__(*args, **kwargs)

    def test_constructor(self):
        vv_h = VV(V(3,3),V(13,3))
        vv_v = VV(V(2,1),V(2,11))
        vv_line = VV(V(10,10),V(20,20))
        self.line_h = Line(vv_h, dialation=0)
        self.line_v = Line(vv_v, dialation=1)
        self.line = Line(vv_line, dialation=0)
        self.assertTrue(V(6,3) in self.line_h.vec_set)
        # dialated
        self.assertTrue(V(2,5) in self.line_v.vec_set)
        self.assertTrue(V(1,5) in self.line_v.vec_set)
        self.assertTrue(V(3,5) in self.line_v.vec_set)
        self.assertTrue(V(11,11) in self.line.vec_set)
        self.assertTrue(V(19,19) in self.line.vec_set)

# TODO: _eq_, _lt


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
        self.assertEqual(sort_lst[0].max_vecvalue.value, 0.8)

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

    def test_vectors(self):
        self.assertEqual(len(self.vmap1.vec_list()), 3)

    def test_round(self):
        vmap3 = self.vmap1.copy()
        vmap3[V(0.1, 0.5)] = 0.2
        self.assertEqual(vmap3[V(0.1, 0.5)], 0.2)
        vmap3_int = vmap3.round()
        self.assertEqual(vmap3_int[V(0, 1)], 0.2)
    #endregion

    def test_min_max_vectors(self):
        v1 = V(10,20)
        v2 = V(20, 5)
        v3 = V(14,19)
        vval1 = VecValue(v1, 0.5)
        vval2 = VecValue(v2, 0.8)
        vval3 = VecValue(v3, 0.7)
        vmap = VMap(list_vecvalues = [vval1,vval2,vval3])
        self.assertEqual(vmap.min_vec, V(10,5))
        self.assertEqual(vmap.max_vec, V(20,20))

    def test_min_max_values(self):
        v1 = V(10,20)
        v2 = V(20, 5)
        v3 = V(14,19)
        vval1 = VecValue(v1, 0.5)
        vval2 = VecValue(v2, 0.8)
        vval3 = VecValue(v3, 0.7)
        vmap = VMap(list_vecvalues = [vval1,vval2,vval3])
        self.assertEqual(vmap.min_vecvalue.vector, V(10, 20))
        self.assertEqual(vmap.min_vecvalue.value, 0.5)
        self.assertEqual(vmap.max_vecvalue.vector, V(20,5))
        self.assertEqual(vmap.max_vecvalue.value, 0.8)

    def test_get_neighboring_vmap_iteration(self):
        xys = [V(10,10),V(11,10),V(12,11),V(13,10),V(9,9),V(20,20),V(3,9),V(100,100),V(101,100),V(102,100),]
        vmap = VMap(list_vecs = xys, fill_value = 1.0)

        initial_count = len(vmap)
        current_count = initial_count
        while(current_count > 0):
            found_vmap = vmap._get_neighboring_from(vmap.first_vec())
            self.assertGreaterEqual(len(found_vmap), 1) # a vector should exist
            found_v0 = found_vmap.first_vec()
            self.assertFalse(found_v0 in vmap) # checked pixel should not exist in vmap
            current_count = len(vmap)
            if PRINT:
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

    def test_get_offset(self):
        line = VMap()
        line.add_line(VV(V(10,200),V(20,100)))
        offset = line.get_offset()
        self.assertEqual(offset.x(), 10)
        self.assertEqual(offset.y(), 100)

    def test_get_diagonal(self):
        line = VMap()
        line.add_line(VV(V(10,200),V(20,100)))
        self.assertEqual(line.diagonal.x(), 10)
        self.assertEqual(line.diagonal.y(), 100)


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

    def test_generate_all_pairs(self):
        v1 = V(0,0)
        v2 = V(2,2)
        v3 = V(2,0)
        vmap = VMap(list_vecs = [v1,v2,v3])
        all_vv = vmap.generate_all_pairs()
        self.assertTrue(VV(v1,v2).swap_to_outward() in all_vv)
        self.assertTrue(VV(v2,v3).swap_to_outward() in all_vv)
        self.assertTrue(VV(v3,v1).swap_to_outward() in all_vv)
        v1s = V(1,1)
        vmap2 = VMap(list_vecs = [v1,v1s,v2])
        all_vv = vmap2.generate_all_pairs()
        self.assertEqual(len(all_vv), 3)

    def test_generate_all_pairs_perf(self):
        if not RUN_PERF:
            return
        vmap = VMap()
        for i in range(0,50):
            x = random.randint(0,10)
            y = random.randint(0,10)
            vmap[V(x,y)] = 1.0
        all_count = vmap.get_combinations_count()
        if PRINT:
            print("\nAll combinations: {}".format(all_count))
        start = time.perf_counter_ns()
        all_vv = vmap.generate_all_pairs()
        end = time.perf_counter_ns()
        if PRINT:
            print("generated vv:{}".format(len(all_vv)))
        duration = (end-start)/1000000
        if PRINT:
            print("time: {}ms".format(duration))

    def test_check_if_covered(self):
        vec_list = [V(10,10),V(3,3),V(5,5),V(1,1),V(2,5)]
        vmap = VMap(list_vecs = vec_list)
        vv_list = vmap.generate_all_pairs()
        vv_set =set(vv_list)
        TestPrint("all vv_list:{}".format(len(vv_list)))

        checked_all = False
        while(not checked_all):
            pre_cnt = len(vv_set)
            (vv_list,vv_set) = VMap._check_if_covered(vv_list,vv_set)
            post_cnt = len(vv_set)
            if post_cnt < pre_cnt:
                TestPrint("Covered found, resulting count:{}".format(post_cnt))
            checked_all = (len(vv_list) == 0)
        TestPrint(vv_set)
        self.assertTrue(VV(V(1,1),V(10,10)) in vv_set)
        self.assertFalse(VV(V(3,3),V(5,5)) in vv_set)
        self.assertTrue(VV(V(1,1),V(2,5)) in vv_set)


    def test_check_if_covered_perf(self):
        if not RUN_PERF:
            return
        vmap = VMap()
        for i in range(0,50):
            x = random.randint(0,10)
            y = random.randint(0,10)
            vmap[V(x,y)] = 1.0
        all_count = vmap.get_combinations_count()
        TestPrint("\nAll combinations: {}".format(all_count))
        all_vv = vmap.generate_all_pairs()
        TestPrint("Filtered vv:{}".format(len(all_vv)))

        vv_set = set(all_vv)
        vv_list = all_vv.copy()
        start = time.perf_counter_ns()
        checked_all = False
        count = 0
        while(not checked_all):
            pre_cnt = len(vv_set)
            (vv_list,vv_set) = VMap._check_if_covered(vv_list,vv_set)
            post_cnt = len(vv_set)
            if post_cnt < pre_cnt:
                TestPrint("\t{}:Covered found, resulting count:{}".format(count,post_cnt))
            checked_all = (len(vv_list) == 0)
            count += 1
        end = time.perf_counter_ns()
        duration = (end-start)/1000000
        TestPrint("time: {}ms".format(duration))
        TestPrint("all combinations{}, filtered:{}, overlapping-excluded:{}".format(all_count,len(all_vv),len(vv_set)))

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
        lst = VMapList(self.vmap1, self.vmap2, self.vmap3)
        with self.assertRaises(TypeError):
            hoge = VMapList('hoge','hage')

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
        #TestPrint(d_landscape, angle_landscape)
        #TestPrint(d_portrait, angle_portrait)
        lst = VMapList(vmap_portrait,vmap_landscape)
        #TestPrint(lst[0], lst[1])
        lst.sort_by_angle()
        #TestPrint(lst[0], lst[1])
        self.assertTrue(V(1,1) in lst[0])


    def test_sort_by_max_value(self):
        self.vmaplist = VMapList(self.vmap3,self.vmap1,self.vmap2)
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


def get_random_VMap(count, min_v=(0,0), max_v=(10,10), val_minmax=(0.0, 1.0)):
    """ Generate random VMap
    """
    ret_vmap = VMap()
    (min_x, min_y) = min_v
    (max_x, max_y) = max_v
    (val_min, val_max) = val_minmax
    val_range = val_max - val_min
    while(len(ret_vmap)<count):
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
        val = val_min + random.randint(0,100) * val_range / 100
        ret_vmap[V(x,y)] = val
    return ret_vmap

class TestLineDetection(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLineDetection, self).__init__(*args, **kwargs)

    def test_constructor(self):
        vmap = get_random_VMap(10)
        vvcache = VVCache(enable_cache=True)
        ld = LineDetection(vmap,min_length=0, max_length = 200,allowed_empty=2,vvcache=vvcache)
        self.assertEqual(ld.vv_count, 10 * 9 / 2)

    def test_sum_along_line(self):
        test_map = VMap()
        test_map.add_line(VV(V(10, 10),V(20, 20)), value = 1.0)
        test_map[V(10,10)] = 2.0
        test_map[V(10,20)] = 3.0
        ld = LineDetection(test_map, allowed_empty = 3)
        sum_diagonal = ld.sum_along_line(test_map, VV(V(10,10),V(20,20)))
        sum_horizontal = ld.sum_along_line(test_map, VV(V(10,10),V(20,10)))
        sum_vertical = ld.sum_along_line(test_map, VV(V(10,10),V(10,20)))
        sum_none = ld.sum_along_line(test_map, VV(V(15,10),V(20,15)))

        self.assertEqual(sum_diagonal, 12) # continuous 2.0 at V(10,10) and 1 x 10
        self.assertEqual(sum_horizontal, 0) # 2.0 at V(10,10)
        self.assertEqual(sum_vertical, 0) # 2.0 at V(10,10), many empty by V(10,20)
        self.assertEqual(sum_none, 0)


    def test_drawings(self):
        PRINT = True
        vmap = VMap()
        line = VV(V(10,10),V(40,20))
        pixels = line.get_pixels_along()
        vmap.add_line(line,  value = 1.0)
        self.assertTrue(V(10,10) in vmap)
        self.assertTrue(V(25,15) in vmap)
        self.assertTrue(V(40,20) in vmap)

        line_rev = line.swap()
        vmap = VMap()
        vmap.add_line(line_rev, value = 1.0)
        vmap = vmap.round()
        self.assertTrue(V(10,10) in vmap)
        self.assertTrue(V(25,15) in vmap)
        self.assertTrue(V(40,20) in vmap)

        vmap = VMap()
        vmap.add_rect(VV(V(10,10),V(20,20)))
        vmap = vmap.round()
        self.assertTrue(V(15,10) in vmap)
        self.assertTrue(V(20,15) in vmap)
        self.assertTrue(V(15,20) in vmap)
        self.assertTrue(V(10,15) in vmap)

    def test_str(self):
        vmap = VMap(list_vecs = [V(1,1),V(2,2)], fill_value = 0.5)
        str = vmap.str_list()

class TestVFilter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestVFilter, self).__init__(*args, **kwargs)

    def test_by_count(self):
        # VMap test, only vmap3 should be excluded.
        vmap3 = get_random_VMap(3)
        vmap10 = get_random_VMap(10)
        vmap20 = get_random_VMap(20)
        filter_cnt = VFilter([vmap10,vmap20,vmap3],5,max_value=20)
        self.assertTrue(filter_cnt.by_count())
        self.assertTrue(filter_cnt.by_count())
        self.assertTrue(filter_cnt.by_count())
        self.assertFalse(filter_cnt.by_count())
        self.assertEqual(len(filter_cnt.filtered), 2)

        # VVMap test, large_vvmap should be excluded.
        vmapLeftTop = get_random_VMap(3,min_v=(0,0),max_v=(3,3))
        long_vmap = get_random_VMap(3,min_v=(50,50),max_v=(55,55))
        long_vmap.add_vmap(vmapLeftTop)
        small_vmap = get_random_VMap(10,min_v=(20,20),max_v=(30,30))
        #TestPrint(long_vvmap)
        #TestPrint(small_vvmap)
#        filter_disp = VFilter([long_vvmap,small_vvmap],30,max_value=100)
###        self.assertTrue(filter_disp.by_count())
   #     self.assertTrue(filter_disp.by_count())
#        self.assertFalse(filter_disp.by_count())
#        self.assertEqual(len(filter_disp.filtered), 1)


class TestGridTiler(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGridTiler, self).__init__(*args, **kwargs)

    def test_constructor(self):
        # SORT_BY_COUNT test
        vmaps = VMapList()
        for i in range(0,10):
            (x, y) = (random.randint(0,100), random.randint(0,100))
            vmap = VMap()
            vmap.add_line(VV(V(x, y), V(x+i, y+i)))
            vmaps.append(vmap)
        tiler_cnt = GridTiler(vmaps)
        tiler_cnt.sort_by_count(reverse=True)
        self.assertEqual(len(tiler_cnt.to_be_tiled[0]), 10)
        # SORT_BY_ANGLE test
        vmap1 = VMap(list_vecs = [V(0,10),V(20,0)])
        vmap2 = VMap(list_vecs = [V(-10,-20),V(0,0),V(-5,-5)])
        tiler_angle = GridTiler(VMapList(vmap1,vmap2))
        tiler_angle.sort_by_angle()
        self.assertEqual(len(tiler_angle.to_be_tiled[0]),2)
        tiler_angle = GridTiler(VMapList(vmap2,vmap1))
        tiler_angle.sort_by_angle(reverse=True)
        self.assertEqual(len(tiler_angle.to_be_tiled[0]),3)
        # SORT_BY_MAX_VALUE test
        vmap1 = VMap(list_vecvalues=[VecValue(V(1,2),3),VecValue(V(2,2),5)])
        vmap2 = VMap(list_vecvalues=[VecValue(V(5,6),10),VecValue(V(7,8),20)])
        tiler_max = GridTiler(VMapList(vmap1,vmap2))
        tiler_max.sort_by_max_value(reverse=True)
        vmap_with_max_value = tiler_max.to_be_tiled[0]
        self.assertEqual(vmap_with_max_value.max_vecvalue.value, 20)

    def test_tiling_add(self):
        vmap1 = VMap(list_vecs = [V(0,11),V(13,0)])
        vmap2 = VMap(list_vecs = [V(-10,0),V(-10,-10),V(-5,-5)])
        vmap3 = VMap(list_vecs = [V(2,3),V(3,4)])
        vmap_list = VMapList(vmap1,vmap2,vmap3)
        #TestPrint(vmap_list)
        tiler_angle = GridTiler(vmap_list)
        tiler_angle.sort_by_angle()
        #TestPrint(tiler_angle)
        self.assertEqual(tiler_angle.grid_w, 13+1)
        self.assertEqual(tiler_angle.grid_h, 11+1)
        self.assertEqual(tiler_angle.tiling_idx, 0)
        tiler_angle.tiling_add()
        self.assertEqual(tiler_angle.tiling_idx, 1)
        next_xy = tiler_angle._tiling_xy()
        #TestPrint(tiler_angle)
        self.assertEqual(next_xy.x(), tiler_angle.grid_w)
        self.assertTrue(tiler_angle.tiling_add())
        #TestPrint(tiler_angle)
        self.assertTrue(tiler_angle.tiling_add())
        #TestPrint(tiler_angle)
        self.assertFalse(tiler_angle.tiling_add())


class TestProgress(unittest.TestCase):
    def test_increment(self):
        a = [1,2,3,4]
        b = []
        progress = Progress(total=len(a))
        self.assertEqual(progress.fraction(), 0)
        idx = 0
        while(progress.has_next()):
            TestPrint(progress)
            b.append(a[idx])
            self.assertTrue(progress.has_next())
            progress.done()
            idx += 1
        TestPrint(progress)
        self.assertEqual(len(b),4)
        self.assertFalse(progress.has_next())

    def test_left_count(self):
        a = [11,11,22,22,33,33]
        progress = Progress(total=len(a))
        while(progress.has_next()):
            TestPrint(progress)
            _ = a.pop()
            progress.done(left_count=len(a))
        TestPrint(progress)
        self.assertEqual(len(a),0)
        self.assertFalse(progress.has_next())
        self.assertEqual(progress.fraction(),1.0)

def TestPrint(str):
    global PRINT
    if PRINT:
        print(str)

if __name__ == '__main__':
    global PRINT, RUN_PERF
    PRINT = False
    RUN_PERF = False
    unittest.main()
