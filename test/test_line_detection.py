import math, importlib, os, sys, time, unittest, random

PATH_TO_TEST = os.getenv("LINE_DET")
sys.path.append(PATH_TO_TEST)
from vectors import V, VV, VecValue, VMap, VMapList, VMapIO
from line_detection import LineDetection, GridTiler, VFilter


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
        ld = LineDetection(vmap,min_length=0, max_length = 200,allowed_empty=2,enable_cache=True)
        self.assertEqual(ld.vv_count, 10 * 9 / 2)

    def test_sum_along_line(self):
        test_map = VMap()
        test_map.add_line(VV(V(10, 10),V(20, 20)), value = 1.0)
        test_map.set_value(V(10,10), 2.0)
        test_map.set_value(V(10,20), 3.0)
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
        pixels.sort()
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
        #print(long_vvmap)
        #print(small_vvmap)
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
        vmaps = []
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
        tiler_angle = GridTiler([vmap1,vmap2])
        tiler_angle.sort_by_angle()
        self.assertEqual(len(tiler_angle.to_be_tiled[0]),2)
        tiler_angle = GridTiler([vmap2,vmap1])
        tiler_angle.sort_by_angle(reverse=True)
        self.assertEqual(len(tiler_angle.to_be_tiled[0]),3)
        # SORT_BY_MAX_VALUE test
        vmap1 = VMap(list_vecvalues=[VecValue(V(1,2),3),VecValue(V(2,2),5)])
        vmap2 = VMap(list_vecvalues=[VecValue(V(5,6),10),VecValue(V(7,8),20)])
        tiler_max = GridTiler([vmap1,vmap2])
        tiler_max.sort_by_max_value(reverse=True)
        vmap_with_max_value = tiler_max.to_be_tiled[0]
        (min_val,max_val) = vmap_with_max_value.get_min_max_values()
        self.assertEqual(max_val.value, 20)

    def test_tiling_add(self):
        vmap1 = VMap(list_vecs = [V(0,11),V(13,0)])
        vmap2 = VMap(list_vecs = [V(-10,0),V(-10,-10),V(-5,-5)])
        vmap3 = VMap(list_vecs = [V(2,3),V(3,4)])
        tiler_angle = GridTiler([vmap1,vmap2,vmap3])
        tiler_angle.sort_by_angle()
        #print(tiler_angle)
        self.assertEqual(tiler_angle.grid_w, 14)
        self.assertEqual(tiler_angle.grid_h, 12)
        self.assertEqual(tiler_angle.tiling_idx, 0)
        tiler_angle.tiling_add()
        self.assertEqual(tiler_angle.tiling_idx, 1)
        next_xy = tiler_angle._tiling_xy()
        self.assertEqual(next_xy.x(), 14)
        #print(tiler_angle)

if __name__=='__main__':
    unittest.main()
