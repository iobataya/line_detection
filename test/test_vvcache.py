import math, random
import importlib
import os
import sys
import unittest
import time

PATH_TO_TEST = os.getenv("LINE_DET")
sys.path.append(PATH_TO_TEST)
from vectors import V, VV, VMap, VMapList, VecValue, VVCache, VMapIO

class TestVVCache(unittest.TestCase):
    """
    This VVCache class stores any values at VV key.
    Values are calculated by registered static method.
    It's effective only for heavy process like generating pixels along a line.
    """
    def __init__(self, *args, **kwargs):
        super(TestVVCache, self).__init__(*args, **kwargs)
        self.vv1 = VV(V(1, 1), V(5, 5))
        self.len1 = self.vv1.distance()
        self.vv2 = VV(V(2, 2), V(5, 5))
        self.len2 = self.vv2.distance()
        self.vv3 = VV(V(3, 3), V(5, 5))
        self.len3 = self.vv3.distance()

    def test_register(self):
        self.vvcache = VVCache()
        self.vvcache.register_cache('length', VV.calc_distance)
        self.assertTrue('length' in self.vvcache)
        self.assertTrue('length' in self.vvcache.func)

    def test_get_value(self):
        self.vvcache = VVCache()
        self.vvcache.register_cache('len', VV.calc_distance)
        self.vvcache.register_cache('cp', V.calc_cross)
        self.vvcache.register_cache('line', VV.calc_line_pixels)
        self.vvcache['len'][self.vv1] = self.len1
        self.vvcache['len'][self.vv2] = self.len2
        self.assertEqual(self.vvcache.get_value('len', self.vv1), self.len1)
        self.assertEqual(self.vvcache.get_value('len', self.vv2), self.len2)
        self.assertEqual(self.vvcache.get_value('len', self.vv3), self.len3)
        v1 = V(1,1)
        v2 = V(1,0)
        cp1_2 = round(self.vvcache.get_value('cp', (v1,v2)), 10)
        exp_cp1_2 = round(abs(-math.sqrt(2.0)/2.0), 10)
        self.assertEqual(cp1_2, exp_cp1_2)
        line = self.vvcache.get_value('line',self.vv1)
        self.assertTrue(V(1,1) in line)
        self.assertTrue(V(5,5) in line)
        self.assertTrue(V(3,3) in line)

    def test_statistics(self):
        self.vvcache = VVCache()
        self.vvcache.register_cache('len', VV.calc_distance)
        self.assertEqual(self.vvcache.read_cnt['len'], 0)
        self.assertEqual(self.vvcache.cache_hit_cnt['len'], 0)
        get_vv1 = self.vvcache.get_value('len', self.vv1)
        get_vv2 = self.vvcache.get_value('len', self.vv2)
        get_vv3 = self.vvcache.get_value('len', self.vv3)
        self.assertEqual(self.vvcache.read_cnt['len'], 3)
        self.assertEqual(self.vvcache.cache_hit_cnt['len'], 0)
        get_vv1_2nd = self.vvcache.get_value('len', self.vv1)
        get_vv2_2nd = self.vvcache.get_value('len', self.vv2)
        get_vv3_2nd = self.vvcache.get_value('len', self.vv3)
        self.assertEqual(self.vvcache.read_cnt['len'], 6)
        self.assertEqual(self.vvcache.cache_hit_cnt['len'], 3)
#        print(self.vvcache)

    def test_line_pixel_performance(self):
        self.lines_vmap = VMap()
        for i in range(10):
            x1 = random.randint(0,50)
            y1 = random.randint(0,50)
            x2 = random.randint(0,50)
            y2 = random.randint(0,50)
            v = random.randint(0,10) / 10.0
            self.lines_vmap.add_line(VV(V(x1,y1),V(x2,y2)),value=v)
        lines_vvs = self.lines_vmap.generate_all_pairs()

        ### 1. Cache disabled
        self.vvcache = VVCache(enable_cache=False)
        self.vvcache.register_cache('line', VV.calc_line_pixels)

        ns_0 = time.perf_counter_ns()
        sum_value = 0
        for vv in lines_vvs:
            trace_line_pixs = self.vvcache.get_value('line',vv)
            for pix in trace_line_pixs:
                if pix in self.lines_vmap:
                    sum_value += self.lines_vmap[pix]
        ns_1 = time.perf_counter_ns()
        ns_total = ns_1 - ns_0
        print("\nwithout cache")
        print("{} lines generated".format(len(lines_vvs)))
        print("{} ms to sum up.".format(ns_total/1000000.0))
        print(self.vvcache)
        self.vvcache.reset_counts()

        ### 2. Cache Enabled
        self.vvcache.enable_cache = True

        ns_0 = time.perf_counter_ns()
        sum_value = 0
        for vv in lines_vvs:
            trace_line_pixs = self.vvcache.get_value('line',vv)
            for pix in trace_line_pixs:
                if pix in self.lines_vmap:
                    sum_value += self.lines_vmap[pix]
        ns_1 = time.perf_counter_ns()
        ns_total = ns_1 - ns_0
        print("\ncache enabled")
        print("{} lines generated".format(len(lines_vvs)))
        print("{} ms to sum up.".format(ns_total/1000000.0))
        print(self.vvcache)
        self.vvcache.reset_counts()

        ### 3. Using the cache
        ns_0 = time.perf_counter_ns()
        sum_value = 0
        for vv in lines_vvs:
            trace_line_pixs = self.vvcache.get_value('line',vv)
            for pix in trace_line_pixs:
                if pix in self.lines_vmap:
                    sum_value += self.lines_vmap[pix]
        ns_1 = time.perf_counter_ns()
        ns_total = ns_1 - ns_0
        print("\nUsing the cache")
        print("{} lines generated".format(len(lines_vvs)))
        print("{} ms to sum up.".format(ns_total/1000000.0))
        print(self.vvcache)

if __name__ == '__main__':
    unittest.main()
