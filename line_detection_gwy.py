# coding=utf-8
"""
Line detection classes on Gwyddion console

2023 Ikuo Obataya, Quantum Desgin Japan
"""
from os import listdir,makedirs
import math, random, re
from os.path import isfile, join, exists,dirname,basename
import gwy, gtk
from vectors import *

class GwyApp:
    def __init__(self):
        self.con = self.get_con()
        if self.con == None:
            self.con = gwy.Container()
            gwy.gwy_app_data_browser_add(self.con)
        self.dialog_started = False
        self.dialog = None

    def flush(glob = ''):
        if glob == '':
            gwy.gwy_app_data_browser_shut_down()
        else:
            con = get_con()
            ids = gwy.gwy_app_data_browser_find_data_by_title(con, glob)
            if ids != -1 and len(ids) > 0:
                for id in ids:
                    data = gwy.gwy_app_data_browser_get(id)
                    con.remove(data)
                print('{} field(s) removed.'.format(len(ids)))

    def show_browser(self):
        gwy.gwy_app_data_browser_show()

    def get_con(self):
        return gwy.gwy_app_data_browser_get_current(gwy.APP_CONTAINER)

    def get_app_settings(self):
        """Get settings of application as Container
        """
        return gwy.gwy_app_settings_get()

    def get_field_id(objtype = gwy.APP_DATA_FIELD_ID):
        """Get current field. if type is not set, DataField is selected.
        Args:
            Object type, if none, Data Field is chosen.
            http://gwyddion.net/documentation/head/pygwy/gwy-module.html#gwy_app_data_browser_get_current

        Returns:
            Current data ID (int)
        """
        return gwy.gwy_app_data_browser_get_current(objtype)

    def get_field(self):
        """Returns current data field or specified data field (DataField)
        """
        return gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)

    def get_graph(self):
        """Returns current graph
        """
        return gwy.gwy_app_data_browser_get_current(gwy.APP_GRAPH)


    def add_field(self, dfield, showit = False, title = ''):
        """
        Adds a field to data_browser/container, then returns id.
        If con is None, field is not added to current container
        """
        if title == '':
            return gwy.gwy_app_data_browser_add_data_field(dfield, self.con, showit)
        else:
            id = gwy.gwy_app_data_browser_add_data_field(dfield, self.con, showit)
            self.set_title(id, title)
        return id

    def add_curve(self, gmodel, showit = False, title = ''):
        """
        Adds a 1D plots as gmodel
        """
        if title == '':
            return gwy.gwy_app_data_browser_add_graph_model(gmodel, self.con, showit)
        else:
            curve_id = gwy.gwy_app_data_browser_add_graph_model(gmodel, self.con, showit)
            self.set_title(curve_id, title)
        return curve_id

    def set_title(self, id, title):
        gwy.gwy_app_set_data_field_title(self.con, id, title)

    def show_log(self, field_id = -1, objtype = gwy.APP_DATA_FIELD):
        """Show log window GUI of current data
        """
        if field_id == -1:
            field_id = self.get_field_id()
        gwy.gwy_app_log_browser_for_channel(self.con, field_id)

    def start_dialog(self, title, starting_msg):
        self.dialog = gtk.Dialog(title, None, gtk.DIALOG_MODAL|gtk.DIALOG_DESTROY_WITH_PARENT,None)
        if self.dialog == None:
            return
        self.dialog_started = True
        gwy.gwy_app_wait_start(self.dialog, starting_msg)

    def finish_dialog(self):
        if self.dialog_started:
            self.dialog_started = False
            if self.dialog == None:
                return
            gwy.gwy_app_wait_finish()

    def set_msg_dialog(self, msg):
        if self.dialog_started:
            alive = gwy.gwy_app_wait_set_message(msg)
            if not alive:
                self.finish_dialog()
            return alive
        else:
            raise ValueError("Dialog hasn't started yet.")

    def set_progress_dialog(self, fraction):
        if self.dialog_started:
            alive = gwy.gwy_app_wait_set_fraction(fraction)
            if not alive:
                self.finish_dialog()
            return alive
        else:
            raise ValueError("Dialog hasn't started yet.")

class GwyContainer:
    def __init__(self, gwy_container):
        if not isinstance(gwy_container, gwy.Container):
            raise ValueError("gwy_container should be gwy.Container")
        self.con = gwy_container

    def show_field(self,id,show=True):
        """
        フィールドを表示する。
        Show/Hide the field by ID (int)
        """
        k = self.data_key(id,'visible')
        self.con.set_boolean_by_name(k,show)

    def get_field(self, field_id = None):
        """
        IDからDataFieldを取得する
        Data field in container by ID
        """
        if field_id is not None:
            return self.con[self.data_key(field_id)]

    def add_field(self, dfield, showit = False, title = ''):
        """
        Adds a field to the container
        """
        return GwyApp.add_field(dfield,self.con,showit,title)

    def find_data_by_title(self, titleglob):
        """
        合致する名前のDataField IDを取得する。ワイルドカード(*)が使用できる。
        Returns id list matching title of data (glob)
        """
        return gwy.gwy_app_data_browser_find_data_by_title(self.con, titleglob)

    def get_field_ids(self):
        if self.con is not None:
            return list(self.con.keys())
        else:
            return []

    def get_titles(self):
        """
        コンテナ内のタイトルを取得する。
        Returns list of titles of container. Returns None if none.
        """
        keys = self.con.keys()
        key_by_names = self.con.keys_by_name()
        titles = []
        for i in range(0,len(keys)):
            name = key_by_names[i]
            val = self.con.get_string(keys[i])
            if 'title' in name:
                titles.append(val)
            else:
                titles.append(None)
        return titles

    def show_field(self,id,show=True):
        """
        フィールドを表示する。
        Show/Hide the field by ID (int)
        """
        k = self.data_key(id,'visible')
        self.con.set_boolean_by_name(k,show)

    def get_meta(self,id,key=''):
        """
        id, keyからメタデータを取得する。
        keyが指定されなければmeta(Container)を取得する。
        Returns meta or data in meta from id and key.
        If key is none, returns meta container.
        If no key in meta, returns None
        """
        meta_of_field = "/{}/meta".format(id)
        if self.con.contains_by_name(meta_of_field)==False:
            return None
        meta = self.con[meta_of_field]
        if key == '':
            return meta
        if meta.contains_by_name(key):
            return meta.con[key]
        else:
            return None

    def is_visible(self,id):
        """
        表示しているかどうか
        Get visibility of field by ID (int)
        """
        key = self.data_key(id,'visible')
        if self.con.contains_by_name(key):
            return self.con[self.data_key(id,'visible')]
        else:
            return False
    def remove_other_channels(self, ids_to_keep):
        """
        Remove channels except for ids_to_keep (array(int))
        """
        all_ids = gwy.gwy_app_data_browser_get_data_ids(self.con)
        for id in all_ids:
            if (id in ids_to_keep)==False:
                self.con.remove_by_prefix("/"+str(id))

    def set_palette_and_color(self, field_id, palette, range_type = 0, min = 0, max = 0):
        """色パレットと色範囲を指定する。range_typeは4種類ある。
        Set palette and color range
        range-type (int)
            0 - from MIN to MAX   - gwy.LAYER_BASIC_RANGE_FULL
            1 - fixed min and max - gwy.LAYER_BASIC_RANGE_FIXED
            2 - Auto cut          - gwy.LAYER_BASIC_RANGE_AUTO
            3 - Auto adapt        - gwy.LAYER_BASIC_RANGE_ADAPT
        """
        self.con.set_string_by_name(self.base_key(field_id,'palette'),palette)
        self.con.set_int32_by_name(self.base_key(field_id,'range-type'),range_type)
        if range_type == 1:
            self.con.set_double_by_name(self.base_key(field_id,'min'),min)
            self.con.set_double_by_name(self.base_key(field_id,'max'),max)

    def extract_channels(self, channels, procs = None):
        """
        指定したチャンネルだけを抽出する。
        すべてのチャンネルにprocsに適用することもできる。
        Extract specified channels (str[]).
        Processes can be run (str[]), parameters for them must be set in advance.
        """
        # 抽出対象をさがすループ
        # loop for searching extracting channels
        extract_chs = []
        for ch in channels:
            ids = gwy.gwy_app_data_browser_find_data_by_title(self.con, ch)
            if ids==None or len(ids)==0:
                continue
            _id = ids[0]
            extract_chs.append(_id)
            # Data processing
            gwy.gwy_app_data_browser_select_data_field(self.con, _id)
            if procs==None or len(procs)==0:
                continue
            for p in procs:
                gwy.gwy_app_run_process_func_in_mode(p,gwy.RUN_IMMEDIATE)

        # 探したチャンネル以外を削除する。
        # remove channels except for CHANNELS
        all_ids = gwy.gwy_app_data_browser_get_data_ids(self.con)
        for id in all_ids:
            if (id in extract_chs)==False:
                self.remove_by_prefix("/"+str(id))

    def what_changed(self, curr,prev,only_changed=True):
        """
        Container内で異なるキーを探す。
        Probe changed data in Container
        """
        LIST_FRMT = '{0:<40}{1}\t{2}\t{3}\t{4}\t{5}'
        curKeys = curr.keys_by_name()
        prevKeys = prev.keys_by_name()

        all_keys = []
        for key in curKeys:
            if not key in all_keys:
                all_keys.append(key)

        for key in prevKeys:
            if not key in all_keys:
                all_keys.append(key)
        all_keys.sort()
        print(LIST_FRMT.format('key','added','del','mod','curr','prev'))
        for key in all_keys:
            prevHas = (key in prevKeys)
            currHas = (key in curKeys)
            added = currHas and (not prevHas)
            deleted = (not currHas) and prevHas
            modified = False
            if prevHas and currHas:
                modified = (curr.get_value_by_name(key) != prev.get_value_by_name(key))
            if currHas:
                v = str(curr.get_value_by_name(key))[0:20]
            else:
                v = '(del)'
            if prevHas:
                pv = str(prev.get_value_by_name(key))[0:20]
            else:
                pv = '(add)'
            if not only_changed:
                print(LIST_FRMT.format(key,added,deleted,modified,v,pv))
            else:
                if added or deleted or modified:
                    print(LIST_FRMT.format(key,added,deleted,modified,v,pv))

    def get_file_paths(self, folder_path,ext="gwy",max=-1,skip=-1):
        """
        Get all files in folder_path with extension of ext.
        max and skip can be set. They are ignored if negative.
        """
        files = [f for f in listdir(folder_path) if (isfile(join(folder_path, f)) and f[-4:]=='.'+ext)]
        if len(files)==0:
            return paths
        # sort by last index number in string
        sort = re.compile(r'\d+')
        if sort.findall(files[0]) > 0:
            files = sorted(files, key=lambda s:int(sort.findall(s)[-1]))
        # skip and max
        if skip>0:
            files = files[skip:]
        if max>0:
            files = files[:max]
        # generate full path of files
        paths = []
        for f in files:
            path = join(folder_path,f)
            paths.append(path)
        return paths

    def load_files(self, folder_path,ext="gwy",gui=True,max=-1,skip=-1):
        """
        Load files to application.
        max and skip can be set.
        """
        paths = self.get_file_paths(folder_path,ext=ext,max=max,skip=skip)
        con = None
        for p in paths:
            con = self.load_file(p,gui)
        return con

    def load_file(filename,gui = True):
        """
        Load a file to application.
        """
        if gui:
            return gwy.gwy_app_file_load(filename)
        else:
            return gwy.gwy_file_load(filename, gwy.RUN_NONINTERACTIVE)

    #region static methods
    @staticmethod
    def base_key(id,key=''):
        """
        baseのキー文字列(eg./0/base/key)をid,keyで取得する。
        Returns key string from id and key (eg./0/base/key)
        """
        if key == '':
            return "/{}/base".format(id)
        else:
            return "/{}/base/{}".format(id,key)

    @staticmethod
    def data_key(id,key=''):
        """
        dataのキー文字列(eg. /0/data/key)
        Returns key string for data from id and key (eg. /0/data/key)
        """
        if key == '':
            return "/{}/data".format(id)
        else:
            return "/{}/data/{}".format(id,key)
    #endregion

class GwyField:
    def __init__(self, src_data, pix_sz_V = V(1,1), threshold=0.2):
        if isinstance(src_data,gwy.DataField):
            # DataField is given
            self.field = src_data
            offset_x = self.field.get_xoffset()
            offset_y = self.field.get_yoffset()
            self.offset = V(offset_x, offset_y)
            self.cols = self.field.get_xres()
            self.rows = self.field.get_yres()
            # preventing from cached quantity
            (self.min_value,self.max_value) = self.field.area_get_min_max_mask(
                                                gwy.DataField(0,0,0,0),gwy.MASK_IGNORE,0,0,self.cols,self.rows)
            self.range_value = self.max_value - self.min_value
            self.vmap = self.generate_map_from_field(threshold=threshold)
        elif isinstance(src_data, VMap):
            # VectorMap is given, vector keys are pixelated.
            app = GwyApp()
            (vmin, vmax) = (src_data.min_vec,src_data.max_vec)
            offset_vec = vmin
            diagonal_vec = vmax - vmin
            (offset_x, offset_y) = (offset_vec.x_int(), offset_vec.y_int())
            (self.cols, self.rows) = (diagonal_vec.x_int() + 1, diagonal_vec.y_int() + 1)
            (px, py) = (pix_sz_V.x(), pix_sz_V.y())
            self.field = gwy.DataField(self.cols, self.rows, self.cols * px, self.rows * py)
            if offset_x != 0 or offset_y != 0:
                self.vmap = src_data.get_translated(V(-offset_x, -offset_y), round_vec = True)
            else:
                self.vmap = src_data.round()
            vecs = self.vmap.vec_list()

            # this takes a long time. TODO: to show progress bar
            vecs_cnt = len(vecs)
            for i in range(0,vecs_cnt):
                vec = vecs[i]
                (px, py) = (vec.x_int(), vec.y_int())
                if px < 0 or py <0 or px >= self.cols or py >= self.rows:
                    raise ValueError("({},{}) out of range for the DataField(cols:{}, rows:{})".format(px,py,self.cols,self.rows))
                val = self.vmap[vec]
                self.field.area_fill(px, py, 1, 1, val)

            self.offset = offset_vec
            (min_vval, max_vval) = (self.vmap.min_vecvalue,self.vmap.max_vecvalue)
            self.min_value = self.vmap[min_vval.vector]
            self.max_value = self.vmap[max_vval.vector]
            self.range_value = self.max_value - self.min_value
            self.field.set_xoffset(self.offset.x())
            self.field.set_yoffset(self.offset.y())
        else:
            print("src_data error.")
            raise ValueError("arguments should be gwy.DataField or VMap.")

    def __str__(self):
        (w, h) = (self.field.get_xres(), self.field.get_yres())
        (_min,_max) = (self.min_value, self.max_value)
        return "GwyField {} x {}, value range:{} - {}".format(w,h,_min,_max)
    """
    Generates dictionary where key and value are (x,y) tuple and value, respectively.
    """
    def generate_map_from_field(self, threshold=0.2):
        threshold_signal = self.min_value + threshold * self.range_value
        vmap = VMap()
        (cols,rows) = (self.field.get_xres(),self.field.get_yres())
        for col in range(0,cols):
            for row in range(0,rows):
                signal = self.field.get_val(col,row)
                if signal>=threshold_signal:
                    v = V(col,row)
                    vmap[v] = signal
        return vmap

    def get_real_xy(self):
        return V(self.field.get_xreal(),self.field.get_yreal())

    def get_res_xy_px(self):
        return V(self.field.get_xres(),self.field.get_yres())

    def get_pixel_size(self):
        pix_x = self.field.get_xreal()/self.field.get_xres()
        pix_y = self.field.get_yreal()/self.field.get_yres()
        return V(pix_x,pix_y)

    def refresh(self):
        self.field.data_changed()

    def get_gaussian_filtered(self, g_pix = 1.0):
        """Gaussian filtered
        Args:
            Pixels for gaussian filter
        Returns:
            GwyField: gaussian filtered data
        """
        dfield_g = self.field.duplicate()
        dfield_g.filter_gaussian(g_pix)
        return GwyField(dfield_g)

    @staticmethod
    def add_vmap(source, idx=-1, showit=True):
        if isinstance(source, (VMapList,list)):
            if idx >= 0:
                vmap_list = VMapList(source[idx])
            else:
                vmap_list = VMapList(source)
        else:
            vmap_list = VMapList(source)
        app = GwyApp()
        all_vmap = VMap()
        count = len(vmap_list)
        for i in range(0,count):
            vmap = vmap_list[i]
            all_vmap.add_vmap(vmap)
        new_field = GwyField(all_vmap)
        id = app.add_field(new_field.field,showit=showit,title='Created')
        app.show_browser()
        return (new_field, id)


class GwyVMapListIO():
    def __init__(self):
        self.io = VMapIO()

    def save(self, filepath, source):
        self.source = source
        self.io.open(filepath,'w',self.source)
        self.progress = Progress()
        app = GwyApp()
        gui_cancelled = False
        app.start_dialog("Save", "Saving VMap list...")
        while(self.io.write_vmap()):
            progress = self.io.progress()
            gui_alive = app.set_progress_dialog(progress)
            if not gui_cancelled and not gui_alive:
                gui_cancelled = True
                break
        self.io.close()
        app.finish_dialog()

    def load(self,filepath):
        self.io.open(filepath,'r', None)
        app = GwyApp()
        gui_cancelled = False
        app.start_dialog("Load", "Loading VMap list...")
        while(self.io.read_vmap()):
            progress = self.io.progress()
            gui_alive = app.set_progress_dialog(progress)
            if not gui_cancelled and not gui_alive:
                gui_cancelled = True
                break
        self.io.close()
        app.finish_dialog()
        return self.io.vmap_list

class GwyCurve():
    """ Class for 1D plot on Gwyddion

    """
    def __init__(self):
        self.gmodel = gwy.GraphModel()
        self.current_col_idx = 0

    def add_curve(self, y_list, x_real = 1, x_offset = 0, color_idx = -1, desc = 'curve'):
        data_line = gwy.DataLine(len(y_list),1,)
        data_line.set_real(x_real)
        data_line.set_offset(x_offset)
        i = 0
        for y in y_list:
            data_line.set_val(i,y)
            i+=1
        curve = gwy.GraphCurveModel()
        curve.props.description = desc
        if color_idx >=0:
            curve.props.color = gwy.gwy_graph_get_preset_color(color_idx)
        else:
            curve.props.color = gwy.gwy_graph_get_preset_color(self.current_col_idx)
            self.current_col_idx += 1
        curve.set_data_from_dataline(data_line,0,len(y_list))
        curve.set_property('mode', gwy.GRAPH_CURVE_LINE)
        self.gmodel.add_curve(curve)
        self.gmodel.set_units_from_data_line(data_line)



class GwyNanosurf:
    def get_CH_names_NSF(channels):
        """
        NanosurfのFWDシグナル名セットを配列で返す。wave, phase, NID(旧ソフト)
        """
        Common_FWD=["Forward - Topography","Forward - Position Z","Forward - Z-Controller In"]
        Wave_FWD  =["Forward - Deflection"]
        Phase_FWD = ["Forward - Analyzer 1 Amplitude","Forward - Phase"]
        NID_FWD =["Z-Axis forward","Deflection forward","Amplitude forward","Phase forward","Z-Axis Sensor forward"]
        ret_array = Common_FWD
        if "wave" in channels:
            ret_array = ret_array + Wave_FWD
        if "phase" in channels:
            ret_array = ret_array + Phase_FWD
        if "NID" in channels:
            ret_array = ret_array + NID_FWD
        return ret_array


def find_chains(app, min_pix,max_pix,gui_cancelled=False):
    app.start_dialog("Calculation","Getting neighboring...")
    dfield = app.get_field()
    gf = GwyField(dfield)
    vmap = gf.vmap.copy()
    initial_count = len(vmap)
    current_count = initial_count
    neighboring = VMapList()
    while(current_count > 0):
        found_vmap = vmap._get_neighboring_from(vmap.first_vec())
        if len(found_vmap) >= min_pix:
            neighboring.append(found_vmap)
        current_count = len(vmap)
        if len(neighboring) >= max_pix:
            break
        progress = initial_count - current_count
        gui_alive = app.set_progress_dialog((progress + 1.0)/initial_count)
        if not gui_cancelled and not gui_alive:
            gui_cancelled = True
            break
    app.finish_dialog()
    return neighboring

def get_vvcache():
    return VVCache(enable_cache=True)


def plot_at_angles(app,dict,step_angle,title='data'):
    x_real = 180
    y_values = []
    target_y_dict = dict
    total_value = 0
    for i in range(0,180,step_angle):
        angle = float(i)
        if angle in target_y_dict:
            y_values.append(target_y_dict[angle])
            total_value += target_y_dict[angle]
        else:
            y_values.append(0)

    if total_value == 0:
        return
#        raise ValueError("No distinguished linear region in this chain.")

    gwycurve = GwyCurve()
    gwycurve.add_curve(y_values, x_real=x_real, color_idx=0, desc=title)
    app.add_curve(gwycurve.gmodel, showit = True)

def draw_lines_at_angles(app, line_results,source_vmap,step_angle,title='Detected lines'):
    emphasized = source_vmap.copy()
    for i in range(0, 180, step_angle):
        angle = float(i)
        vv_dict = line_results.vv_sum_max_at_angles
        if angle in vv_dict:
            val = line_results.sum_at_angles[angle]
            if val < line_results.avr_sum_z:
                continue
            emphasized.add_line(vv_dict[angle], value = val)
    emphasized_gf = GwyField(emphasized)
    app.add_field(emphasized_gf.field, showit=True, title=title)
