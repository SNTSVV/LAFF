from __future__ import division

import platform
import random
import os
import re

import numpy as np
import pandas as pd


class Utilities:

    T_ENV_NOTEBOOK = "notebook"
    T_ENV_RECIPE = "recipe"
    T_ENV_PYTHON = "python"

    T_DATA_NUM = "number"
    T_DATA_CATE = "category"
    T_DATA_TEXT = "text"
    T_DATA_EMPTY_C = 'unknown'
    T_DATA_EMPTY_N = -77777.77

    T_ORDER_SEQ = "seq"
    T_ORDER_RAND = "rand"

    T_TYPE_ALL = "all"
    T_TYPE_PART = "partial"
    T_TYPE_INC = "incremental"
    T_TYPE_SAMPLE = "sample"

    # RAN = 1
    T_ALG_MFM = "mfm"
    T_ALG_FLS = "fls"
    T_ALG_ARM = "arm"
    T_ALG_LAFF = "laff"
    T_ALG_NAIVE= "naive"

    @staticmethod
    def get_config_file_path():
        """
        get the absolute path of the file "config.json"
        """
        def get_base_directory(dir_name, nb_iterations):
            name = os.path.split(dir_name)[0]
            if nb_iterations == 0:
                return name
            else:
                return get_base_directory(name, nb_iterations - 1)

        def join_directories(base_path, directories):
            if not directories:
                return base_path
            else:
                base_path = os.path.join(base_path, directories[0])
                return join_directories(base_path, directories[1:])

        def get_separator():
            if 'Windows' in platform.system():
                separator = '\\'
            else:
                separator = '/'
            return separator

        def find_path(file_name):
            o_path = os.getcwd()
            separator = get_separator()
            str = o_path
            str = str.split(separator)
            while len(str) > 0:
                spath = separator.join(str) + separator + file_name
                leng = len(str)
                if os.path.exists(spath):
                    return spath
                str.remove(str[leng - 1])

        # cwd = os.getcwd()
        # config_file_path = ""
        # if env_type is Utilities.T_ENV_NOTEBOOK:
        #     # for Dataiku notebook
        #     dir_name = os.path.dirname(cwd)
        #     project_name = os.path.basename(dir_name)
        #     dss_home = get_base_directory(dir_name, 2)
        #     config_file_path = join_directories(dss_home, ["projects", project_name, "lib", "config.json"])
        #
        # elif env_type is Utilities.T_ENV_RECIPE:
        #     # for Dataiku recipe
        #     cwd = os.getcwd()
        #     project_home = get_base_directory(cwd, 3)
        #     dss_home = get_base_directory(cwd, 5)
        #     project_name = os.path.basename(project_home)
        #     config_file_path = join_directories(dss_home, ["projects", project_name, "lib", "config.json"])
        #
        # elif env_type is Utilities.T_ENV_PYTHON:
        # for a normal Python project, config.json is in the root folder of this project
        config_file_path = find_path("config.json")

        return config_file_path

    @staticmethod
    def is_number(num):
        pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
        result = pattern.match(num)
        if result:
            return True
        else:
            return False

    @staticmethod
    def coerce_to_unicode(x):
        """
        return the unicode of an input
        """
        import sys
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return np.unicode(x, 'utf-8')
            else:
                return np.unicode(x)
        else:
            return str(x)

    @staticmethod
    def map_col_types(columns, types):
        col_types = dict()
        for i in range(0, len(columns)):
            col_types[columns[i]] = types[i]
        return col_types

    @staticmethod
    def get_valid_col_types(ds, col_types):
        """
        get column type for each column in current dataset
        """
        cur_col_types = {}
        for col in ds.columns.values:
            cur_col_types[col] = col_types[col]
        return cur_col_types

    @staticmethod
    def get_cols_by_type(col_types, query_types):
        """
        get columns in certain types
        :param col_types:
        :param query_types: a list of certain types
        :return: a list of columns
        """
        columns = []
        for key, value in col_types.items():
            if value in query_types:
                columns.append(key)
        return columns

    @staticmethod
    def check_valid_time(values):
        """
        check the very large outliers
        """
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        new_values = []
        outlier_step = 10 * IQR
        for nu in values:
            if (nu < Q1 - outlier_step) | (nu > Q3 + outlier_step):
                pass
            else:
                new_values.append(nu)
        return new_values

    @staticmethod
    def add_status_to_frame(target, true_values, ranked_values, remained_flags, train_time, predict_time):
        result = pd.DataFrame(
            columns=['Target', 'Evidence', 'Truth', 'Ranked', 'Distribution', "Model", "Remained", "Train", "Predict"])
        result["Target"] = [target for _ in range(0, len(true_values))]
        result["Truth"] = true_values
        result["Ranked"] = ranked_values
        # result["Distribution"] = ranked_probability
        result["Remained"] = remained_flags
        result["Train"] = train_time
        result["Predict"] = predict_time
        return result

    @staticmethod
    def str_to_list(list_of_str):
        from ast import literal_eval
        list_of_list = []
        for str_vals in list_of_str:  # from string to list
            str_vals_in_list = literal_eval(str_vals)
            list_of_list.append(str_vals_in_list)
        return list_of_list

    @staticmethod
    def create_necessary_folder(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            

class MyLogger:
    def __init__(self, level):
        import datetime
        self.debug_level = level
        self.now = datetime.datetime.now()

    def format_input(self, elem):
        _temp = ""
        if isinstance(elem, list):
            _temp += "Printing list:\n"
            for e in elem:
                _temp += self.format_input(e)
                _temp += "\n"
            return _temp
        elif isinstance(elem, pd.Series):
            _temp += "Printing Series: \n"
            for e in elem.iteritems():
                _temp += str(e)
                _temp += "\n"
            return _temp
        elif isinstance(elem, dict):
            _temp += "Printing dictionnary: \n"
            for e1, e2 in elem.items():
                _temp += str(e1) + " : " + str(e2)
                _temp += "\n"
            return _temp
        else:
            _temp += str(elem) + " "
            return _temp

    def error(self, *msg):
        temp = ""
        for i in list(msg):
            temp += self.format_input(i)
        print(self.now.strftime("%Y-%m-%d %H:%M:%S") + "- ERROR - %s " % temp)

    def info(self, *msg):
        temp = ""
        for i in list(msg):
            temp += self.format_input(i)
        print(self.now.strftime("%Y-%m-%d %H:%M:%S") + "- INFO - %s " % temp)

    def debug(self, *msg):
        if self.debug_level == "debug":
            temp = ""
            for i in list(msg):
                temp += self.format_input(i)
            print(self.now.strftime("%Y-%m-%d %H:%M:%S") + "- BEBUG - %s " % temp)
