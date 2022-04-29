# coding: utf-8
from __future__ import division

import copy
import shutil

from algorithms.preprocessing import Preprocessing as proc, Preprocessing
from tools.utils import *


class DataGenerator:

    @staticmethod
    def get_data_with_default_order(df, field_order):
        """
        reorganize the dataframe based on the default field order
        Parameters:
            df: the entire dataset
            field_order: the order of field in the configuration file
        Return:
            df_ordered: ordered data
        """
        df_ordered = pd.DataFrame()
        for col in field_order:
            df_ordered[col] = df[col]
        return df_ordered

    @staticmethod
    def ratio_sampling(df, tvt_ratio, rd_flag=False, train_sample_ratio=1.0):
        """
        sampling according to the train_valid_test_ratio. it keeps original index of df
        Parameters:
            df: the entire dataset
            tvt_ratio: ratio of train set, validation set, and test set
            rd_flag: randomly sample or not
        Return:
            train set
            validation set
            test set (they will keep the index of the original dataset)
        """
        total = tvt_ratio[0] + tvt_ratio[1] + tvt_ratio[2]
        train_rate = 1.0 * tvt_ratio[0] / total
        validation_rate = 1.0 * tvt_ratio[1] / total

        if rd_flag:
            df = df.sample(frac=1.0)
            df = df.reset_index()
        print(df.shape)
        cut_index1 = int(df.shape[0] * train_rate)
        cut_index2 = int(df.shape[0] * (train_rate + validation_rate))
        train = df.iloc[0:cut_index1]
        validation = df.iloc[cut_index1:cut_index2]
        test = df.iloc[cut_index2:]
        print("cut point of the data: ", cut_index1, cut_index2)
        if rd_flag:
            train = train.drop("index", axis=1)
            validation = validation.drop("index", axis=1)
            test = test.drop("index", axis=1)
        if 1.0 > train_sample_ratio > 0.0:
            train = train.sample(frac=train_sample_ratio, random_state=35)
        train = train.copy(deep=True)
        validation = validation.copy(deep=True)
        test = test.copy(deep=True)
        return train, validation, test

    @staticmethod
    def num_sampling(df, tvt_cut, rd_flag=False):
        """
       sampling according to the train_valid_test_number. it keeps original index of df
       Parameters:
            df: the entire dataset
            tvt_cut: number of train set, validation set, and test set
            rd_flag: randomly sample or not
       Return:
            train set
            validation set
            test set (they will keep the index of the original dataset)
       """
        total = df.shape[0]
        train_rate = 1.0 * tvt_cut[0] / total
        validation_rate = 1.0 * (tvt_cut[1] - tvt_cut[0]) / total
        test_rate = 1 - train_rate - validation_rate
        return DataGenerator.ratio_sampling(df, [train_rate, validation_rate, test_rate], rd_flag)

    @staticmethod
    def read_train_test(train_path, test_path, col_id_path, ds_splitter):
        """
        read the preprocessed data from a parth
        Parameters:
            train_path: the path of the training set
            test_path: the path of the testing set
            col_id_path: the path of the col_val_id file
            ds_splitter: splitter in the file
        Return:
            train set, test set, and col_val_id file
        """
        train_set = pd.read_csv(train_path, ds_splitter)
        test_set = pd.read_csv(test_path, ds_splitter)
        col_val_id = pd.read_csv(col_id_path, ds_splitter)
        return train_set, test_set, col_val_id

    @staticmethod
    def calc_num_of_candidates(raw_ds_path, ds_splitter):
        """
        calculate the number of candidates for each field
        Parameters:
            raw_ds_path: the path of the raw data file
            ds_splitter: splitter in the file
        Return:
            num_of_candidates dict (field and number)
        """
        df = pd.read_csv(raw_ds_path, ds_splitter)
        num_of_candidates = {}
        for col in df.columns:
            num_of_unique = df[col].nunique()
            num_of_candidates[col] = num_of_unique
        return num_of_candidates

    @staticmethod
    def get_test_set_per_target(test_set_all):
        """
        get the test set for each target
        Parameters:
            test_set_all: the test set
        Return:
            test_per_target: the input instances on each target field
        """
        test_per_target = {}
        for target, df_region in test_set_all.groupby('target'):
            test_per_target[target] = df_region

        return test_per_target

    @staticmethod
    def get_raw_df(ds_name, config):
        """
        read the raw data file
        Parameters:
            ds_name: the name of the dataset
            config: configuration file
        Return:
            raw_df: a dataframe for the raw data
        """
        raw_folder = config['dataset']['root_path'] + config['dataset']['raw_folder']
        raw_config = config['dataset'][ds_name]  # the name of the original dataframe
        ds_splitter = raw_config['splitter']  # splitter of the original dataframe
        raw_df = pd.read_csv(raw_folder + ds_name + ".csv", ds_splitter)
        return raw_df

    @staticmethod
    def raw_preprocessing(config, raw_df, col_types, ds_field_order):
        """
        generate the (preprocessed) training set and the testing set
        from the original data in dataset/raw under a given configuration
        Parameters:
            config: configuration file
            raw_df: the raw dataset
            col_types: types of columns in raw_df
            ds_field_order: default filling order of fields (columns) in a form
        Return:
            the training set
            the validation set (empty in the experiments)
            the testing set
            the mapping between the original values and the id in the dataset (json)
        """
        train_ratio = config['dataset']['train_ratio']
        test_ratio = config['dataset']['test_ratio']
        empty_ratio = config['dataset']['empty_ratio']
        unique_ratio = config['dataset']['unique_ratio']
        train_sample_ratio = config['dataset']['train_sample']

        print(col_types)
        raw_df = proc.coerce_feature_types(raw_df, col_types)
        raw_df = raw_df.replace("nan", np.nan)  # coerce_feature_types will convert np.NaN to a string "nan"
        raw_df = DataGenerator.get_data_with_default_order(raw_df, ds_field_order)

        # remove columns that do not provide information for form filling
        raw_df = proc.rm_empty_col(raw_df, empty_ratio, col_types)
        raw_df = proc.rm_unique_col(raw_df, unique_ratio, col_types)
        raw_df = proc.rm_same_col(raw_df)
        raw_df = proc.rm_mixed_col(raw_df, col_types)
        # remove empty rows or rows with only one value
        raw_df.dropna(axis=0, how='all', inplace=True)
        raw_df.dropna(axis=0, thresh=2, inplace=True)
        raw_df.info()

        # df = df[['sex', 'tissue', 'cell_line', 'cell_type', 'disease',
        #          'ethnicity']]  # to be deleted (just to check the results)
        # cols = []
        # for col in raw_df.columns:
        #     if col in ['sex', 'tissue', 'cell_line', 'cell_type', 'disease', 'ethnicity']:
        #         cols.append(col)
        # raw_df = raw_df[cols]
        # # generating training, validation, testing sets
        print("Generating training, validation, testing sets")
        tvt_ratio = [train_ratio, 0, test_ratio]  # train_validation_test
        train_set, validate_set, test_set = DataGenerator.ratio_sampling(raw_df, tvt_ratio, False, train_sample_ratio)

        print("shape of training, validation, testing sets: ", train_set.shape, validate_set.shape, test_set.shape)

        # fill empty rows with default values
        print("Fill empty rows with default values")
        col_types = Utilities.get_valid_col_types(train_set, col_types)
        columns = Utilities.get_cols_by_type(col_types, [Utilities.T_DATA_NUM])
        train_set = proc.impute_with_mean(train_set, columns)
        columns = Utilities.get_cols_by_type(col_types, [Utilities.T_DATA_CATE, Utilities.T_DATA_TEXT])
        train_set = proc.impute_with_label(train_set, columns, Utilities.T_DATA_EMPTY_C)

        # rescale numerical columns
        print("Rescale and discretize numerical columns")
        columns = Utilities.get_cols_by_type(col_types, [Utilities.T_DATA_NUM])
        train_set, validate_set, test_set = proc.rescale_num_minmax(train_set, validate_set, test_set, columns)
        train_set, validate_set, test_set = proc.numerical_discretize(train_set, validate_set, test_set, columns)

        train_set, validate_set, test_set, col_val_id = proc.map_col_with_id(train_set, validate_set, test_set, col_types)
        print(col_val_id)
        col_val_id_df = pd.DataFrame()
        for key, value in col_val_id.items():
            col_val_id_field = pd.DataFrame(list(value.items()), columns=['val', 'id'])
            col_val_id_field["field"] = key
            frames = [col_val_id_df, col_val_id_field]
            col_val_id_df = pd.concat(frames, ignore_index=True)  # testing instances for all targets

        return train_set, validate_set, test_set, col_val_id_df

    @staticmethod
    def check_train_test_exist(ds_path, ds_name, fill_type, fill_order):
        """
        check the existance of the train set and test set
        Parameters:
            ds_path: the path of the train set and test set
            ds_name: the name of dataset
            fill_type: fill type (all, partial, incremental)
            fill_order: fill order (seq, rand)
        Return:
            true/false: whether the files exist
        """
        check_train = check_test = False
        if fill_type == Utilities.T_TYPE_ALL:
            check_train = os.path.exists(ds_path + ds_name + "_train.csv")
            check_test = os.path.exists(ds_path + ds_name +"_"+ fill_order +"_test.csv")
        elif fill_type == Utilities.T_TYPE_PART:
            check_train = os.path.exists(ds_path + ds_name + "_train.csv")
            check_test = os.path.exists(ds_path+ fill_type + "/" + ds_name + "_" + fill_type + "_test_1.csv")
        elif fill_type == Utilities.T_TYPE_INC or fill_type == Utilities.T_TYPE_SAMPLE:
            check_train = os.path.exists(ds_path + fill_type+"/" + ds_name + fill_type+"_1_train.csv")
            check_test = os.path.exists(ds_path + fill_type+"/" + ds_name + fill_type+"_1_"+fill_order+"_test.csv")

        if check_train and check_test:  # the training and testing files exist
            return True
        else:
            return False

    @staticmethod
    def save_files(ds, ds_splitter, file_path, file_name):
        """
        save a file (create directories if necessary)
        Parameters:
            ds: the dataframe to save
            ds_splitter: splitter in the file
            file_path: the path to save the file
            file_name: the name of the file
        """
        if os.path.exists(file_path + file_name):
            print('dataset ' + file_name + ' exists')
        else:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            ds.to_csv(file_path + file_name, sep=ds_splitter, index=False)

    @staticmethod
    def matched_files_in_folder(folder, match_str):
        """
        match the files containing certain string
        Parameters:
            folder: the folder containing files
            match_str: the string to match
        return
            files: names of the matched files
        """
        files = os.listdir(folder)
        files = [file for file in files if match_str in file]
        return files

    @staticmethod
    def gen_ds_seq(config):
        """
        generate datasets for sequential filling; save the dataset to the folder
        Parameters:
            config: the configuration
        """
        train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']
        raw_folder = config['dataset']['root_path'] + config['dataset']['raw_folder']
        fill_type = config["predict"]["fill_type"]
        fill_order = config["predict"]["fill_order"]
        ds_names = config['dataset']['names']

        for ds_name in ds_names:
            if DataGenerator.check_train_test_exist(train_test_folder, ds_name, fill_type, fill_order):
                print(ds_name+" for sequential filling exists")
                continue
            print("generating sequential filling dataset for "+ds_name)
            ds_config = config['dataset'][ds_name]
            col_types = ds_config['field_types']
            field_order = list(col_types.keys())
            ds_splitter = ds_config['splitter']  # splitter of the original dataframe
            raw_df = pd.read_csv(raw_folder + ds_name + ".csv", ds_splitter)
            train_set, validate_set, test_set, col_val_id_df \
                = DataGenerator.raw_preprocessing(config, raw_df, col_types, field_order)
            unknown_id = Preprocessing.get_unknown_id(col_val_id_df)

            field_order = np.array(test_set.columns.tolist())
            first_field = field_order[0]
            test_instances = pd.DataFrame()

            for col in field_order:
                if col_types[col] != Utilities.T_DATA_CATE:  # the target should be categorical fields
                    continue
                target = col
                if target == first_field:  # don't predict the first field
                    continue

                test_set_cp = test_set.copy(deep=True)  # copy of the test data set
                # remove fields that the target is empty: cannot be evaluated
                # empty_id = unknown_id[col][Utilities.T_DATA_EMPTY_C]
                empty_id = unknown_id[col]
                test_set_cp = test_set_cp[test_set_cp[target] != empty_id]
                target_index = int(np.argwhere(field_order == target))  # get the index of the target field
                filled_fields = field_order[:target_index]  # fields before target_index are filled fields
                empty_fields = field_order[target_index+1:]  # fields after target_index are unfilled fields
                # set empty field with the id of "unknown"
                for item in empty_fields:
                    # empty_id = col_val_id[item][Utilities.T_DATA_EMPTY_C]
                    empty_id = unknown_id[item]
                    test_set_cp[item] = empty_id
                filled_has_val = []
                # remove rows if the value of filled fields are all empty
                for i, row in test_set_cp.iterrows():
                    count = 0
                    for field in filled_fields:
                        # if test_set_cp.at[i, field] == col_val_id[field][Utilities.T_DATA_EMPTY_C]:
                        if test_set_cp.at[i, field] == unknown_id[field]:
                            count = count + 1
                    if not count == len(filled_fields):
                        filled_has_val.append(i)
                test_set_cp = test_set_cp.loc[filled_has_val]
                test_set_cp["target"] = target  # testing instances for a target
                frames = [test_instances, test_set_cp]
                test_instances = pd.concat(frames, ignore_index=True)  # testing instances for all targets

            # save the dataset
            DataGenerator.save_files(train_set, ds_splitter, train_test_folder, ds_name+"_train.csv")
            DataGenerator.save_files(test_instances, ds_splitter, train_test_folder, ds_name+"_"+fill_order+"_test.csv")
            DataGenerator.save_files(col_val_id_df, ds_splitter, train_test_folder, ds_name+"_val_id.csv")

    @staticmethod
    def gen_ds_rand(config):
        """
        generate datasets for random filling; save the dataset to the folder
        Parameters:
            config: the configuration
        """
        train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']
        raw_folder = config['dataset']['root_path'] + config['dataset']['raw_folder']
        fill_order = config["predict"]["fill_order"]
        fill_type = config["predict"]["fill_type"]
        ds_names = config['dataset']['names']

        for ds_name in ds_names:
            if DataGenerator.check_train_test_exist(train_test_folder, ds_name, fill_type, fill_order):
                print(ds_name+" for random filling exists")
                continue
            print("generating random filling dataset for " + ds_name)
            ds_config = config['dataset'][ds_name]
            col_types = ds_config['field_types']
            field_order = list(col_types.keys())
            ds_splitter = ds_config['splitter']  # splitter of the original dataframe
            raw_df = pd.read_csv(raw_folder + ds_name + ".csv", ds_splitter)
            train_set, validate_set, test_set, col_val_id_df \
                = DataGenerator.raw_preprocessing(config, raw_df, col_types, field_order)

            field_order = np.array(test_set.columns.tolist())  # the default field order
            test_set_cp = test_set.copy(deep=True)  # copy of the test data set
            field_order_index = np.arange(len(field_order))
            unknown_id = Preprocessing.get_unknown_id(col_val_id_df)
            field_order_nan_id = [unknown_id[col] for col in field_order]  # id of 'unknown'

            test_instances = []
            for i, row in test_set_cp.iterrows():
                random.shuffle(field_order_index)  # randomly generate a field input order for each test instance
                filled_count = 0
                # iterate each field in this random order: simulate filling under this order
                for j in range(0, len(field_order_index)):
                    field_index = field_order_index[j]
                    field = field_order[field_index]
                    val = test_set_cp.at[i, field]
                    empty_id = field_order_nan_id[field_index]
                    if val != empty_id:
                        filled_count = filled_count + 1
                    # if there is at least one filled field, the current field is categorical,
                    # and the true value of the current field is not empty, we can take it as a target for prediction
                    if filled_count > 1 and col_types[field] == Utilities.T_DATA_CATE and val != empty_id:
                        target = field
                        test_instance = list(row)
                        # under this random order (field_order_index),
                        # set the fields after the current field to be empty
                        for k in range(j + 1, len(field_order_index)):
                            empty_index = field_order_index[k]
                            test_instance[empty_index] = field_order_nan_id[empty_index]
                        test_instance.append(target)  # we have a testing instance
                        test_instances.append(test_instance)  # it records all testing instances for all fields
            test_instances_fields = field_order.tolist() + ["target"]
            test_instances = pd.DataFrame(test_instances, columns=test_instances_fields)
            test_instances.sort_values("target", inplace=True)  # sort all testing instances by the target field

            # save the dataset
            DataGenerator.save_files(train_set, ds_splitter, train_test_folder, ds_name+"_train.csv")
            DataGenerator.save_files(test_instances, ds_splitter, train_test_folder, ds_name+"_"+fill_order+"_test.csv")
            DataGenerator.save_files(col_val_id_df, ds_splitter, train_test_folder, ds_name+"_val_id.csv")

    @staticmethod
    def gen_ds_partial_k_filled(test_set, col_val_id_df, col_types, k):
        """
        generate test set with k filled fields
        Parameters:
            test_set: testing set for generating sequential filling
            col_val_id_df: mapping between values in test set and id
            col_types: the type of each colum
            k: number of filled fields
        Return:
            testing instances with k filled fields
        """
        field_order = np.array(test_set.columns.tolist())  # the default field order
        test_set_cp = test_set.copy(deep=True)  # copy of the test data set
        field_order_index = np.arange(len(field_order))
        unknown_id = Preprocessing.get_unknown_id(col_val_id_df)
        field_order_nan_id = [unknown_id[col] for col in field_order]  # id of 'unknown'

        test_instances = []
        for target_index in range(len(field_order_index)):

            for i, row in test_set_cp.iterrows():
                # count the number of fields in a row which have values
                filled_count = 0
                for j in range(0, len(field_order_index)):
                    field_index = field_order_index[j]  # field index
                    field = field_order[field_index]  # field name
                    val = test_set_cp.at[i, field]  # copy the value of the cell i of the field
                    empty_id = field_order_nan_id[field_index]  # get the nan value of the field
                    if val != empty_id:
                        filled_count = filled_count + 1  # compute number of filled fields

                field = field_order[target_index]
                field_index = field_order_index[target_index]
                val = test_set_cp.at[i, field]  # copy the value of the cell i of the field
                empty_id = field_order_nan_id[field_index]  # get the nan value of the field
                # if the number of fields having values is larger than the number of filled fields required,
                # the current field is categorical,
                # and the true value of the current field is not empty, we can take it as a target for prediction
                if filled_count > k and col_types[field] == Utilities.T_DATA_CATE and val != empty_id:
                    # print(filled_count)
                    target = field
                    test_instance = list(row)

                    # list of fields (features) that can be set empty
                    features = []
                    for x in range(len(field_order_index)):
                        if x == field_index:
                            continue
                        else:
                            features.append(x)

                    # already empty fields in the test instance
                    emp = []
                    for ind in range(len(features)):
                        v = features[ind]
                        if test_instance[v] == field_order_nan_id[v]:
                            emp.append(v)

                    # fields that will be set to empty
                    poss_empt_col = set(features) - set(emp)
                    if filled_count == k:
                        not_empty = set(features) - set(emp)
                    else:
                        not_empty = random.sample(list(poss_empt_col), k)
                    empty_col = set(features) - set(not_empty)

                    # set the fields in empty_col to empty
                    for a in range(len(field_order_index)):
                        if a in empty_col:
                            empty_index = field_order_index[a]
                            test_instance[empty_index] = field_order_nan_id[empty_index]

                    test_instance.append(target)  # we have a testing instance
                    test_instances.append(test_instance)  # it records all testing instances for all fields

        test_instances_fields = field_order.tolist() + ["target"]
        test_instances = pd.DataFrame(test_instances, columns=test_instances_fields)
        test_instances.sort_values("target", inplace=True)  # sort all testing instances by the target field
        return test_instances

    @staticmethod
    def not_enough_test(num_of_fields, pre_test_num, cur_test_num, th):
        if num_of_fields > 1:
            return 1.0*cur_test_num/pre_test_num < th
        else:
            return False

    @staticmethod
    def gen_ds_partial(config):
        """
        generate datasets which a certain number of fileds are filled; save the dataset to the folder
        Parameters:
            config: the configuration
        """
        train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']
        raw_folder = config['dataset']['root_path'] + config['dataset']['raw_folder']
        fill_order = config["predict"]["fill_order"]
        fill_type = config["predict"]["fill_type"]
        ds_names = config['dataset']['names']
        for ds_name in ds_names:
            if DataGenerator.check_train_test_exist(train_test_folder, ds_name, fill_type, fill_order):
                print(ds_name+" for partial filling exists")
                continue
            print("generating random filling dataset for " + ds_name)
            ds_config = config['dataset'][ds_name]
            col_types = ds_config['field_types']
            field_order = list(col_types.keys())
            ds_splitter = ds_config['splitter']  # splitter of the original dataframe
            raw_df = pd.read_csv(raw_folder + ds_name + ".csv", ds_splitter)
            train_set, validate_set, test_set, col_val_id_df \
                = DataGenerator.raw_preprocessing(config, raw_df, col_types, field_order)

            pre_test_num = 0
            for i in range(1, len(test_set.columns)):
                test_instances = DataGenerator.gen_ds_partial_k_filled(test_set, col_val_id_df, col_types, k=i)
                if DataGenerator.not_enough_test(i, pre_test_num, len(test_instances), 0.05):
                    break
                DataGenerator.save_files(test_instances, ds_splitter, train_test_folder+fill_type+"/",
                                         ds_name+"_"+fill_type+"_test_"+str(i)+".csv")
                pre_test_num = len(test_instances)
            DataGenerator.save_files(train_set, ds_splitter, train_test_folder, ds_name+"_train.csv")
            DataGenerator.save_files(col_val_id_df, ds_splitter, train_test_folder, ds_name+"_val_id.csv")

    @staticmethod
    def gen_ds_incremental_raw(config):
        """
        generate raw datasets with different numbers of input instances
        Parameters:
            config: the configuration
        """
        raw_folder = config['dataset']['root_path'] + config['dataset']['raw_folder']
        rounds = config['predict']['rounds']  # number of rounds
        fill_type = config["predict"]["fill_type"]
        ds_names = config['dataset']['names']

        for ds_name in ds_names:
            ds_config = config['dataset'][ds_name]
            ds_splitter = ds_config['splitter']  # splitter of the original dataframe
            raw_df = pd.read_csv(raw_folder + ds_name + ".csv", ds_splitter)

            print("create incremental raw datasets for "+ds_name)
            counter = 1
            size = len(raw_df)  # size of the dataset in term of number of rows
            per_fold_size = int(size / rounds)+1  # fold size
            while counter <= rounds:
                fold_size = counter * per_fold_size + 1
                if fold_size > size:
                    fold_size = size
                fold_df = raw_df.head(fold_size)
                DataGenerator.save_files(fold_df, ds_splitter, raw_folder+fill_type+"/",
                                         ds_name+"_"+fill_type+"_" +str(counter)+".csv")
                print("size of fold: ", fold_size)
                counter = counter + 1

    @staticmethod
    def gen_ds_incremental(config):
        """
        generate datasets in an incremental manner; save the dataset to the folder
        Parameters:
            config: the configuration
        """
        DataGenerator.gen_ds_incremental_raw(config)
        train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']
        rounds = config['predict']['rounds'] # number of rounds
        fill_order = config["predict"]["fill_order"]
        fill_type = config["predict"]["fill_type"]
        ds_names = config['dataset']['names']

        for ds_name in ds_names:
            if DataGenerator.check_train_test_exist(train_test_folder, ds_name, fill_type, fill_order):
                print(ds_name + " for incremental filling exists")
                continue
            for r in range(1, rounds + 1):
                config_round = copy.deepcopy(config)
                ds_name_round = ds_name+"_"+fill_type+"_"+str(r)
                config_round['dataset']['names'] = [ds_name_round]
                config_round['dataset'][ds_name_round] = config['dataset'][ds_name]
                config_round['dataset']['train_test_folder'] = config['dataset']['train_test_folder']+fill_type+"/"
                config_round['dataset']['raw_folder'] = config['dataset']['raw_folder']+fill_type+"/"
                config_round['dataset']['train_ratio'] = 1.0 * (r-1)/r
                config_round['dataset']['test_ratio'] = 1- 1.0 * (r - 1) / r

                if fill_order == Utilities.T_ORDER_SEQ:
                    DataGenerator.gen_ds_seq(config_round)
                elif fill_order == Utilities.T_ORDER_RAND:
                    DataGenerator.gen_ds_rand(config_round)

    @staticmethod
    def gen_ds_sample_raw(config):
        """
        generate raw datasets with different numbers of input instances
        Parameters:
            config: the configuration
        """
        raw_folder = config['dataset']['root_path'] + config['dataset']['raw_folder']
        rounds = config['predict']['rounds']  # number of rounds
        fill_type = config["predict"]["fill_type"]
        ds_names = config['dataset']['names']

        for ds_name in ds_names:
            raw_src = raw_folder + ds_name + ".csv"
            # ds_config = config['dataset'][ds_name]
            # ds_splitter = ds_config['splitter']  # splitter of the original dataframe
            # raw_df = pd.read_csv(raw_folder + ds_name + ".csv", config['dataset'][ds_name])
            print("create sample raw datasets for " + ds_name)
            counter = 1
            while counter <= rounds:
                raw_dst = raw_folder + fill_type + "/" + ds_name+ "_"+fill_type+"_"+str(counter)+".csv"
                if not os.path.exists(raw_folder + fill_type + "/"):
                    os.makedirs(raw_folder + fill_type + "/")
                if not os.path.exists(raw_dst):
                    shutil.copy(raw_src, raw_dst)
                counter = counter + 1

    @staticmethod
    def gen_ds_sample(config):
        """
        generate datasets in an incremental manner; save the dataset to the folder
        Parameters:
            config: the configuration
        """
        DataGenerator.gen_ds_sample_raw(config)
        train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']
        rounds = config['predict']['rounds']  # number of rounds
        fill_order = config["predict"]["fill_order"]
        fill_type = config["predict"]["fill_type"]
        ds_names = config['dataset']['names']

        for ds_name in ds_names:
            if DataGenerator.check_train_test_exist(train_test_folder, ds_name, fill_type, fill_order):
                print(ds_name + " for sampling filling exists")
                continue
            for r in range(1, rounds + 1):
                config_round = copy.deepcopy(config)
                ds_name_round = ds_name + "_" + fill_type + "_" + str(r)
                config_round['dataset']['names'] = [ds_name_round]
                config_round['dataset'][ds_name_round] = config['dataset'][ds_name]
                config_round['dataset']['train_test_folder'] = config['dataset']['train_test_folder'] + fill_type + "/"
                config_round['dataset']['raw_folder'] = config['dataset']['raw_folder'] + fill_type + "/"
                # ratio to sample (from (round 1) 100% down to (round n) (1.0 - (r-1)/rounds)*100%)
                # every round removes about 1/rounds rows of instances
                # config_round['dataset']['train_sample'] = 1.0 - (r-1)/rounds
                config_round['dataset']['train_sample'] = r/rounds

                if fill_order == Utilities.T_ORDER_SEQ:
                    DataGenerator.gen_ds_seq(config_round)
                elif fill_order == Utilities.T_ORDER_RAND:
                    DataGenerator.gen_ds_rand(config_round)