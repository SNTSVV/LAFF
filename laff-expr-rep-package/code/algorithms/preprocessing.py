from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

from tools.utils import Utilities


class Preprocessing:

    @staticmethod
    def coerce_feature_types(df, col_types):
        """
        convert value of cells to unicode presentation
        Parameters:
            df:  the dataframe that the user is dealing with
            col_types: columns and types
        Returns:
           converted dataframe
        """
        columns = Utilities.get_cols_by_type(col_types, Utilities.T_DATA_CATE)
        for col in columns:
            df[col] = df[col].apply(Utilities.coerce_to_unicode)
        columns = Utilities.get_cols_by_type(col_types, Utilities.T_DATA_TEXT)
        for col in columns:
            df[col] = df[col].apply(Utilities.coerce_to_unicode)
        columns = Utilities.get_cols_by_type(col_types, Utilities.T_DATA_NUM)
        for col in columns:
            df[col] = df[col].apply(Utilities.coerce_to_unicode)
        # for col in columns:
        #     if df[col].dtype == np.dtype('M8[ns]' and env_type is not Utilities.T_ENV_PYTHON):
        #         from dataiku.doctor.utils import datetime_to_epoch
        #         df[col] = datetime_to_epoch(df[col])
        #         # print("hello")
        #     else:
        #         df[col] = df[col].astype('double')
        return df

    # ---REMOVE UNNECESSARY COLUMNS---
    @staticmethod
    def rm_empty_col(df, empty_ratio, col_types):
        """
        remove the empty columns of a dataframe
        Parameters:
            df:  the dataframe that the user is dealing with
        Returns:
           converted dataframe
        """
        num_of_record = df.shape[0]
        for col in df.columns:
            if not col_types[col] == Utilities.T_DATA_CATE:
                num_of_values = df[col].count()
                if 0 != num_of_record and (1 - 1.0 * num_of_values / num_of_record) > empty_ratio:
                    df = df.drop(col, axis=1)
        return df

    @staticmethod
    def rm_unique_col(df, unique_ratio, col_types):
        """
        remove columns that the number of unique values is larger than a ratio
        Parameters:
            df:  the dataframe that the user is dealing with
            unique_ratio: the ratio of unique values
        Returns:
           converted dataframe
        """
        for col in df.columns:
            if not col_types[col] == Utilities.T_DATA_CATE:
                num_of_unique = df[col].nunique()
                num_of_values = df[col].count()
                if num_of_values is 0:
                    df = df.drop(col, axis=1)
                elif 1.0 * num_of_unique / num_of_values > unique_ratio:
                    df = df.drop(col, axis=1)
        return df

    @staticmethod
    def rm_same_col(df):
        """
        remove columns that the value is the same
        Parameters:
            df:  the dataframe that the user is dealing with
        Returns:
           converted dataframe
        """
        for col in df.columns:
            num_of_unique = df[col].nunique()
            if num_of_unique is 1:
                df = df.drop(col, axis=1)
        return df

    @staticmethod
    def rm_mixed_col(df, col_types):
        """
        Revise columns with both numbers and categories, like '7.6, 5.5, c, 4.5, c'.
        if it has only one string value, this string could indicate null value
        if it has more than one string, remove this numerical column
        Parameter:
            df:  the dataframe that the user is dealing with
            col_types: type of the columns
        Returns:
           converted dataframe
        """
        columns = [col for col, col_type in col_types.items() if col_type == Utilities.T_DATA_NUM]
        for col in columns:
            categorical_parts = []
            counts = df[col].value_counts()
            for val, _ in counts.items():
                val = str(val)
                if not Utilities.is_number(val):
                    categorical_parts.append(val)
            if len(categorical_parts) == 0:
                continue
            # if it has only one categorical value (e.g., 'c'), we take it as an empty cell.
            elif len(categorical_parts) == 1:
                df[col].loc[df[col] == categorical_parts[0]] = np.nan
                df[col] = df[col].astype(float)
            # if it has several categorical values, we remove this column
            else:
                df = df.drop(col, axis=1)
        return df

    @staticmethod
    def rm_text_col(df, col_types):
        """
        remove all textual columns
        Parameter:
            df:  the dataframe that the user is dealing with
            col_types: type of the columns
        Returns:
           converted dataframe
        """
        for col, col_type in col_types.items():
            if col_type == Utilities.T_DATA_TEXT:
                df = df.drop(col, axis=1)
        return df

    # ---IMPUTE & RESCALE---
    @staticmethod
    def impute_with_mean(train_set, columns):
        """
        assign the mean value of a column to the empty cells in this column
        :param ds: data to process
        :param columns: valid columns
        """
        for col in columns:
            v = train_set[col].mean()
            train_set[col] = train_set[col].fillna(v)
        return train_set

    @staticmethod
    def impute_with_label(train_set, columns, label):
        """
        assign a predefined label to the empty cells in this column
        :param train_set: data to process
        :param columns: valid columns
        :param label: the impute value
        """
        for col in columns:
            # train_set[col][train_set[col]=="nan"]=label
            train_set[col] = train_set[col].fillna(label)
        return train_set

    @staticmethod
    def rescale_num_minmax(train_set, validation_set, test_set, columns):
        """
        rescale numerical columns with min-max values
        :param train_set: data to process
        :param validation_set: data to process
        :param test_set: data to process
        :param columns: list of numerical columns
        :return: the processed data
        """
        for col in columns:
            min_val = train_set[col].min()
            max_val = train_set[col].max()
            scale = max_val - min_val
            shift = min_val
            if scale != 0.:
                train_set[col] = (train_set[col] - shift).astype(np.float64) / scale
                validation_set[col] = (validation_set[col] - shift).astype(np.float64) / scale
                test_set[col] = (test_set[col] - shift).astype(np.float64) / scale
        return train_set, validation_set, test_set

    # ---DISCRETIZATION---
    @staticmethod
    def __rescale_input_vector(input_vector):
        """
        rescale the input_vector (a column) with MinMaxScaler
        :param input_vector:
        :return: the rescaled data, the scaler (MinMaxScaler)
        """
        scaler = MinMaxScaler()
        data = input_vector.reshape(-1, 1)
        scaler.fit(data)
        data = scaler.transform(data)
        data = [elem[0] for elem in data]
        return np.array(data), scaler

    @staticmethod
    def __continuous_to_clusters(input_vector, max_depth, eps=0.01):
        dt = DecisionTreeRegressor(max_depth=max_depth)
        clf = dt.fit(input_vector.reshape(-1, 1), input_vector.reshape(-1, 1))

        threshold = list(clf.tree_.threshold)
        while -2.0 in threshold: threshold.remove(-2.0)
        threshold = sorted(threshold)
        threshold.append(max(input_vector) + (2 * eps))
        threshold.insert(0, min(input_vector) - eps)

        regions = [(round(threshold[i], 4), round(threshold[i + 1] - eps, 4)) for i in range(len(threshold) - 1)]
        clusters = {}
        for reg in regions:
            clusters[reg] = [elem for elem in
                             input_vector[np.where(np.logical_and(input_vector >= reg[0], input_vector <= reg[1]))]]
        score = np.mean(
            [np.mean(
                (input_vector[np.where(np.logical_and(input_vector >= v[0], input_vector <= v[1]))]
                 - np.mean(input_vector[np.where(np.logical_and(input_vector >= v[0], input_vector <= v[1]))])) ** 2)
                for v in regions])
        return regions, clusters, score

    @staticmethod
    def __discretize(input_vector, max_max_depth=8, eps=0.001):
        original_vector = input_vector
        input_vector, scaler = Preprocessing.__rescale_input_vector(input_vector)
        last_error = 100
        last_regions = []
        for i in range(1, max_max_depth):
            regions, clusters, score = Preprocessing.__continuous_to_clusters(input_vector, i)
            variation = last_error - score
            if variation <= eps:
                break
            last_error = score
            last_regions = regions

        regions_rescaled = scaler.inverse_transform(np.array(last_regions))
        output_vector = np.zeros(len(input_vector)).astype(str)
        for v in regions_rescaled:
            output_vector[np.where(np.logical_and(original_vector >= v[0], original_vector <= v[1]))] = str(v)
        return output_vector, regions_rescaled

    @staticmethod
    def numerical_discretize(train_set, validation_set, test_set, columns):
        """
        transform a numerical column into several categories according to some ranges of the numbers
        the ranges are decided automatically with a decision tree
        :param train_set: data to process
        :param validation_set: data to process
        :param test_set: data to process
        :param columns: column ot process
        :return: the processed data
        """
        discretized_col_info = {}
        for col in columns:
            train_vector = np.array(train_set[col])
            train_discretized, regions_rescaled = Preprocessing.__discretize(train_vector)
            train_set[col] = train_discretized

            validation_vector = np.array(validation_set[col])
            validation_discretized = np.zeros(len(validation_vector)).astype(str)
            test_vector = np.array(test_set[col])
            test_discretized = np.zeros(len(test_vector)).astype(str)
            for v in regions_rescaled:
                validation_discretized[
                    np.where(np.logical_and(validation_vector >= v[0], validation_vector <= v[1]))] = str(v)
                test_discretized[np.where(np.logical_and(test_vector >= v[0], test_vector <= v[1]))] = str(v)
            validation_set[col] = validation_discretized
            test_set[col] = test_discretized
        return train_set, validation_set, test_set

    @staticmethod
    def map_col_with_id(train_set, validation_set, test_set, col_type):
        """
        Map values in a column into IDs, from 0 to n (for original values in categorical columns and discretized
        values in numerical columns). If a value does not appear in the training set, it is mapped to Type.EMPTY_C (-1)
        :param train_set: data to process
        :param validation_set: data to process
        :param test_set: data to process
        :param col_type: all columns
        :return: the processed data
        """
        def __map_col_with_id(train, valid, test, col, col_val_id_map):
            """mapping according to the training set"""
            train_cur, valid_cur, test_cur = pd.DataFrame(train, columns=[col]), \
                                             pd.DataFrame(valid, columns=[col]), pd.DataFrame(test, columns=[col])
            values = train_cur[col].value_counts()
            values = values.iloc[np.lexsort((values.index, -values.values))]
            tmp_val_id_map = dict(zip(values.keys(), [str(n) + "X" for n in range(0, len(values))]))
            if Utilities.T_DATA_EMPTY_C not in tmp_val_id_map:
                tmp_val_id_map[Utilities.T_DATA_EMPTY_C] = str(len(tmp_val_id_map))+"X"
            col_val_id_map[col] = tmp_val_id_map
            train_cur[col] = train_cur[col].map(col_val_id_map[col]).astype(str)
            unknow_label = col_val_id_map[col][Utilities.T_DATA_EMPTY_C]
            valid_cur[col] = valid_cur[col].map(col_val_id_map[col]).fillna(unknow_label).astype(str)
            test_cur[col] = test_cur[col].map(col_val_id_map[col]).fillna(unknow_label).astype(str)
            return np.array(train_cur[col]), np.array(valid_cur[col]), np.array(test_cur[col])

        col_val_id_map = {}
        for (col, cur_type) in col_type.items():
            train_set[col], validation_set[col], test_set[col] = \
                __map_col_with_id(train_set[col], validation_set[col], test_set[col], col, col_val_id_map)
        return train_set, validation_set, test_set, col_val_id_map

    @staticmethod
    def get_unknown_id(col_val_id):
        """
        get the id for unknown
        param col_val_id: val-id map of each field
        return unknown_id: field-unknown_id map
        """
        unknown_df = col_val_id[col_val_id["val"] == Utilities.T_DATA_EMPTY_C]
        ids = list(unknown_df["id"])
        fields = list(unknown_df["field"])
        unknown_id = dict(zip(fields, ids))
        return unknown_id
