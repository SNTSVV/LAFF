# coding: utf-8
import json
import time

import pandas as pd

from algorithms.preprocessing import Preprocessing
from tools.dataset_io import DataGenerator
from tools.utils import Utilities, MyLogger


class FLS:

    @staticmethod
    def alg(train_set_all, test_set_all, col_val_id, excluded_targets, my_logger):
        """
       the algorithm of MFM
       Paramter:
           train_set_all: instances in the training set
           test_set_all: instances in the testing set
           col_val_id: elements in file of col_val_id
           excluded_targets: fields that do not predict
       Return:
           predict_details: the detail of prediction for each test instance
       """
        unknown_ids = Preprocessing.get_unknown_id(col_val_id)
        print("id for null: "+str(unknown_ids))

        # get test_set according to the number of filled fields and conduct prediction
        test_per_target = DataGenerator.get_test_set_per_target(test_set_all)

        # start training
        # print(test_per_target)
        # print(len(test_set_part))
        my_logger.info("model building")
        options_info = {}
        id_val_info = {}
        val_id_info = {}
        for target, test_set in test_per_target.items():
            values = col_val_id[col_val_id['field'] == target]['val'].tolist()
            # rank the values alphabetically
            sorted_values = sorted(values, key=str.lower)
            if 'unknown' in sorted_values:
                position = sorted_values.index('unknown')
                del sorted_values[position]
            options_info[target] = sorted_values

            # col_val_id in a mapp
            col_val_id_field = col_val_id[col_val_id['field'] == target]
            tmp_id = {}
            tmp_val = {}
            for _, row in col_val_id_field.iterrows():
                tmp_id[row['id']] = row['val']
                tmp_val[row['val']] = row['id']
            id_val_info[target] = tmp_id
            val_id_info[target] = tmp_val
        # no training is needed
        train_start_t = time.time()
        train_end_t = time.time()
        time_to_train = int(round((train_end_t - train_start_t) * 1000))
        my_logger.info("time to train:")
        my_logger.info(time_to_train)
        # print(options_info)

        # [[m.get(ele, ele) for ele in lst] for lst in l]
        my_logger.info("prediction")
        predict_details = pd.DataFrame()  # prediction details for all targets
        for target, test_set in test_per_target.items():
            if target in excluded_targets:
                continue
            print("prediction on " + target)
            true_values = test_set[target].values  # get the true value from the target
            true_values_str = [id_val_info[target][val] for val in true_values]

            ranked_values = []
            time_predict = []
            index = 0
            for _, row in test_set.iterrows():
                # start prediction
                # image users know the value to fill, they can search by the first letter
                search_letter = true_values_str[index]
                search_letter_low = search_letter[0].lower()
                search_letter_up = search_letter[0].upper()
                predict_start_t = time.time()
                # search all the options start with the same first letter
                ranked_value_all = options_info[target]
                ranked_value = [val for val in ranked_value_all if val[0]==search_letter_low or val[0]==search_letter_up]
                # check if the searched results contain the true_value
                # print(search_letter)
                # print(ranked_value)

                # map the searched value back to the id
                ranked_value = [val_id_info[target][val] for val in ranked_value]
                # end prediction
                predict_end_t = time.time()
                ranked_values.append(ranked_value)  # retrieve values with a start letter
                index = index + 1
                time_predict.append(int(round((predict_end_t - predict_start_t) * 1000)))

            none_input = [None for _ in true_values]
            true_input = [True for _ in true_values]
            predict_details_target \
                = Utilities.add_status_to_frame(target, true_values, ranked_values, true_input, time_to_train,
                                                time_predict)
            frames = [predict_details, predict_details_target]
            predict_details = pd.concat(frames, ignore_index=True)  # prediction details for all targets

        return predict_details
