# coding: utf-8
import json
import time

import pandas as pd

from algorithms.preprocessing import Preprocessing
from tools.dataset_io import DataGenerator
from tools.utils import Utilities, MyLogger


class MFM:

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
        train_start_t = time.time()
        most_freq_info = {}
        for target, test_set in test_per_target.items():
            # rank the values based on the frequency
            most_freq = train_set_all[target].value_counts().index.tolist()
            unknown = unknown_ids[target]
            if unknown in most_freq:
                position = most_freq.index(unknown)
                del most_freq[position]
            most_freq_info[target] = most_freq
        # end training
        train_end_t = time.time()
        time_to_train = int(round((train_end_t - train_start_t) * 1000))
        my_logger.info("time to train:")
        my_logger.info(time_to_train)

        my_logger.info("prediction")
        predict_details = pd.DataFrame()  # prediction details for all targets
        for target, test_set in test_per_target.items():
            if target in excluded_targets:
                continue
            print("prediction on " + target)
            true_values = test_set[target].values  # get the true value from the target
            ranked_values = []
            time_predict = []
            for _, row in test_set.iterrows():
                # start prediction
                predict_start_t = time.time()
                ranked_value = most_freq_info[target]
                # end prediction
                predict_end_t = time.time()
                ranked_values.append(ranked_value)  # retrieve the frequency-based ranking list
                time_predict.append(int(round((predict_end_t - predict_start_t) * 1000)))

            none_input = [None for _ in true_values]
            true_input = [True for _ in true_values]
            predict_details_target \
                = Utilities.add_status_to_frame(target, true_values, ranked_values, true_input, time_to_train,
                                                time_predict)
            frames = [predict_details, predict_details_target]
            predict_details = pd.concat(frames, ignore_index=True)  # prediction details for all targets

        return predict_details
