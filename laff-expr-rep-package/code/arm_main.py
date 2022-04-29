# coding: utf-8
from __future__ import division
import json
import pandas as pd
import time

from algorithms.association import RuleBased
from algorithms.preprocessing import Preprocessing
from tools.dataset_io import DataGenerator
from tools.utils import Utilities, MyLogger


class ARM:

    @staticmethod
    def alg(train_set_all, test_set_all, col_val_id, excluded_targets, alg_config, my_logger):
        """
       the algorithm of ARM
       Paramter:
           train_set_all: instances in the training set
           test_set_all: instances in the testing set
           col_val_id: elements in file of col_val_id
           excluded_targets: fields that do not predict
           alg_config: parameters for ARM
       Return:
           predict_details: the detail of prediction for each test instance
       """

        unknown_ids = Preprocessing.get_unknown_id(col_val_id)
        print(unknown_ids)

        # start training
        train_start_t = time.time()
        rules, _, _ = RuleBased.rule_generation(train_set_all, alg_config["minsup"], alg_config["minconf"])
        fnl_rules = RuleBased.rules_filtering(rules, unknown_ids)
        # end training
        train_end_t = time.time()
        time_train = int(round((train_end_t - train_start_t) * 1000))
        my_logger.info("time to train:")
        my_logger.info(time_train)

        # save rule as df: fnl_rules_df = pd.DataFrame(fnl_rules)
        my_logger.info("prediction")
        predict_details = pd.DataFrame()
        test_per_target = DataGenerator.get_test_set_per_target(test_set_all)
        for target, test_set in test_per_target.items():
            if target in excluded_targets:
                continue
            print("prediction on " + target)
            test_for_prediction, true_values = RuleBased.prepare_test_set(test_set, unknown_ids, target)
            matched_rules = RuleBased.rules_matching(fnl_rules,
                                                     target)  # get rules with the same consequent as the target
            matched_rules_df = pd.DataFrame(matched_rules)

            ranked_values = []
            time_predict = []
            remained_flags = []
            counter = 0
            for test_instance in test_for_prediction:
                predict_start_t = time.time()
                matched_rules_df = RuleBased.calc_matching_score(matched_rules, test_instance, matched_rules_df)
                ranked_value, not_zero = RuleBased.rule_ranking(target, test_instance, unknown_ids[target],
                                                                matched_rules_df)
                remained_flags.append(not_zero)
                # end prediction
                predict_end_t = time.time()
                ranked_values.append(ranked_value)  # retrieve the frequency-based ranking list
                time_predict.append(int(round((predict_end_t - predict_start_t) * 1000)))

            predict_details_target \
                = Utilities.add_status_to_frame(target, true_values, ranked_values, remained_flags, time_train,
                                                time_predict)
            frames = [predict_details, predict_details_target]
            predict_details = pd.concat(frames, ignore_index=True)  # testing instances for all targets

        return predict_details
