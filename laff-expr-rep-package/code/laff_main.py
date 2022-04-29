# coding: utf-8
from __future__ import division
import json
import time

import pandas as pd
import numpy as np

from algorithms.clustering import Clustering
from algorithms.prediction import Prediction
from algorithms.preprocessing import Preprocessing
from algorithms.bayesian import *
from tools.dataset_io import DataGenerator
from tools.utils import Utilities, MyLogger


class LAFF:

    @staticmethod
    def alg(train_set_all, test_set_all, col_val_id, num_of_candidates, recommend_ratio, excluded_targets,
            model_root_path, alg_config, my_logger):
        """
        the algorithm of LAFF
        Paramter:
            train_set_all: instances in the training set
            test_set_all: instances in the testing set
            col_val_id: elements in file of col_val_id
            num_of_candidates: number of candidates for each field
            recommend_ratio: ratio of options for recommendation
            excluded_targets: fields that do not predict
            model_root_path: the path of the trained mode (bayesian network)
            alg_config: parameters for LAFF
        Return:
            predict_details: the detail of prediction for each test instance
        """

        unknown_ids = Preprocessing.get_unknown_id(col_val_id)
        print(unknown_ids)

        # start training
        train_start_t = time.time()
        model_path = model_root_path + "_global.dat"
        state_names0, bn_infer0, time0 = Bayesian.build_model(train_set_all, model_path)

        my_logger.info("edges0: " + str(bn_infer0.model.edges))
        main_fields = Bayesian.get_main_fields(bn_infer0)
        print("main field: " + str(main_fields))
        train_main = train_set_all[main_fields]

        # get the number of cluster and clustering
        n = Clustering.select_cluster_num(train_set_all, alg_config["min_cls_num"], alg_config["max_cls_num"])
        # n = 5
        print("the number of cluster is: ", n)

        # train BN on each cluster
        centroids, cluster_list, kmode_model, n = Clustering.cluster_on_fields(n, train_main, train_set_all)
        print(" the number of cluster is: ", n)
        model_path = model_root_path + "_local.dat"
        state_names, bn_infer, time_train_each = Bayesian.build_cluster_model(cluster_list, model_path)

        # end training
        train_end_t = time.time()
        time_train = int(round((train_end_t - train_start_t) * 1000))

        # get a list of BN models (include the model trained on all the data, which is at the end of the list)
        state_names.append(state_names0)
        bn_infer.append(bn_infer0)
        time_train_each.append(time0)
        my_logger.info("model edges:")
        for i in range(len(bn_infer)):
            print(bn_infer[i].model.edges)

        predict_details = pd.DataFrame()

        # if partial and importance are false, we run experiment in the original way
        # (one test set that contains different number of filled fields)
        my_logger.info("prediction")

        test_per_target = DataGenerator.get_test_set_per_target(test_set_all)

        for target, test_set in test_per_target.items():
            if target in excluded_targets:
                continue
            true_values, test_for_prediction, test_evidences = Prediction.prepare_test_set(test_set, target,
                                                                                           unknown_ids)

            selection_start_t = time.time()  # start selection
            test_for_selection = test_for_prediction[main_fields]
            selected_models = Prediction.model_selection(centroids, target, test_for_selection,
                                                         bn_infer, test_evidences, alg_config["use_local"])
            selection_end_t = time.time()  # start selection
            avg_selection_time = (int(round((selection_end_t - selection_start_t) * 1000))) / len(test_for_selection)

            print("prediction on " + target)
            predict_results, test_evidence_real, predict_time = Prediction.predict(test_evidences, selected_models,
                                                                                   state_names, target, bn_infer)
            filter_start_t = time.time()  # start selection
            recommend_num = int(num_of_candidates[target] * recommend_ratio)
            if recommend_num == 0:
                recommend_num = 1
            ranked_values, ranked_probability = Prediction.get_ranked_values(predict_results, unknown_ids[target])
            remained_flags = Prediction.filter(target, bn_infer, ranked_probability, test_evidences,
                                               selected_models, alg_config["filter_th"], recommend_num)
            filter_end_t = time.time()  # start selection
            avg_filter_time = (int(round((filter_end_t - filter_start_t) * 1000))) / len(test_for_selection)
            time_predict = list(np.array(predict_time) + avg_selection_time + avg_filter_time)

            # save prediction info to a file
            predict_details_target \
                = Utilities.add_status_to_frame(target, true_values, ranked_values, remained_flags, time_train,
                                                time_predict)
            frames = [predict_details, predict_details_target]
            predict_details = pd.concat(frames, ignore_index=True)  # testing instances for all targets

        return predict_details
