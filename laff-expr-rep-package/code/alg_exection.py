# coding: utf-8
import json

import pandas as pd

from arm_main import ARM
from laff_main import LAFF
from mfm_main import MFM
from fls_main import FLS
from naive_main import NAIVE
from tools.dataset_io import DataGenerator
from tools.evaluation import Eval
from tools.utils import Utilities, MyLogger

# configuration
config_file = Utilities.get_config_file_path()
with open(config_file)as cf:
    config = json.load(cf)
# parameters
my_logger = MyLogger(config['logging']["level"])
train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']
raw_folder = config['dataset']['root_path'] + config['dataset']['raw_folder']
ds_names = config['dataset']['names']
fill_order = config["predict"]["fill_order"]
fill_type = config["predict"]["fill_type"]
alg = config["predict"]["algorithm"]
recommend_ratio = config['predict']['recommend_ratio']
laff_param = config["laff_param"]
arm_param = config["arm_param"]
if "laff" in alg:
    eval_filter = laff_param['use_filter']
else:
    eval_filter = True
run_alg = config["eval"]["run_alg"]
run_eval = config["eval"]["run_eval"]

Utilities.create_necessary_folder(train_test_folder + "/model")
Utilities.create_necessary_folder(train_test_folder + "/results/details")
Utilities.create_necessary_folder(train_test_folder + "/results/tmp")


def execute_certain_alg():
    """
    execute an algorithm according to the algorithm name
    for example: it executes laff when variable 'alg' contains the string 'laff'
    """
    results = pd.DataFrame()
    if Utilities.T_ALG_MFM in alg:
        results = MFM.alg(train_set_all, test_set_all, col_val_id, exclued_targets, my_logger)
    if Utilities.T_ALG_FLS in alg:
        results = FLS.alg(train_set_all, test_set_all, col_val_id, exclued_targets, my_logger)
    elif Utilities.T_ALG_ARM in alg:
        results = ARM.alg(train_set_all, test_set_all, col_val_id, exclued_targets, arm_param, my_logger)
    elif Utilities.T_ALG_LAFF in alg:
        results = LAFF.alg(train_set_all, test_set_all, col_val_id, num_of_candidates, recommend_ratio,
                           exclued_targets, model_root_path, laff_param, my_logger)
    elif Utilities.T_ALG_NAIVE in alg:
        results = NAIVE.alg(train_set_all, test_set_all, col_val_id, exclued_targets, arm_param, my_logger)
    return results


for ds_name in ds_names:
    my_logger.info("current data set is: " + ds_name)
    my_logger.info("current algorithm: " + alg)

    raw_ds_path = raw_folder + ds_name + ".csv"
    model_root_path = train_test_folder + "model/" + ds_name
    ds_config = config['dataset'][ds_name]
    ds_splitter = ds_config['splitter']
    exclued_targets = ds_config['excluded_targets']
    num_of_candidates = DataGenerator.calc_num_of_candidates(raw_ds_path, ds_splitter)
    print("options per field: " + str(num_of_candidates))

    # run algorithms based on the fill_type (all, partial, incremental)
    if fill_type == Utilities.T_TYPE_ALL:
        details_path = train_test_folder + "results/details/" + ds_name + "_" + fill_order + "_" + alg + ".csv"
        if run_alg:  # run algorithm
            # generate the necessary datasets
            if fill_order == Utilities.T_ORDER_SEQ:
                DataGenerator.gen_ds_seq(config)
            elif fill_order == Utilities.T_ORDER_RAND:
                DataGenerator.gen_ds_rand(config)

            # read the training, tesing data
            train_path = train_test_folder + ds_name + "_train.csv"
            test_path = train_test_folder + ds_name + "_" + fill_order + "_test.csv"
            col_id_path = train_test_folder + ds_name + "_val_id.csv"
            train_set_all, test_set_all, col_val_id \
                = DataGenerator.read_train_test(train_path, test_path, col_id_path, ds_splitter)

            # run algorithm and save results
            predict_details = execute_certain_alg()
            predict_details.to_csv(details_path, index=False)

        if run_eval:  # evaluate algorithm
            stats_df = Eval.eval_predict_details(details_path, raw_ds_path, ds_splitter, alg, eval_filter,
                                                 recommend_ratio, exclued_targets)
            eval_result_path = train_test_folder + "results/tmp/" + ds_name + "_" + fill_order + "_" + alg + "_eval.csv"
            stats_df.to_csv(eval_result_path, index=False)

    elif fill_type == Utilities.T_TYPE_PART:
        DataGenerator.gen_ds_partial(config)
        matched_files = DataGenerator.matched_files_in_folder(train_test_folder + fill_type + "/",
                                                              ds_name + "_" + fill_type + "_test")
        print(matched_files)  # get all the testing set (the number of filled fields is different)
        for i in range(1, len(matched_files) + 1):
            details_path = train_test_folder + "results/details/" + ds_name + "_" + fill_type + "_" + str(
                i) + "_" + alg + ".csv"
            if run_alg:
                # read the training, tesing data
                train_path = train_test_folder + ds_name + "_train.csv"
                test_path = train_test_folder + fill_type + "/" + ds_name + "_" + fill_type + "_test_" + str(i) + ".csv"
                col_id_path = train_test_folder + ds_name + "_val_id.csv"
                train_set_all, test_set_all, col_val_id \
                    = DataGenerator.read_train_test(train_path, test_path, col_id_path, ds_splitter)

                # run algorithm and save results
                predict_details = execute_certain_alg()
                predict_details.to_csv(details_path, index=False)

            if run_eval:  # evaluate algorithm
                stats_df = Eval.eval_predict_details(details_path, raw_ds_path, ds_splitter, alg, eval_filter,
                                                     recommend_ratio, exclued_targets)
                eval_result_path = train_test_folder + "results/tmp/" + ds_name + "_" + fill_type + "_" + str(
                    i) + "_" + alg + "_eval.csv"
                stats_df.to_csv(eval_result_path, index=False)

    elif fill_type == Utilities.T_TYPE_INC or fill_type == Utilities.T_TYPE_SAMPLE:

        if fill_type == Utilities.T_TYPE_INC:
            DataGenerator.gen_ds_incremental(config)
        elif fill_type == Utilities.T_TYPE_SAMPLE:
            DataGenerator.gen_ds_sample(config)
        matched_files = DataGenerator.matched_files_in_folder(train_test_folder + fill_type + "/", fill_order + "_test")
        matched_files = [file for file in matched_files if ds_name in file]
        print(matched_files)  # get all the dataset (the number of input instances is different)

        for i in range(1, len(matched_files) + 1):
            print(i)
            details_path = train_test_folder + "results/details/" + ds_name + "_" + fill_type + "_" + str(
                i) + "_" + fill_order + "_" + alg + ".csv"
            if run_alg:  # run algorithm
                # read the training, tesing data, and the model trained on each data subset
                train_path = train_test_folder + fill_type + "/" + ds_name + "_" + fill_type + "_" + str(i) + "_train.csv"
                test_path = train_test_folder + fill_type + "/" + ds_name + "_" + fill_type + "_" + str(i) + "_" + fill_order + "_test.csv"
                col_id_path = train_test_folder + fill_type + "/" + ds_name + "_" + fill_type + "_" + str(i) + "_val_id.csv"
                model_root_path = train_test_folder + "model/" + ds_name + "_" + fill_type + "_" + str(i)
                train_set_all, test_set_all, col_val_id \
                    = DataGenerator.read_train_test(train_path, test_path, col_id_path, ds_splitter)

                # run algorithm and save results
                predict_details = execute_certain_alg()
                predict_details.to_csv(details_path, index=False)

            if run_eval:  # evaluate algorithm
                stats_df = Eval.eval_predict_details(details_path, raw_ds_path, ds_splitter, alg, eval_filter,
                                                     recommend_ratio, exclued_targets)
                eval_result_path = train_test_folder + "results/tmp/" + ds_name + "_" + fill_type + "_" + str(
                    i) + "_" + fill_order + "_" + alg + "_eval.csv"
                stats_df.to_csv(eval_result_path, index=False)
