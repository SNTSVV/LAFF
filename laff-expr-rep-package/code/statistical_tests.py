from __future__ import division

# coding: utf-8
import pandas as pd
import json
import numpy
from ast import literal_eval
from tools.evaluation import Eval
from tools.utils import Utilities
from scipy.stats import mannwhitneyu


def value_type(val_list, val_type):
    results = []
    for item in val_list:
        if val_type == "string":
            results.append(str(item))
        elif val_type == "num":
            results.append(item)
    return results


def get_col_values(df, col, col_type):
    """
    Get the value of a col and convert them to a given type
    :param df: results dataframe
    :col: the column
    :col_type: the target type
    :return: a list of values in the column
    """
    values = df[col].values.tolist()
    # print(values)
    final_values = []
    for item in values:  # from string to list
        item_val = literal_eval(item)
        item_val_converted = value_type(item_val, col_type)
        final_values.append(item_val_converted)

    return final_values


def get_avg_prec(df):
    """
    Compute average precision for each row in the dataframe

    :param df: results dataframe
    :return: av_prec list of average precision for each test instance
    """
    true_class = df['Truth'].values.tolist()
    ranked_list = get_col_values(df, "Ranked", "string")
    # print (true_class)
    # print (ranked_list)
    result = []
    for i in range(len(true_class)):
        rank = []
        for j in range(len(ranked_list[i])):
            if true_class[i] == ranked_list[i][j]:
                rank.append(1)
            else:
                rank.append(0)
        result.append(rank)
    avg_prec = []
    for item in result:
        ap = Eval.average_precision(item)
        avg_prec.append(ap)
    df['Average precision'] = avg_prec
    return avg_prec


config_file = Utilities.get_config_file_path()
print(config_file)
with open(config_file)as cf:
    config = json.load(cf)
ds_train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']
ds_names = config['dataset']['names']
fill_order = config['predict']['fill_order']

# calculate the average precision of different algorithms
laff_results = pd.read_csv(ds_train_test_folder + "results/details/" + ds_names[0] + "_" + fill_order + "_laff.csv")
laff_results_remained = laff_results.loc[laff_results['Remained'] == True]
avg_prec_laff = get_avg_prec(laff_results_remained)
# print (avprec_laff)
mfm_results = pd.read_csv(ds_train_test_folder + "results/details/" + ds_names[0] + "_" + fill_order + "_mfm.csv")
avg_prec_mfm = get_avg_prec(mfm_results)

arm_results = pd.read_csv(ds_train_test_folder + "results/details/" + ds_names[0] + "_" + fill_order + "_arm.csv")
avg_prec_arm = get_avg_prec(arm_results)

fls_results = pd.read_csv(ds_train_test_folder + "results/details/" + ds_names[0] + "_" + fill_order + "_fls.csv")
avg_prec_fls = get_avg_prec(fls_results)

naive_results = pd.read_csv(ds_train_test_folder + "results/details/" + ds_names[0] + "_" + fill_order + "_naive.csv")
avg_prec_naive = get_avg_prec(naive_results)

avg_prec_laff = numpy.array(avg_prec_laff)
avg_prec_mfm = numpy.array(avg_prec_mfm)
avg_prec_arm = numpy.array(avg_prec_arm)
avg_prec_fls = numpy.array(avg_prec_fls)
avg_prec_naive = numpy.array(avg_prec_naive)
# mannwhitneyu test
stat, p = mannwhitneyu(avg_prec_laff, avg_prec_arm)
print('stat=%.3f, p=%.3f' % (stat, p))

print(" LAFF vs ARM :", fill_order)
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

stat, p = mannwhitneyu(avg_prec_laff, avg_prec_mfm)
print('stat=%.3f, p=%.3f' % (stat, p))

print(" LAFF vs MFM ", fill_order)
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')


stat, p = mannwhitneyu(avg_prec_laff, avg_prec_fls)
print('stat=%.3f, p=%.3f' % (stat, p))

print(" LAFF vs FLS ", fill_order)
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

stat, p = mannwhitneyu(avg_prec_laff, avg_prec_naive)
print('stat=%.3f, p=%.3f' % (stat, p))

print(" LAFF vs naive ", fill_order)
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')