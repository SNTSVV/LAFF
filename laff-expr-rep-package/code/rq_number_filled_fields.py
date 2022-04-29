import json
import pandas as pd
from tools.utils import Utilities, MyLogger
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import numpy as np

config_file = Utilities.get_config_file_path()
print(config_file)
with open(config_file)as cf:
    config = json.load(cf)

fill_orders = config['eval']['fill_orders']
algorithms = config['eval']['algorithms']
ds_names = config['dataset']['names']
ds_train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']
ds_splitter = config['dataset'][ds_names[0]]['splitter']
alg = config["predict"]["algorithm"]
fill_type = config["predict"]["fill_type"]

test_details = pd.read_csv(ds_train_test_folder + "partial/" + ds_names[0] + "_partial_test_" + str(1) + ".csv",
                           ds_splitter)
number_of_fields = len(test_details.columns) - 2  # 2: the target column and the target value column
# print (ds_names)
ds_train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']
# print (number_of_fields)

for ds_name in ds_names:
    alg_row = {}
    alg_row_pcr = {}
    alg_row_mean_mrr = {}
    alg_row_mean_pcr ={}
    state_df_mean_mrr = pd.DataFrame(columns=['NFF', 'mean', 'label'])
    state_df_mean_pcr = pd.DataFrame(columns=['NFF', 'mean', 'label'])

    outlier_mrr =[]
    alg_mrr = []
    NFF_mrr = []
    outlier_pcr =[]
    alg_pcr=[]
    NFF_pcr =[]

    for algorithm in algorithms:

        for fill_order in fill_orders:
            state_df_mrr = pd.DataFrame(columns=['lw','lq','med','uq','uw','NFF'])
            state_df_pcr = pd.DataFrame(columns=['lw', 'lq', 'med', 'uq', 'uw', 'NFF'])

            for i in range(number_of_fields - 1):
                try:
                    df = pd.read_csv(
                        ds_train_test_folder + "results/tmp/" + ds_names[0] + "_" + fill_type + "_" + str(
                            i) + "_" + algorithm + "_eval.csv")
                    ax = df.plot.box(figsize=(8, 6), showmeans=True)
                    ax.grid()
                    stats = boxplot_stats(df.MRR.values)
                    alg_row['lw'] = round(stats[0]["whislo"],3)
                    alg_row['lq'] = round(stats[0]["q1"],3)
                    alg_row['med'] = round(stats[0]["med"],3)
                    alg_row['uq'] = round(stats[0]["q3"],3)
                    alg_row['uw'] = round(stats[0]["whishi"],3)
                    alg_row['NFF'] = i

                    stats_pcr = boxplot_stats(df.PCR.values)
                    alg_row_pcr['lw'] = round(stats_pcr[0]["whislo"],3)
                    alg_row_pcr['lq'] = round(stats_pcr[0]["q1"],3)
                    alg_row_pcr['med'] = round(stats_pcr[0]["med"],3)
                    alg_row_pcr['uq'] = round(stats_pcr[0]["q3"],3)
                    alg_row_pcr['uw'] = round(stats_pcr[0]["whishi"],3)
                    alg_row_pcr['NFF'] = i

                    mrr=df['MRR'].tolist()
                    for v in mrr:
                        if v > stats[0]["whishi"] or v <stats[0]["whislo"]:
                            outlier_mrr.append(v)
                            alg_mrr.append(algorithm)
                            NFF_mrr.append(i)

                    alg_row_mean_mrr['NFF'] = i
                    alg_row_mean_mrr['mean'] = round(df["MRR"].mean(),3)
                    alg_row_mean_mrr['label'] = algorithm
                    alg_row_mean_pcr['NFF'] = i
                    alg_row_mean_pcr['mean'] = round(df["PCR"].mean(),3)
                    alg_row_mean_pcr['label'] = algorithm

                    pcr = df['PCR'].tolist()
                    for v in pcr:
                        if v > stats_pcr[0]["whishi"] or v < stats_pcr[0]["whislo"]:
                            outlier_pcr.append(v)
                            alg_pcr.append(algorithm)
                            NFF_pcr.append(i)


                except:
                    continue
                state_df_mrr = state_df_mrr.append(alg_row, ignore_index=True)
                state_df_mrr_t=state_df_mrr.T
                index=['LW', 'LQ', 'MED', 'UQ', 'UW', 'NFF']
                state_df_mrr_t['DIST']=index
                state_df_mrr_t.set_index('DIST')

                state_df_pcr = state_df_pcr.append(alg_row_pcr, ignore_index=True)
                state_df_pcr_t = state_df_pcr.T
                index = ['LW', 'LQ', 'MED', 'UQ', 'UW', 'NFF']
                state_df_pcr_t['DIST'] = index
                state_df_pcr_t.set_index('DIST')

                state_df_mean_mrr = state_df_mean_mrr.append(alg_row_mean_mrr, ignore_index=True)
                state_df_mean_pcr = state_df_mean_pcr.append(alg_row_mean_pcr, ignore_index=True)

        state_df_mrr_t=state_df_mrr_t[['DIST', 0, 1, 2]]
        state_df_mrr_t.columns = ['DIST', 'MRR1', 'MRR2', 'MRR3']
        state_df_mrr_t.drop(state_df_mrr_t.tail(1).index, inplace=True)  # drop last row
        state_df_mrr_t.to_csv(ds_train_test_folder + "results/rq4-num-fields-" + ds_names[0] + "_"
                        + algorithm + "_mrr_results" + ".csv", index=False)
                #state_df_pcr = state_df_pcr.append(alg_row_pcr, ignore_index=True)
        state_df_pcr_t = state_df_pcr_t[['DIST', 0, 1, 2]]
        state_df_pcr_t.columns = ['DIST', 'MRR1', 'MRR2', 'MRR3']
        state_df_pcr_t.drop(state_df_pcr_t.tail(1).index, inplace=True)  # drop last n rows
        state_df_pcr_t.to_csv(ds_train_test_folder + "results/rq4-num-fields-" + ds_names[0] + "_"
                                   + algorithm + "_pcr_results" + ".csv", index=False)

        state_df_mean_mrr=state_df_mean_mrr.drop_duplicates()
        state_df_mean_mrr=state_df_mean_mrr.sort_values(["NFF"], ascending=True)
        state_df_mean_mrr.to_csv(ds_train_test_folder + "results/rq4-num-fields-" + ds_names[0] +
                            "_mean_mrr_results" + ".csv",index=False)
        state_df_mean_pcr=state_df_mean_pcr.drop_duplicates()
        state_df_mean_pcr = state_df_mean_pcr.sort_values(["NFF"], ascending=True)
        state_df_mean_pcr.to_csv(ds_train_test_folder + "results/rq4-num-fields-" + ds_names[0] +
                                    "_mean_pcr_results" + ".csv", index=False)

        data_outlier_mrr = {'NFF': NFF_mrr, 'oulier': outlier_mrr, 'algorithm': alg_mrr}
        outlier_mrr_df = pd.DataFrame.from_dict(data_outlier_mrr)
        outlier_mrr_df=outlier_mrr_df.drop_duplicates()
        outlier_mrr_df.to_csv(ds_train_test_folder + "results/rq4-num-fields-" + ds_names[0] +
                                 "_outlier_mrr" + ".csv", index=False)

        data_outlier_pcr = {'NFF': NFF_pcr, 'oulier': outlier_pcr, 'algorithm': alg_pcr}
        outlier_pcr_df = pd.DataFrame.from_dict(data_outlier_pcr)
        outlier_pcr_df = outlier_pcr_df.drop_duplicates()
        outlier_pcr_df.to_csv(ds_train_test_folder + "results/rq4-num-fields-" + ds_names[0] +
                              "_outlier_pcr" + ".csv", index=False)

