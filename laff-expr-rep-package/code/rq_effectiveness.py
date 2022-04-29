# accuracy, usefulness, and running time
import json
import pandas as pd

from tools.utils import Utilities, MyLogger

config_file = Utilities.get_config_file_path()
print(config_file)
with open(config_file)as cf:
    config = json.load(cf)

fill_type = config['eval']['fill_type']
fill_orders = config['eval']['fill_orders']
algorithms = config['eval']['algorithms']

ds_names = config['dataset']['names']
ds_train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']
# rep_folder = config['dataset']['replication_folder_path']

stats_dataframe = pd.DataFrame(
    columns=['Algorithm', 'MRR-seq', 'PCR-seq', 'MRR-rand', 'PCR-rand', 'Train Time', 'Test time avg',
             'Test time min', 'Test time max'])

for ds_name in ds_names:
    for alg in algorithms:
        alg_row = {}
        for fill_order in fill_orders:
            df = pd.read_csv(ds_train_test_folder+"results/tmp/" + ds_name + "_" + fill_order + "_" + alg + "_eval.csv")
            alg_row['Algorithm'] = df["Algorithm"].values[0]
            alg_row['MRR-'+fill_order] = df["MRR"].values[0]
            alg_row['PCR-'+fill_order] = df["PCR"].values[0]
            alg_row['Train Time'] = df["Train Time"].values[0]/1000
            alg_row['Test time avg'] = df["Test time avg"].values[0]
            alg_row['Test time min'] = df["Test time min"].values[0]
            alg_row['Test time max'] = df["Test time max"].values[0]
        stats_dataframe = stats_dataframe.append(alg_row, ignore_index=True)

    print(stats_dataframe)
    stats_dataframe.to_csv(
        ds_train_test_folder + "results/rq1-2-effectiveness-" + ds_name + ".csv",
        index=False)
