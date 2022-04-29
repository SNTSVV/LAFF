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
    columns=['Algorithm', 'MRR-seq', 'PCR-seq', 'MRR-rand', 'PCR-rand'])

for ds_name in ds_names:
    for alg in algorithms:
        alg_row = {}
        for fill_order in fill_orders:
            df = pd.read_csv(
                ds_train_test_folder + "results/tmp/" + ds_name + "_" + fill_order + "_" + alg + "_eval.csv")
            alg_row['Algorithm'] = alg
            alg_row['MRR-'+fill_order] = df["MRR"].values[0]
            alg_row['PCR-'+fill_order] = df["PCR"].values[0]
        # print(alg_row)
        stats_dataframe = stats_dataframe.append(alg_row, ignore_index=True)

    print(stats_dataframe)
    stats_dataframe.to_csv(
        ds_train_test_folder + "results/rq3-variants-" + ds_name + ".csv", index=False)
