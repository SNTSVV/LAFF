import json
import pandas as pd

from tools.utils import Utilities, MyLogger

env_type = "python"
config_file = Utilities.get_config_file_path()
print(config_file)
with open(config_file)as cf:
    config = json.load(cf)

fill_orders = config['eval']['fill_orders']
rounds = config['predict']['rounds']
algorithms = config['eval']['algorithms']
ds_names = config['dataset']['names']
ds_train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']
ds_splitter = config['dataset'][ds_names[0]]['splitter']
alg = config["predict"]["algorithm"]
fill_type = config["predict"]["fill_type"]

test_details = pd.read_csv(ds_train_test_folder + "partial/" + ds_names[0] + "_partial_test_" + str(1) + ".csv",
                           ds_splitter)
number_of_fields = len(test_details.columns) - 2  # 2: the target column and the target value column
ds_train_test_folder = config['dataset']['root_path'] + config['dataset']['train_test_folder']

num_train_ncbi = 59284  # size of the original training set

for ds_name in ds_names:
    alg_row = {}
    for algorithm in algorithms:
        for fill_order in fill_orders:
            state_df = pd.DataFrame(columns=['Size', 'MRR', 'PCR'])
            for i in range(rounds):
                try:
                    # percentage of the data for each round e.g if we 4 rounds we will got 25% of the sample
                    df = pd.read_csv(ds_train_test_folder + "results/tmp/" + ds_names[0] + "_" + fill_type + "_"
                                     + str(i + 1) + "_" + fill_order + "_" + algorithm + "_eval.csv")
                    alg_row['MRR'] = df["MRR"].values[0]
                    alg_row['PCR'] = df["PCR"].values[0]
                    alg_row['Size'] = (num_train_ncbi * ((i + 1) / rounds))
                except:
                    continue
                # print (alg_row)
                state_df = state_df.append(alg_row, ignore_index=True)
                state_df.to_csv(ds_train_test_folder + "results/rq5-size-of-data-" + ds_names[0] + "_"
                                + algorithm + "_" + fill_order + ".csv", index=False)

