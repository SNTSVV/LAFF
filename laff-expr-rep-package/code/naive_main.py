
# coding: utf-8
from __future__ import division
import json
import pandas as pd
import time
from itertools import chain, combinations
from algorithms.association import RuleBased
from algorithms.preprocessing import Preprocessing
from tools.dataset_io import DataGenerator
from tools.utils import Utilities, MyLogger
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from algorithms.prediction import Prediction


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def determine_target_feature_combinations(columns):
    tf_combinations = []
    for target in columns:
        
        features = list(set(columns) - set([target]))
    
        powersets = list(powerset(features))

        for item in powersets :
            if list(item) == []: #skip empty sets
                continue
            item_list = list(item)
            item_list = [col for col in columns if col in item_list]
            tf_c = item_list + [target]
            tf_combinations.append(tf_c)
            
    return tf_combinations


def get_ranked_list(probabilities, classes):
    init_ranked = []
    init_prob = []
    for p in range(len(probabilities)):  # get only classess with probabilities bigger than 0
        if probabilities[p] > 0:
            init_prob.append(probabilities[p])
            init_ranked.append(classes[p])
    
    prediction_list = dict(zip(init_ranked, init_prob))
    ranked_list = sorted(prediction_list, key=prediction_list.get, reverse=True)
    return ranked_list


def get_dict(train_set):
    """
    Convert ID in the training set to values
    :param train_set: training set
    :return
    """
    dicts = []
    train_set_tmp = pd.DataFrame(columns=train_set.columns.values.tolist())
    for col in train_set.columns:
        train_set_tmp[col] = train_set[col].astype('category')
        dicts.append(dict(enumerate(train_set_tmp[col].cat.categories)))
    maps = zip(train_set_tmp.columns.tolist(), dicts)
    dict_map = dict(maps)
    return dict_map


def map_dataframe_values(train_set, dict_map, inverse):
    """
    Convert the values back to the ID
    :param train_set: training set
    :param dict_map: dictionnary of IDs
    :param inverse:
    :return
    """
    train_set_tmp = pd.DataFrame(columns=train_set.columns.values.tolist())
    for col in train_set.columns:
        val = train_set[col].values.tolist()
        if inverse == False:
            val_map = {v: k for k, v in dict_map[col].items()}
        else:
            val_map = dict_map[col]
        val = [val_map[k] for k in val]
        train_set_tmp[col] = val
    return train_set_tmp


class NAIVE:

    @staticmethod
    def alg(train_set_all, test_set_all, col_val_id, excluded_targets, alg_config, my_logger):
        """
        the algorithm of NAIVE
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
        columns = list(train_set_all.columns)
        # start training
        train_start_t = time.time()
        tf_combinations = determine_target_feature_combinations(columns)
        models = []
        keys = []
        for item in tf_combinations:
            keys.append('-'.join(item))
            train_set = train_set_all[item]  # select the train set columns based on each combination
            target = item[-1]  #get the target (the last element of the list )
            features = [val for val in item if val != target]
            dict_map = get_dict(train_set)  # get dictionary of mapped categories
            train_set = map_dataframe_values(train_set, dict_map, inverse=False)
            X_train = train_set[features]  # get the features
            y_train = train_set[[target]]
            clf = DecisionTreeClassifier()   # Create Decision Tree classifer object
            clf = clf.fit(X_train,y_train)   # Train Decision Tree Classifer
            models.append(clf)
        models_dict = dict(zip(keys, models))
        # end training
        train_end_t = time.time()
        time_to_train = int(round((train_end_t - train_start_t) * 1000))
        my_logger.info("time to train:")
        my_logger.info(time_to_train)

        my_logger.info("prediction")
        predict_details = pd.DataFrame()
        test_per_target = DataGenerator.get_test_set_per_target(test_set_all)
        for target, test_set in test_per_target.items():
            if target in excluded_targets:
                continue

            true_values, test_for_prediction, test_evidences = Prediction.prepare_test_set(test_set, target, unknown_ids)
            ranked_values=[]
            #model selection
            time_predict = []
            for item in test_evidences:
                
                evidence = [col for col in columns if col in item.keys()]  # get the list of features in the evidence

                evidence_target = evidence + [target]  # add the target in the evidence
                evidence_key = '-'.join(evidence_target) #create string contain the features and the target based on the evidence to check for the model
                if evidence_key in list(models_dict.keys()): #check if the test instance order can be found in the model_dicts
                    clf = models_dict[evidence_key]

                    X_test = pd.DataFrame(columns=evidence)  # the evidence
                    X_test = X_test.append(item, ignore_index=True)
                    X_test = map_dataframe_values(X_test, dict_map, inverse=False)
                    predict_start_t = time.time()
                    y = clf.predict_proba(X_test)

                    classes = clf.classes_
                    probabilities = y[0]
                    ranked_list = get_ranked_list(probabilities, classes)
                    mapped_ranked_list = []
 
                    for item in ranked_list:
                        mapped_ranked_list.append(dict_map[target][item])
                    predict_end_t = time.time()
                    ranked_values.append(mapped_ranked_list)
                    time_predict.append(int(round((predict_end_t - predict_start_t) * 1000)))
                    
                else: #if not try to look if the different permutations can be found
                    print("no", evidence_key)

            none_input = [None for _ in true_values]
            true_input = [True for _ in true_values]
            predict_details_target \
                = Utilities.add_status_to_frame(target, true_values, ranked_values, true_input, time_to_train,
                                            time_predict)
            frames = [predict_details, predict_details_target]
            predict_details = pd.concat(frames, ignore_index=True)  # prediction details for all targetss


        return predict_details










