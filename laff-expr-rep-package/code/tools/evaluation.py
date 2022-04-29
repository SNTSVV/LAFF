from __future__ import division
import numpy as np
import pandas as pd

from algorithms.prediction import Prediction
from tools.dataset_io import DataGenerator
from tools.utils import Utilities


class Eval:

    @staticmethod
    def mean_reciprocal_rank(rs):
        """
        Score is reciprocal of the rank of the first relevant item
        First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
        param rs: List of lists
        param rank of relevant item in each query
        returns: mean reciprocal rank
        """
        rs = (np.asarray(r).nonzero()[0] for r in rs)  # nonzero compute the indice of non zero elements
        return round(np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs]), 3)

    @staticmethod
    def r_precision(r):
        """
            Score is precision after all relevant documents have been retrieved
            Relevance is binary (nonzero is relevant).

            Parameters:
                r: List or numpy
                 Relevance scores in rank order
            Returns:
                Double
                     Precision
        """
        r = np.asarray(r) != 0
        z = r.nonzero()[0]
        if not z.size:
            return 0.
        return round(np.mean(r[:z[-1] + 1]), 3)

    @staticmethod
    def precision_at_k(r, k):
        '''
           Score is precision @ k
           Relevance is binary (nonzero is relevant).

           Parameters:
               r: List
                  Relevance score in a query
               k: integer
                  position where we want to compute precision
           Returns:
               Double
                    Precision at the position k
        '''
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return round(np.mean(r), 3)

    @staticmethod
    def recall_at_k(len_test, rs, k=10):
        """
            By default, we only recommend at most 10 candidate items
            Score is recall @ k
            Relevance is binary (nonzero is relevant).

           Parameters:
               r: List
                  Relevance score in a query
               k: integer
                  position where we want to compute recall
           Returns:
                array
                    array that present recall from 1 to k
        """
        hit_count = np.zeros(k, dtype=int)
        for r in rs:
            for i in range(0, len(r)):
                if i < len(hit_count):
                    hit_count[i] += r[i]
        for i in range(0, len(hit_count) - 1):
            hit_count[i + 1] = hit_count[i] + hit_count[i + 1]
        return hit_count / (1.0 * len_test)

    @staticmethod
    def average_precision(r):
        """
            Score is average precision (area under PR curve)
            Relevance is binary (nonzero is relevant).

            Parameters:
                r: List
                   List of ranked items of a query
            Returns:
                Double
                    Average precision of the query

        """
        r = np.asarray(r) != 0
        out = [Eval.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return round(np.mean(out), 3)

    @staticmethod
    def mean_average_precision(rs):
        '''
            Score is mean average precision
            Relevance is binary (nonzero is relevant).

            Parameters:

                rs: List
                    List of ranked lists of the different predictions made
            Returns:
                Double
                    Mean Average Precision
        '''
        return round(np.mean([Eval.average_precision(r) for r in rs]), 3)

    @staticmethod
    def rank_to_rs(ranked, y_true):
        """
        This function compare between the true values and the predicted one and create
        a list that represents the position of true value in the ranked list as 1 and the
        others as 0
        param ranked: list of ranked lists resulted of our methods
        param y_true: list of true values in the test set
        return rs: ranking list where 1 represents the index of the true value in our prediction for each query
        """
        rs = []
        for i in range(len(ranked)):
            x = []
            for j in range(len(ranked[i])):
                if y_true[i] == ranked[i][j]:
                    x.append(1)
                else:
                    x.append(0)
            rs.append(x)
        return rs

    @staticmethod
    def detect_outliers(values):
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        new_values = []
        outlier_step = 10 * IQR
        for nu in values:
            if (nu < Q1 - outlier_step) | (nu > Q3 + outlier_step):
                pass
            else:
                new_values.append(nu)
        return new_values

    @staticmethod
    def prediction_time(prediciton_time_list):
        """
        calculate the prediction time
        """
        time = Utilities.check_valid_time(prediciton_time_list)
        return np.mean(time), np.min(time), np.max(time)

    @staticmethod
    def hit_rate(true_values, ranked_values, remained_flags, use_filter, recommend_num):
        """
        calculate the hit rate
        """
        num_of_hit = 0
        num_of_predict = 0
        for count in range(0, len(remained_flags)):
            remained_flag = remained_flags[count]
            true_value = true_values[count]
            ranked_value = ranked_values[count]

            if use_filter and remained_flag == True:
                num_of_predict = num_of_predict + 1
                if true_value in ranked_value[0:recommend_num]:
                    num_of_hit = num_of_hit + 1
            elif not use_filter:
                num_of_predict = num_of_predict + 1
                if true_value in ranked_value[0:recommend_num]:
                    num_of_hit = num_of_hit + 1
        if num_of_predict == 0:
            return 0
        else:
            return 1.0 * num_of_hit / num_of_predict

    @staticmethod
    def coverage_rate(remained_flags, use_filter):
        coverage_all = len(remained_flags)
        coverage = 0
        if use_filter == False:
            coverage = len(remained_flags)
        else:
            for remained in remained_flags:
                if remained == True:
                    coverage = coverage + 1
        return coverage, coverage_all

    @staticmethod
    def eval_predict_details(details_path, raw_ds_path, splitter, alg, use_filter, recommend_ratio, exclued_targets):
        num_of_candidates = DataGenerator.calc_num_of_candidates(raw_ds_path, splitter)
        predict_details = pd.read_csv(details_path)

        avg_mrr = 0
        num_of_target = 0
        num_of_target_mrr = 0
        train_time_all = 0
        predict_time_all = 0
        prediciton_time_list = []
        coverage = 0
        need_all = 0
        hit_rate = 0

        for target, predict_details_target in predict_details.groupby('Target'):
            if target in exclued_targets:
                continue
            true_values = predict_details_target["Truth"].tolist()
            ranked_values = predict_details_target["Ranked"].tolist()
            ranked_values = Utilities.str_to_list(ranked_values)
            remained_flags = predict_details_target["Remained"].tolist()
            train_time = predict_details_target["Train"].tolist()
            predict_time = predict_details_target["Predict"].tolist()

            remained_values, remained_true_values = Prediction.get_filtered_results(remained_flags, ranked_values,
                                                                                    true_values, use_filter)

            c, c_all = Eval.coverage_rate(remained_flags, use_filter)
            coverage = coverage + c
            need_all = need_all + c_all

            results = Eval.rank_to_rs(remained_values, remained_true_values)
            mrr = Eval.mean_reciprocal_rank(results)
            if not np.math.isnan(mrr):
                num_of_target_mrr = num_of_target_mrr + 1
            else:
                mrr = 0
            train_time_all = train_time[0]
            predict_time_all = predict_time_all + sum(predict_time)
            prediciton_time_list.extend(predict_time)
            avg_mrr = avg_mrr + mrr
            recommend_num = round(num_of_candidates[target] * recommend_ratio)

            h_rate = Eval.hit_rate(true_values, ranked_values, remained_flags, use_filter, recommend_num)
            hit_rate = hit_rate + h_rate

            num_of_target = num_of_target + 1

        stats_df = pd.DataFrame(
            columns=['Algorithm', 'MRR', 'PCR', 'Train Time', 'Test time avg', 'Test time min', 'Test time max'])
        avg_mrr = avg_mrr / num_of_target_mrr
        p_time_avg, p_time_min, p_time_max = Eval.prediction_time(prediciton_time_list)
        row = {}
        row['Algorithm'] = alg
        row['MRR'] = avg_mrr
        row['PCR'] = 1.0 * coverage / need_all
        row['Train Time'] = train_time_all
        row['Test time avg'] = p_time_avg
        row['Test time min'] = p_time_min
        row['Test time max'] = p_time_max

        stats_df = stats_df.append(row, ignore_index=True)
        # print(stats_df)
        return stats_df

    @staticmethod
    def eval_predict_details_r4(details_path, raw_ds_path, splitter, alg, use_filter, recommend_ratio, exclued_targets):
        num_of_candidates = DataGenerator.calc_num_of_candidates(raw_ds_path, splitter)
        predict_details = pd.read_csv(details_path)

        avg_mrr = 0
        num_of_target = 0
        num_of_target_mrr = 0
        train_time_all = 0
        predict_time_all = 0
        prediciton_time_list = []
        coverage = 0
        need_all = 0
        hit_rate = 0
        row = {}
        stats_df = pd.DataFrame(columns=['Algorithm', 'Target', 'MRR'])

        for target, predict_details_target in predict_details.groupby('Target'):
            if target in exclued_targets:
                continue
            true_values = predict_details_target["Truth"].tolist()
            ranked_values = predict_details_target["Ranked"].tolist()
            ranked_values = Utilities.str_to_list(ranked_values)
            remained_flags = predict_details_target["Remained"].tolist()
            train_time = predict_details_target["Train"].tolist()
            predict_time = predict_details_target["Predict"].tolist()

            remained_values, remained_true_values = Prediction.get_filtered_results(remained_flags, ranked_values,
                                                                                    true_values, use_filter)

            c, c_all = Eval.coverage_rate(remained_flags, use_filter)
            coverage = coverage + c
            need_all = need_all + c_all

            results = Eval.rank_to_rs(remained_values, remained_true_values)
            mrr = Eval.mean_reciprocal_rank(results)
            if not np.math.isnan(mrr):
                num_of_target_mrr = num_of_target_mrr + 1
            else:
                mrr = 0
            train_time_all = train_time[0]
            predict_time_all = predict_time_all + sum(predict_time)
            prediciton_time_list.extend(predict_time)
            avg_mrr = avg_mrr + mrr
            recommend_num = round(num_of_candidates[target] * recommend_ratio)

            h_rate = Eval.hit_rate(true_values, ranked_values, remained_flags, use_filter, recommend_num)
            hit_rate = hit_rate + h_rate

            num_of_target = num_of_target + 1
            row['Algorithm'] = alg
            row['Target'] = target
            row['MRR'] = mrr
            row['PCR'] = 1.0 * c / c_all

            stats_df = stats_df.append(row, ignore_index=True)
        # print(stats_df)
        return stats_df
