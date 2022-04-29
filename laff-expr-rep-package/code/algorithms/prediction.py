import time


class Prediction:

    @staticmethod
    def prepare_test_set(test, target, unknown_ids):
        """
        get tue true value and the testing set for prediction (target is unknown)
        param test: original test set
        param target: current target
        param unknown_ids: the id of unknown for each target
        return true_values: list of true values
        return test_for_prediction: test set for prediction
        return test_evidences: test set for prediction (in dictionary)
        """
        true_values = test[target].values  # get the true value from the target
        test_for_prediction = test.copy(deep=True)
        test_for_prediction[target] = unknown_ids[target]  # set the target value as unknown
        test_for_prediction.drop("target", axis=1, inplace=True)  # eliminate the "target" column
        # convert test instances into evidence for BN inference {col:val}
        test_evidences = []
        cols = test_for_prediction.columns.tolist()
        for _, row in test_for_prediction.iterrows():
            evidence = row.to_dict()
            for col in cols:
                if evidence[col] == unknown_ids[col]:
                    del evidence[col]
            test_evidences.append(evidence)
        return true_values, test_for_prediction, test_evidences

    @staticmethod
    def model_selection(centroids, target, test_for_selection, bn_infer, test_evidences, use_local):
        """
        select model that will be used in prediction
        param centroids: the centroids of clusters
        param target: the name of the current target field
        param test_clustering: testing set used for model selection
        param bn_infer: (list of) the trained BN model for variable inference
        param test_evidences: test set for prediction (in dictionary)
        return selected_models: id of the selected model
        """
        test_list = test_for_selection.values.tolist()
        centroids_list = centroids.values.tolist()
        selected_models = []
        if not use_local:  # only predict with the global BN
            selected_models = [len(centroids_list) for n in range(len(test_list))]
        elif use_local:  # select models under different conditions
            dist_list = Prediction.calc_distances(test_list, centroids_list)
            num_experts = Prediction.num_of_experts(dist_list)
            selected_models = Prediction.select_model(dist_list, num_experts)

            # update the selected model
            # the local model should contain one evidence to predict the target
            for i in range(len(selected_models)):
                model_id = selected_models[i]
                if model_id == (len(bn_infer) - 1):  # it already uses the global model
                    continue
                evidence = test_evidences[i]
                nodes = bn_infer[model_id].model.nodes
                evi_key_count = 0
                for key, _ in evidence.items():
                    if key in nodes:
                        evi_key_count = evi_key_count + 1
                if evi_key_count == 0 or target not in nodes:
                    model_id = len(bn_infer) - 1
                selected_models[i] = model_id
        return selected_models

    @staticmethod
    def num_of_experts(dist_list):
        """
        compute the number of experts for each test instance
        param dist_list: distances to centroids of each test instance
        return num_experts: number of centroids that have minimum distance with the test instance
        """
        num_experts = []
        for elem in dist_list:
            minimum = min(elem)
            num_of_expert = 0
            for i in range(len(elem)):
                if elem[i] == minimum:
                    num_of_expert = num_of_expert + 1
            num_experts.append(num_of_expert)
        return num_experts

    @staticmethod
    def calc_distances(test_list, centroids_list):
        """
        compute mismatch distance between all testing instances and centroids
        param test_list: a list of testing instances
        param centroids: centroid of each cluster
        return dist_list: distances to centroids of each test instance
        """
        dist_list = []
        for test_instance in test_list:
            dist_to_centroids = []
            # measure distance between a given row and the different centroids
            for elem in centroids_list:
                dist_to_centroids.append(Prediction.mismatch_dist(test_instance, elem))
            dist_list.append(dist_to_centroids)
        return dist_list

    @staticmethod
    def mismatch_dist(a, b):
        """
        compute distances between two lists (the number of mismatch)
        param a, b: two lists
        return sum: number of mismatched elements in the two lists
        """
        dist = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                dist = dist + 1
        return dist

    @staticmethod
    def select_model(dist_list, num_experts):
        """
        select the model that needs to be used during the inference
        param dist_list: distances to centroids of each test instance
        param num_experts: number of centroids that have minimum distance with the test instance
        return model_id: id of the selected model
        """
        model_id = []
        for i in range(len(dist_list)):
            if num_experts[i] == 1:
                minimum = min(dist_list[i])
                model_id.append(dist_list[i].index(minimum))
            else:
                model_id.append(len(dist_list[i]))
        return model_id

    @staticmethod
    def predict(test_evidence, selected_models, cbel, target, bn_infer):
        """
        predict the probability distribution of test instances
        param test_evidence: test set for prediction in dictionary
        param selected_models: id of the selected model
        param cbel: constraint based estimators list (basic information for structure learning)
        param target: the current target field
        param bn_infer: the trained BN model for variable inference
        return predict_results: predicted probability distribution of test instances
        return time_predict: time for each prediction
        """
        predict_results = []
        test_evidence_real = []
        time_predict = []

        # print("the total number of test instances is " + str(len(test_evidence)))
        for counter, evidence in enumerate(test_evidence):

            predict_start_t = time.time()   # start prediction
            model_id = selected_models[counter]

            nodes = bn_infer[model_id].model.nodes
            #state = cbel[model_id].state_names
            state = cbel[model_id]
            evidence_in_nodes = {}  # check if the model has the values of input instances
            for key, val in evidence.items():
                if key in nodes and val in state[key]:
                    evidence_in_nodes[key] = val
            test_evidence_real.append(evidence_in_nodes)
            prob_distribution = bn_infer[model_id].query([target], evidence=evidence_in_nodes, show_progress=False)
            candidate_values = state[target]
            predicted_val_prob = list(zip(candidate_values, prob_distribution.values))
            predicted_val_prob = sorted(predicted_val_prob, key=lambda t: t[1], reverse=True)

            predict_end_t = time.time() # end prediction
            time_predict.append(int(round((predict_end_t - predict_start_t)*1000)))
            predict_results.append(predicted_val_prob)
        return predict_results, test_evidence_real, time_predict

    @staticmethod
    def get_ranked_values(predict_results, unknown_id):
        """
        get the possible values ranked by the probability distribution
        param predict_results: predicted probability distribution of test instances
        return ranked_values: ranked candidate values of each test instance
        return ranked_probability: ranked probability of each test instance
        """
        ranked_values = []
        ranked_probability = []
        for results in predict_results:
            ranked_value = []
            probability_value = []
            for result in results:
                if result[0] != unknown_id:
                    ranked_value.append(result[0])
                    probability_value.append(result[1])
            ranked_values.append(ranked_value)
            ranked_probability.append(probability_value)
        return ranked_values, ranked_probability

    @staticmethod
    def filter(target, bn_infer, ranked_distribution, test_evidences,
               selected_models, filter_th, recommend_num):
        """
        analyze whether a suggestion should be filtered or not
        param target: the current target field
        param bn_infer: (list of) the trained BN model for variable inference
        param ranked_distribution: ranked probability of each test instance
        param test_evidences: test set for prediction (in dictionary)
        param selected_models: id of the selected model
        param filter_th: the threshold to filter a result
        return remained_flags: a list of flag indicates which prediction should remain (True)
        """
        remained_flags = []
        for count in range(0, len(ranked_distribution)):
            # get the information of a single prediction
            distribution = ranked_distribution[count]
            model_id = selected_models[count]
            evidence = test_evidences[count]
            infer = bn_infer[model_id]

            # check whether the target directly depends on some fields in the evidence
            parent_flag = False
            evidence_keys = evidence.keys()
            parents = infer.model.get_parents(target)
            for parent in parents:
                if parent in evidence_keys:
                    parent_flag = True
                    break

            top1_prob = distribution[0]
            topk_prob = 0.0
            for i in range(0, len(distribution)):
                if i < recommend_num:
                    topk_prob = topk_prob + distribution[i]
            # filter based on the distribution of top1 and the parents_flag
            if (topk_prob > filter_th and parent_flag == False) \
                    or (top1_prob > 0.01 and parent_flag == True):  # at least the probability should not be almost zero
                remained_flags.append(True)
            else:
                remained_flags.append(False)

        return remained_flags

    @staticmethod
    def get_filtered_results(remained_flags, ranked_values, true_values, use_filter):
        """
        param remained_flags: a list of flag indicates which prediction should remain (True)
        param ranked_values: ranked candidate values of each test instance
        param true_values: list of true values
        param use_filter: whether use filter to remove some prediction
        return remained_values: the prediction for each not-filtered instance
        return remained_true_values: the ground truth for each not-filtered instance
        """
        remained_values = []
        remained_true_values = []
        if use_filter:
            for i in range(0, len(remained_flags)):
                remained_flag = remained_flags[i]
                if remained_flag == True:
                    remained_values.append(ranked_values[i])
                    remained_true_values.append(true_values[i])
        else:
            remained_values = ranked_values
            remained_true_values = true_values

        return remained_values, remained_true_values
