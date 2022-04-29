import datetime
import os
import warnings

from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore


class Bayesian:

    @staticmethod
    def read_model(model_path):
        """
        read a saved model file
        param model_path: the path of the model file
        """
        model_file = open(model_path, 'r')
        lines = model_file.readlines()

        edges = []
        for line in lines:
            i = 0
            while i < len(line):
                if line[i] == '(':
                    end_index = line[i + 1:].index(')') + i
                    first, second = line[i + 1: end_index].split(',')
                    edges.append((first.strip().replace("'", ""), second.strip().replace("'", "")))
                    i = end_index + 1
                i = i + 1
        return edges

    @staticmethod
    def write_model(model_path, edges):
        """
        write a model to a file
        param model_path: the path of the model file
        param edges: the model
        """
        model_file = open(model_path, 'w')
        model_file.write(str(edges))

    @staticmethod
    def build_model(train, model_path):
        """
        Estimate the Bayesian network based on the training set
        param train: data for training the model (Dataframe)
        param model_path: the path to save the model
        return cbel: constraint based estimators list (basic information for structure learning)
        return bn_infer: the trained BN model for variable inference
        return time: time spent to create the model
        """
        warnings.simplefilter("ignore")
        print("training on dataset " + str(train.shape))
        start_t = datetime.datetime.now()

        # build the model from the mode file or from scratch
        if not os.path.exists(model_path):
            cbel = HillClimbSearch(train, scoring_method=BicScore(train))
            pgm = cbel.estimate()
            bn_model = BayesianModel(pgm.edges)
            Bayesian.write_model(model_path, pgm.edges)
        else:
            edges = Bayesian.read_model(model_path)
            bn_model = BayesianModel(edges)

        bn_model.fit(train)
        bn_infer = VariableElimination(bn_model)
        state_names = {}
        for cpd in bn_model.get_cpds():
            for key, val in cpd.state_names.items():
                state_names[key] = val
        end_t = datetime.datetime.now()
        time = end_t - start_t
        return state_names, bn_infer, time

    @staticmethod
    def get_main_fields(bn_infer):
        """
        determine the main fields
        param: probability_graph: directed acyclic graph of the model
        return main_field: a list of fields
        """
        main_field = set()
        for node in bn_infer.model.nodes():
            if not bn_infer.model.get_parents(node):
                main_field.add(node)
            # main_field.add(node)    # used for checking
        return list(main_field)

    @staticmethod
    def build_cluster_model(cluster_list, model_path):
        """
        builid a list of BN model on a list of dataset (cluster)
        param cluster_list: a list of dataset
        return cbel: (list of) constraint based estimators list (basic information for structure learning)
        return bn_infer: (list of) the trained BN model for variable inference
        return time: (list of) time spent to create the model
        """
        cbel, bn_model, bn_infer, time = [], [], [], []
        model_path = model_path[0:(len(model_path) - len(".dat"))]
        for i in range(len(cluster_list)):
            cbel_i, bn_infer_i, time_i = Bayesian.build_model(cluster_list[i], model_path + "_" + str(i) + ".dat")
            cbel.append(cbel_i)
            bn_infer.append(bn_infer_i)
            time.append(time_i)
            # print(time_i)
        return cbel, bn_infer, time
