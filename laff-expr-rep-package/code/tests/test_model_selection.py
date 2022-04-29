from __future__ import division
import numpy as np
import pytest

# from ml.Model_selection import  *


# test_mean_reciprocal_rank
from algorithms.prediction import Prediction


@pytest.mark.parametrize('a,b, result',
                         [
                             (  # Test case 1:
                                     # Test Input:
                                     # a
                                     [0, 46, 36, 40, 26, 32],
                                     # b
                                     [0, 14, 48, 68, 26, 49],
                                     # Test output:
                                     4),
                             (  # Test case 2:
                                     # Test Input:
                                     # a
                                     [0, 14, 48, 68, 26, 49],
                                     # b
                                     [0, 14, 48, 68, 26, 49],
                                     # Test output:
                                     0)
                         ])
def test_matching_dissimilaity(a, b, result):
    assert Prediction.mismatch_dist(a, b) == result


@pytest.mark.parametrize('test_val,centroids,results',
                         [
                             # Test case 1:
                             # Test input:
                             # test_val:
                             ([[1, 3, 4, 2], [2, 0, 1, 4], [1, 2, 3, 4], [2, 2, 4, 4]],
                              # centroids:
                              [[1, 2, 3, 4], [4, 3, 4, 2]],
                              # Test output:
                              [[3, 1], [3, 4], [0, 4], [2, 3]])
                         ])
def test_calc_distances(test_val, centroids, results):
    expected_list_of_distances = Prediction.calc_distances(test_val, centroids)
    assert results == expected_list_of_distances


@pytest.mark.parametrize('list_of_dist,results',
                         [
                             # Test case 1:
                             # Test input:
                             (
                                     # list_of_dist:
                                     [[4, 4, 5, 1, 3], [5, 3, 5, 2, 4], [5, 3, 5, 2, 4], [4, 4, 5, 1, 3],
                                      [3, 3, 4, 1, 3], [4, 2, 4, 2, 4], [3, 3, 4, 2, 3], [4, 4, 5, 3, 3]],
                                     # Test output:
                                     [1, 1, 1, 1, 1, 2, 1, 2])
                         ])
def test_number_of_experts(list_of_dist, results):
    assert Prediction.num_of_experts(list_of_dist) == results


@pytest.mark.parametrize('list_of_dist, experts,results',
                         [
                             # Test case 1:
                             # Test input:
                             (
                                     # list_of_dist:
                                     [[1, 4, 5, 4, 3], [5, 2, 5, 3, 4], [5, 2, 5, 2, 4], [4, 4, 5, 1, 3],
                                      [3, 3, 4, 1, 3], [4, 2, 4, 2, 4], [3, 3, 4, 2, 3], [4, 4, 5, 3, 3]],
                                     # experts:
                                     [1, 1, 2, 1, 1, 2, 1, 2],
                                     # Test output:
                                     [0, 1, 5, 3, 3, 5, 3, 5])
                         ])
def test_model_selection(list_of_dist, experts, results):
    assert Prediction.select_model(list_of_dist, experts) == results
