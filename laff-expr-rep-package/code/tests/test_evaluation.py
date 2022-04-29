from __future__ import division
import pytest

from tools.evaluation import *





#test_mean_reciprocal_rank
@pytest.mark.parametrize('rs, result',
                         [
                             (
                              #Test case 1:
                              #Test Input:
                              [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]],
                              #Test output:
                              0.75),
                             (
                              #Test case2:
                              #Test input:
                              [[0, 0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              #Test output:
                              0)
                         ])
def test_mean_reciprocal_rank(rs,result):
    assert Eval.mean_reciprocal_rank(rs) == result

#test_r_precision
@pytest.mark.parametrize('r, result',
                         [
                             (
                              #Test case 1:
                              #Test Input:
                              [0, 0, 0, 1],
                              #Test output:
                              0.25),
                             (
                              #Test case2:
                              #Test Input:
                              [0, 0, 0, 0],
                              #Test output:
                              0),
                             ( #Test case3:
                              #Test Input:
                              [1,0,0],
                              #Test output:
                              1)
                         ])

def test_r_precision(r, result):
    assert Eval.r_precision(r) == result



@pytest.mark.parametrize('r, k, result',
                        [
                            ( #Test case 1:
                              #Test Input:
                             #ranked list
                             [0, 0, 1],
                             # k
                             1,
                             #Test output:
                             0),
                            (
                             #Test case 2:
                             #Test input:

                             [0, 0, 1], 4,
                             #Test output:
                             pytest.raises(ValueError))
                        ])

#test_precision_at_k
def test_precision_at_k(r, k, result):
    if k < len(r):
        assert Eval.precision_at_k(r, k) == result
    else:
        with pytest.raises(ValueError) as e:
            Eval.precision_at_k(r, k)
        assert str(e.value) == 'Relevance score length < k'

#test_recall_at_k
@pytest.mark.parametrize('len_test,rs,k, result',
                        [
                            (#Test case 1: ranked list and k bigger than 0 and less then len(rs)
                             #Test input:
                             #len_test
                             3,
                             #rs
                             [[1, 0, 0],[ 0,1, 0],[ 0, 0, 1]],
                             # k
                             3,
                             #Test output:
                             [0.3333333333333333, 0.6666666666666666, 1.0]),

                            #Test case 2:
                            #Test input:
                            (3,
                             [[1, 0, 0],[ 0,1, 0],[ 0, 0, 1]],
                             2,
                             #Test output:
                             [0.3333333333333333, 0.6666666666666666]),
                            (
                            #Test case 3: negatif k
                             #Test input:
                             3,
                             [[1, 0, 0],[ 0,1, 0],[ 0, 0, 1]],-3,
                             #Test output:
                             pytest.raises(ValueError))
                        ]
                        )

def test_recall_at_k(len_test,rs, k, result):
    if (k >=0):
        obtained_result =  Eval.recall_at_k(len_test, rs, k)
        obtained_result= obtained_result.tolist()
        diff = set(obtained_result) - set(result)
        assert diff== set()
    else:
        with pytest.raises(ValueError) as e:
            Eval.recall_at_k(len_test, rs, k)
        assert str(e.value) == 'negative dimensions are not allowed'




#test_average_precision
@pytest.mark.parametrize('rs, result',
                        [   #Test case
                            (
                            #Test input:
                            [1, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                            #Test output
                            0.783),
                        ]
                         )
def test_average_precision(rs, result):
    assert Eval.average_precision(rs) == result


#test_mean_average_precision
@pytest.mark.parametrize('rs, result',
                        [   #Test case:
                            (
                            #Test input:
                            [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]],
                            #Test output
                            0.783),
                            (
                            #Test input
                            [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1],[0]],
                            #Test output:
                            0.392)
                        ]
                        )
def test_mean_average_precision(rs, result):
    assert Eval.mean_average_precision(rs) == result


#test_rank_to_rs
@pytest.mark.parametrize('ranked, y_true, result',
                            [
                                (#Test case 1:
                                 #Test input:
                                 #ranked
                                 [[0,2,1],[2,1,0],[1,0,2],[1,0,2],[1,0,2]],
                                 # y_true
                                  [1, 2, 0, 2, 1],
                                 #Test output:
                                 [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
                            ])
def test_rank_to_rs(ranked, y_true, result):
    obtained_result= Eval.rank_to_rs(ranked,y_true)
    assert obtained_result == result