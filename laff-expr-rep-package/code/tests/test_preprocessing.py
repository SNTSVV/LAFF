from __future__ import division
import pytest
import pandas as pd
import numpy as np
from algorithms.preprocessing import Preprocessing as proc

from pandas.testing import assert_frame_equal


# Different test cases (test input and expected outputs)
@pytest.mark.parametrize('df, col_types, ratio, result',
                         [  # remove empty columns
                             # Test case : Dataframe does not contain any empty column
                             # Test input:
                             (pd.DataFrame(data={'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                                                 'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
                                                 'age': [42, 52, 36, 24, 73],
                                                 'income': [10000, 24000, 31000, 20000, 30000],
                                                 'sex': ['M', 'F', 'F', 'M', 'F'],
                                                 'cat': [1, 2, 1, 2, 2]},
                                           columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat']),
                              dict(zip(['first_name', 'last_name', 'age', 'income', 'sex', 'cat'],
                                       ['text', 'text', 'text', 'text', 'text', 'text'])),
                              0.9,
                              # Expected output:
                              (pd.DataFrame(data={'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                                                  'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
                                                  'age': [42, 52, 36, 24, 73],
                                                  'income': [10000, 24000, 31000, 20000, 30000],
                                                  'sex': ['M', 'F', 'F', 'M', 'F'],
                                                  'cat': [1, 2, 1, 2, 2]},
                                            columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat']), []
                               )),
                             # Test case:  The dataframe contain one empty column
                             # Test input:
                             (pd.DataFrame(data={'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                                                 'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
                                                 'age': [42, 52, 36, 24, 73],
                                                 'income': [10000, 24000, 31000, 20000, 30000],
                                                 'sex': ['M', 'F', 'F', 'M', 'F'],
                                                 'cat': [np.nan, np.nan, np.nan, np.nan, np.nan]},
                                           columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat']),
                                dict(zip(['first_name', 'last_name', 'age', 'income', 'sex', 'cat'], ['text', 'text', 'text', 'text', 'text', 'text'])),
                              0.9,
                              # Expected output:
                              (pd.DataFrame(data={'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                                                  'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
                                                  'age': [42, 52, 36, 24, 73],
                                                  'income': [10000, 24000, 31000, 20000, 30000],
                                                  'sex': ['M', 'F', 'F', 'M', 'F']},
                                            columns=['first_name', 'last_name', 'age', 'income', 'sex']), ['cat']
                               )),

                             # Test case:  Empty dataframe
                             # Test input:
                             (pd.DataFrame(data={'first_name': [np.nan, np.nan, np.nan, np.nan, np.nan],
                                                 'last_name': [np.nan, np.nan, np.nan, np.nan, np.nan],
                                                 'age': [np.nan, np.nan, np.nan, np.nan, np.nan],
                                                 'income': [np.nan, np.nan, np.nan, np.nan, np.nan],
                                                 'sex': [np.nan, np.nan, np.nan, np.nan, np.nan],
                                                 'cat': [np.nan, np.nan, np.nan, np.nan, np.nan]},
                                           columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat']),
                                dict(zip(['first_name', 'last_name', 'age', 'income', 'sex', 'cat'], ['text', 'text', 'text', 'text', 'text', 'text'])),
                              0.9,
                              # Expected output:
                              (pd.DataFrame(columns=[], index=[0, 1, 2, 3, 4]),
                               ['first_name', 'last_name', 'age', 'income', 'sex', 'cat'])
                              )
                         ])
def test_rm_empty_col(df, col_types, ratio, result):
    result_df = proc.rm_empty_col(df, ratio, col_types)
    assert_frame_equal(result_df, result[0])


# test_remove_key_columns Different test cases (test input and expected outputs)
@pytest.mark.parametrize('df, col_types, ratio, result',
                         [
                             # Test case 1: Data frame does not contain key columns
                             # Test input:
                             (pd.DataFrame(data={'first_name': ['Jason', 'Tina', 'Tina', 'Jason', 'Amy'],
                                                 'last_name': ['Miller', 'Jacobson', 'Cooze', 'Miller', 'Cooze'],
                                                 'age': [52, 52, 73, 24, 73],
                                                 'income': [10000, 24000, 30000, 10000, 30000],
                                                 'sex': ['M', 'F', 'F', 'M', 'F'],
                                                 'cat': [1, 2, 1, 2, 2]},
                                           columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat']),
                                dict(zip(['first_name', 'last_name', 'age', 'income', 'sex', 'cat'],
                                         ['text', 'text', 'text', 'text', 'text', 'text'])),
                              0.9,
                              # Expected outputs:
                              # returned df
                              (pd.DataFrame(data={'first_name': ['Jason', 'Tina', 'Tina', 'Jason', 'Amy'],
                                                  'last_name': ['Miller', 'Jacobson', 'Cooze', 'Miller', 'Cooze'],
                                                  'age': [52, 52, 73, 24, 73],
                                                  'income': [10000, 24000, 30000, 10000, 30000],
                                                  'sex': ['M', 'F', 'F', 'M', 'F'],
                                                  'cat': [1, 2, 1, 2, 2]},
                                            columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat']),
                               # removed columns
                               []
                               )),

                             # Test case 2 : Dataframe that contains low variation columns
                             # Test input
                             (pd.DataFrame(data={'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                                                 'last_name': ['Miller', 'Jacobson', 'Ali', 'Miller', 'Cooze'],
                                                 'age': [42, 52, 36, 24, 73],
                                                 'income': [10000, 24000, 31000, 20000, 30000],
                                                 'sex': ['M', 'F', 'F', 'M', 'F'],
                                                 'cat': [1, 2, 1, 2, 2]},
                                           columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat']),
                                dict(zip(['first_name', 'last_name', 'age', 'income', 'sex', 'cat'],
                                         ['text', 'text', 'text', 'text', 'text', 'text'])),
                              0.9,
                              # Expected output:
                              # returned df
                              (pd.DataFrame(data={'last_name': ['Miller', 'Jacobson', 'Ali', 'Miller', 'Cooze'],
                                                  'sex': ['M', 'F', 'F', 'M', 'F'],
                                                  'cat': [1, 2, 1, 2, 2]},
                                            columns=['last_name', 'sex', 'cat']),
                               # removed columns
                               ['first_name', 'age', 'income']
                               )),

                             # Test case 3:  Dataframe that contains only key columns
                             # Test input:
                             (pd.DataFrame(data={'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                                                 'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
                                                 'age': [42, 52, 36, 24, 73],
                                                 'income': [10000, 24000, 31000, 20000, 30000],
                                                 'cat': [1, 2, 4, 3, 5]},
                                           columns=['first_name', 'last_name', 'age', 'income', 'cat']),
                                dict(zip(['first_name', 'last_name', 'age', 'income', 'sex', 'cat'],
                                         ['text', 'text', 'text', 'text', 'text', 'text'])),
                              0.9,
                              # Expected output
                              # returned df
                              (pd.DataFrame(columns=[], index=[0, 1, 2, 3, 4]),
                               # Removed columns
                               ['first_name', 'last_name', 'age', 'income', 'cat']))

                         ])
def test_rm_unique_col(df, col_types, ratio, result):
    result_ = proc.rm_unique_col(df, ratio, col_types)
    assert_frame_equal(result_, result[0])


# test_rm_mixed_col
@pytest.mark.parametrize('df, col_types, result',
                         [
                             #Test case 1: Data frame does not contain mixed columns
                             #Test input:
                             (pd.DataFrame(data={'first_name': ['Jason', 'Tina', 'Tina', 'Jason', 'Amy'],
                                                 'last_name': ['Miller', 'Jacobson', 'Cooze', 'Miller', 'Cooze'],
                                                 'age': [52, 52, 73, 24, 73],
                                                 'income': [10000, 24000, 30000, 10000, 30000],
                                                 'sex': ['M', 'F', 'F', 'M', 'F'],
                                                 'cat': [1, 2, 1, 2, 2]},
                                           columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat']),
                              {"first_name": "text", "last_name": "text", "age": "number", "income": "number",
                               "sex": "category", "cat": "number"},
                              # Expected outputs:
                              # returned df
                              (pd.DataFrame(data={'first_name': ['Jason', 'Tina', 'Tina', 'Jason', 'Amy'],
                                                  'last_name': ['Miller', 'Jacobson', 'Cooze', 'Miller', 'Cooze'],
                                                  'age': [52, 52, 73, 24, 73],
                                                  'income': [10000, 24000, 30000, 10000, 30000],
                                                  'sex': ['M', 'F', 'F', 'M', 'F'],
                                                  'cat': [1, 2, 1, 2, 2]},
                                            columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat']),
                               # removed columns
                               []
                               )),

                             # Test case 2 : Dataframe that contains one and two mixed columns
                             #Test input
                             (pd.DataFrame(data={'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                                                 'last_name': ['Miller', 'Jacobson', 'Ali', 'Miller', 'Cooze'],
                                                 'age': [42, 52, "c", 24, 73],
                                                 'income': [10000, 24000, "c", "a", 30000],
                                                 'sex': ['M', 'F', 'F', 'M', 'F'],
                                                 'cat': [1, 2, 1, 2, 2]},
                                           columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat']),
                              {"first_name": "text", "last_name": "text", "age": "number", "income": "number",
                               "sex": "category", "cat": "number"},
                              #Expected output:
                              #returned df
                              (pd.DataFrame(data={'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                                                 'last_name': ['Miller', 'Jacobson', 'Ali', 'Miller', 'Cooze'],
                                                 'age': [42, 52, np.nan, 24, 73],
                                                 'sex': ['M', 'F', 'F', 'M', 'F'],
                                                 'cat': [1, 2, 1, 2, 2]},
                                            columns=['first_name', 'last_name', 'age', 'sex', 'cat']),
                               #removed columns
                               ['first_name', 'age', 'income']
                               ))
                         ])
def test_rm_mixed_col(df, col_types, result):
    result_ = proc.rm_mixed_col(df, col_types)
    assert_frame_equal(result_, result[0])


@pytest.mark.parametrize('df,numerical_columns, result',
                         [  # impute with mean
                             # Test case :
                             # Test input:
                             (pd.DataFrame(data={'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                                                 'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
                                                 'age': [42, 52, 36, np.nan, 73],
                                                 'income': [10000, np.nan, 31000, 20000, 30000],
                                                 'sex': ['M', 'F', 'F', 'M', 'F'],
                                                 'cat': [1, 2, 1, 2, 2]},
                                           columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat']),
                              ['age', 'income'],
                              # Expected output:
                              (pd.DataFrame(data={'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                                                  'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
                                                  'age': [42, 52, 36, 50.75, 73],
                                                  'income': [10000.0, 22750.0, 31000.0, 20000.0, 30000.0],
                                                  'sex': ['M', 'F', 'F', 'M', 'F'],
                                                  'cat': [1, 2, 1, 2, 2]},
                                            columns=['first_name', 'last_name', 'age', 'income', 'sex', 'cat'])
                              ))])
def test_impute_with_mean(df, numerical_columns, result):
    resulted_df = proc.impute_with_mean(df, numerical_columns)

    assert_frame_equal(resulted_df, result)
