import os
import sys
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

BASE_PATH = \
    os.path.dirname(os.path.abspath(__file__)) + '/../analysis_for_kaggle'
sys.path.append(BASE_PATH)
import predict

TEST_PATH = f'{BASE_PATH}/../tests'


class TestPredict(unittest.TestCase):
    # titanic table binary classification
    def test_titanic(self):
        # gbdt
        result = predict.main(f'{TEST_PATH}/titanic/test_config.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output.csv'))
        assert_frame_equal(
            result['Y_pred_proba_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/proba_output.csv'))
        # lgb
        result = predict.main(f'{TEST_PATH}/titanic/test_config2.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output2.csv'))
        assert_frame_equal(
            result['Y_pred_proba_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/proba_output2.csv'))
        # torch
        result = predict.main(f'{TEST_PATH}/titanic/test_config3.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output3.csv'))
        # ensemble(stacking)
        result = predict.main(f'{TEST_PATH}/titanic/test_config4.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output4.csv'))
        # ensemble(vote)
        result = predict.main(f'{TEST_PATH}/titanic/test_config5.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output5.csv'))
        # pseudo
        result = predict.main(f'{TEST_PATH}/titanic/test_config6.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output6.csv'))
        # all folds
        result = predict.main(f'{TEST_PATH}/titanic/test_config7.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output7.csv'))
        # feature_selection
        result = predict.main(f'{TEST_PATH}/titanic/test_config8.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output8.csv'))
        # target encoding
        result = predict.main(f'{TEST_PATH}/titanic/test_config9.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output9.csv'))
        # error sampling
        result = predict.main(f'{TEST_PATH}/titanic/test_config10.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output10.csv'))
        # undersampling bagging
        result = predict.main(f'{TEST_PATH}/titanic/test_config11.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output11.csv'))
        # undersampling random
        result = predict.main(f'{TEST_PATH}/titanic/test_config12.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/titanic/output12.csv'))

    # house table regression
    def test_house(self):
        # svr
        result = predict.main(f'{TEST_PATH}/house/test_config.json')
        """
        assert_frame_equal(
            result['Y_pred_df'].round(0),
            pd.read_csv(f'{TEST_PATH}/house/output.csv').round(0))
        """
        # keras
        result = predict.main(f'{TEST_PATH}/house/test_config2.json')
        # ensemble(vote)
        result = predict.main(f'{TEST_PATH}/house/test_config3.json')

    # digit_part table multi lable classification
    def test_digit_part(self):
        # lgb
        result = predict.main(f'{TEST_PATH}/digit_part/test_config.json')
        # keras(lstm)
        result = predict.main(f'{TEST_PATH}/digit_part/test_config2.json')

    # cactus_part image binary classification
    def test_cactus_part(self):
        # keras(vgg16)
        result = predict.main(f'{TEST_PATH}/cactus_part/test_config.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/cactus_part/output.csv'))

    # disaster text binary classification
    def test_disaster(self):
        # tf-idf
        result = predict.main(f'{TEST_PATH}/disaster/test_config.json')
        assert_frame_equal(
            result['Y_pred_df'],
            pd.read_csv(f'{TEST_PATH}/disaster/output.csv'))
        # bert
        # result = predict.main(f'{TEST_PATH}/disaster/test_config2.json')


if __name__ == '__main__':
    unittest.main()
