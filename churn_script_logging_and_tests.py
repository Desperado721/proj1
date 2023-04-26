"""
These 5 test cases are used to test functionality of churn library

Author: Jie Lyu
Creation Date: 2/12/2023
"""

import os
import logging
import pytest
import pandas as pd
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def bank_data_df():
    """
    dataset we used for training a model
    """
    bank_data_df = pd.read_csv("./data/bank_data.csv")
    return bank_data_df


@pytest.fixture
def cat_columns():
    """
    category columns that should be encode into numerical variables
    """
    return [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]


@pytest.fixture
def cat_columns_new():
    """
    Column names after group by Attrition_Flag
    """
    return ['Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn']


@pytest.fixture
def keep_cols():
    """
    The columns that can be used to train a model.
    """
    return ['Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
            'Income_Category_Churn', 'Card_Category_Churn']


@pytest.fixture(scope="module")
def new_bank_data_df(bank_data_df, cat_columns):
    """
    bank dataframe with catogoriy columns encoded into numerical ones.
    """
    new_bank_data_df = cls.encoder_helper(bank_data_df, cat_columns, "Churn")
    return new_bank_data_df


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        bank_data_df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert bank_data_df.shape[0] > 0
        assert bank_data_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(bank_data_df):
    '''
    test perform eda function
    '''
    try:
        cls.perform_eda(bank_data_df)
        assert os.path.isfile('./images/eda/churn.png')
        assert os.path.isfile('./images/eda/customer_age.png')
        assert os.path.isfile('./images/eda/normalize.png')
        assert os.path.isfile('./images/eda/total_trans_ct.png')
        assert os.path.isfile('./images/eda/corr.png')
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError:
        logging.error("Testing perform_eda: some file didn't exist")


def test_encoder_helper(bank_data_df, cat_columns, cat_columns_new):
    '''
    test encoder helper
    '''
    try:
        new_bank_data_df = cls.encoder_helper(bank_data_df, cat_columns, "Churn")
        assert new_bank_data_df.shape[0] > 0, "The output doesn't appear to have rows and columns"
        assert new_bank_data_df.shape[1] > 0, "The output doesn't appear to have rows and columns"
        for k in cat_columns_new:
            assert k in list(
                new_bank_data_df.columns), "The %s in the newly generated dataframes column" % k
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError:
        logging.error("Testing perform_eda: some file didn't exist")


def test_perform_feature_engineering(bank_data_df, cat_columns):
    '''
    test perform_feature_engineering

    '''
    new_bank_data_df = cls.encoder_helper(bank_data_df, cat_columns, "Churn")
    try:
        bank_x_train, bank_x_test, bank_y_train, bank_y_test = cls.perform_feature_engineering(
            new_bank_data_df, 'Churn')

        assert bank_x_train.shape[0] > 0, "The training set is not empty"
        assert bank_x_test.shape[0] > 0, "The testing dataset is not empty"
        assert bank_y_train.shape[0] > 0, "The label of training set is not empty"
        assert bank_y_test.shape[0] > 0, "The label of testing set is not empty"
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError:
        logging.error("Testing perform_feature_engineering: ERROR")


def test_train_models(bank_data_df, cat_columns):
    '''
    test train_models
    '''
    new_bank_data_df = cls.encoder_helper(bank_data_df, cat_columns, "Churn")
    bank_x_train, bank_x_test, bank_y_train, bank_y_test = cls.perform_feature_engineering(
        new_bank_data_df, 'Churn')
    try:
        cls.train_models(bank_x_train, bank_x_test, bank_y_train, bank_y_test)
        assert os.path.isfile('./images/results/cls_report_lr.png')
        assert os.path.isfile('./images/results/cls_report_rfc.png')
        assert os.path.isfile('./images/results/feature_rfc.png')
        assert os.path.isfile('./images/results/roc.png')
        logging.info("Testing train_models: SUCCESS")
    except AssertionError:
        logging.error("Testing train_models: ERROR")


if __name__ == "__main__":
    pass
