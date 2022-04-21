import pandas as pd
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype


def get_train_test_data_split(data: pd.DataFrame, val_start: pd.Timestamp = pd.Timestamp('2020-06-27 00:00:00'),
                              test_days: int = 1):
    """
    function for split data in train and test by date
    :param data:
    :param val_start:
    :param val_days:
    :return:
    """
    train = data.loc[data['date'] < val_start, :].reset_index(drop=True)
    val_end = val_start + pd.Timedelta(test_days, unit='D')
    test = data.loc[(data['date'] >= val_start) & (data['date'] < val_end), :].reset_index(drop=True)
    return train, test


def pd_to_scr_matrix(data: pd.DataFrame):
    user_id_cats = CategoricalDtype(sorted(data.user.unique()), ordered=True)
    survey_id_cats = CategoricalDtype(sorted(data.survey.unique()), ordered=True)

    row = data.user.astype(user_id_cats).cat.codes
    col = data.survey.astype(survey_id_cats).cat.codes
    sparse_matrix = csr_matrix((data["status"], (row, col)),
                               shape=(user_id_cats.categories.size, survey_id_cats.categories.size))

    return sparse_matrix, user_id_cats, survey_id_cats
