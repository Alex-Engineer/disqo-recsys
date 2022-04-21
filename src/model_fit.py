"""
Fit model using all data
"""
from typing import List
import joblib
import pandas as pd
from lightfm import LightFM
import os
from src.utils import pd_to_scr_matrix
from pandas.api.types import CategoricalDtype
import pickle
import logging

logger = logging.getLogger(__file__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s  %(processName)-10s  %(name)s - %(levelname)s: %(message)s'))
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def prepare_data(path_to_data='../data/recsys_data.csv'):
    logger.info('prepare data')
    data = pd.read_csv(path_to_data)
    data['date'] = pd.to_datetime(data['date'])

    most_popular_items = list(data.loc[data['status'] == 1, 'survey'].value_counts().index[0:3])

    data_user_status = data.groupby(['user']).agg({'status': ['mean', 'count']})
    data_user_status.columns = ['mean_status', 'count_status']

    bad_users = list(data_user_status.loc[(data_user_status['mean_status'] == 0) &
                                          (data_user_status['count_status'] > 100), :].index)

    data_survey_status = data.groupby(['survey']).agg({'status': ['mean', 'count']})
    data_survey_status.columns = ['mean_status', 'count_status']
    bad_surveys = list(data_survey_status.loc[(data_survey_status['mean_status'] == 0) &
                                              (data_survey_status['count_status'] > 100), :].index)

    data = data.loc[data['user'].isin(bad_users) == False, :]
    data = data.loc[data['survey'].isin(bad_surveys) == False, :]

    return data, most_popular_items


def fit_model(data, epochs=30):
    train_matrix, user_id_cats, survey_id_cats = pd_to_scr_matrix(data)
    model = LightFM(loss='warp')
    model.fit(train_matrix, epochs=epochs)
    logger.info('Model is fitted')
    return model, user_id_cats, survey_id_cats


def save_model(model: LightFM, user_id_cats: CategoricalDtype, survey_id_cats: CategoricalDtype,
               most_popular_items: List[int], models_path: str = '../data/models', model_name: str = 'model.joblib'):
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    model_path = os.path.abspath(os.path.join(models_path, model_name))
    joblib.dump(model, model_path)

    path_to_user_id_cats = os.path.abspath(os.path.join(models_path, 'user_id_cats.pickle'))
    path_to_survey_id_cats = os.path.abspath(os.path.join(models_path, 'survey_id_cats.pickle'))

    with open(path_to_user_id_cats, 'wb') as file:
        pickle.dump(user_id_cats, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path_to_survey_id_cats, 'wb') as file:
        pickle.dump(survey_id_cats, file, protocol=pickle.HIGHEST_PROTOCOL)

    path_to_most_popular_items = os.path.abspath(os.path.join(models_path, 'most_popular_items.pickle'))
    with open(path_to_most_popular_items, 'wb') as file:
        pickle.dump(most_popular_items, file, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('model and data saved')


def main():
    data, most_popular_items = prepare_data()
    model, user_id_cats, survey_id_cats = fit_model(data)
    save_model(model, user_id_cats, survey_id_cats, most_popular_items)


if __name__ == "__main__":
    main()
