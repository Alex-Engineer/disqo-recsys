"""
module for inference
"""
import os
import pickle
from typing import List
import numpy as np
import joblib
from lightfm import LightFM
import pandas as pd
import logging

logger = logging.getLogger(__file__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s  %(processName)-10s  %(name)s - %(levelname)s: %(message)s'))
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class RecModel:
    def __init__(self, models_path, model_name: str = 'model.joblib'):
        self.model, self.user_id_cats, self.survey_id_cats, self.most_popular_items = self.load_model(models_path)
        logger.debug('Load model')
        self.data = self.load_data()
        logger.debug('Load data')
        self.n_items = len(self.survey_id_cats)

    def predict(self, user: int, k: int = 3):
        previous_seen_items = set(self.data.loc[self.data['user'] == user, 'survey'])
        logger.debug(f"Previous seen items for user {user}: {previous_seen_items}")
        if user in self.user_id_cats:
            logger.debug('Use Lightfm model')
            user_id = int(np.where(self.user_id_cats == user)[0])
            scores = self.model.predict(user_id, np.arange(self.n_items))
            top_items = list(self.survey_id_cats[np.argsort(-scores)])
            top_items = [item for item in top_items if item not in previous_seen_items]
            prediction = top_items[:k]
            logger.info(f'Prediction for user {user}: {prediction}')
            return prediction

        else:
            logger.debug('Use cold start model')
            top_items = self.most_popular_items
            top_items = [item for item in top_items if item not in previous_seen_items]
            prediction = top_items[:k]
            logger.info(f'Prediction for user {user}: {prediction}')
            return prediction

    @staticmethod
    def load_data(path_to_data='../data/recsys_data.csv'):
        data = pd.read_csv(path_to_data)
        data['date'] = pd.to_datetime(data['date'])
        return data

    @staticmethod
    def load_model(models_path, model_name: str = 'model.joblib'):
        model_path = os.path.abspath(os.path.join(models_path, model_name))
        model: LightFM = joblib.load(model_path)

        path_to_user_id_cats = os.path.abspath(os.path.join(models_path, 'user_id_cats.pickle'))
        path_to_survey_id_cats = os.path.abspath(os.path.join(models_path, 'survey_id_cats.pickle'))

        with open(path_to_user_id_cats, 'rb') as file:
            user_id_cats: np.ndarray = np.array(pickle.load(file).categories)

        with open(path_to_survey_id_cats, 'rb') as file:
            survey_id_cats: np.ndarray = np.array(pickle.load(file).categories)

        path_to_most_popular_items = os.path.abspath(os.path.join(models_path, 'most_popular_items.pickle'))
        with open(path_to_most_popular_items, 'rb') as file:
            most_popular_items: List[int] = pickle.load(file)

        return model, user_id_cats, survey_id_cats, most_popular_items


def main():
    rec_model = RecModel('../data/models')
    prediction = rec_model.predict(0)

    prediction = rec_model.predict(40881)


if __name__ == "__main__":
    main()
