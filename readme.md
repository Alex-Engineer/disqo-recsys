Create a recommendation system based on the data in recsys_data.csv file. The goal
is to recommend surveys with the highest completion probability to users (up to 3
surveys to each user).

The columns are:
* user - the id of the user
* survey - the id of the survey
* status - whether the completion of the survey was successful or not (1 -
successful, 0 - unsuccessful)
* date - the date of the survey completion attempt.

Note: Surveys have a limited lifetime but exact lifetime information is not available
(it can be 20 minutes, a week, a month, etc.). If you can create a recommendation
system that pays more attention to newer surveys, it would be better. You can
disregard this part if it makes the task harder for you.
---
# Installation
create directory ./data and put recsys_data.csv in ./data

create Python Virtual Environment
```
python3 -m venv env
pip install -r requirements.txt
```
---
# Jupyter notebooks with experiments
### Exploratory data analysis summary
src/notebooks/EDA notebook.ipynb
### Very simple baseline
src/notebooks/very simple baseline.ipynb
### lightfm baseline
src/notebooks/lightfm baseline.ipynb
### lightfm with clean data
src/notebooks/lightfm with clean data.ipynb

# Experiments results
Metrics was calculated on test part of data (1 day). I use very simple train/test split by date.
For cold start users I used most popular items.

| Experiment      | map@k  | precision@k  | recall@k  |
| ----------- | ----------- |----------- |----------- |
| baseline      | 0.00019       | 0.00024       | 0.00040       |
| all data   | 0.00234         | 0.00255       | 0.00331       |
| clean data   | **0.00288**        | **0.00276**       | **0.00382**       |

The best results show Lightfm model fitted on clean data without anomalies.

---
# Code for fit and inference
```
python model_fit.py
python inference.py
```
---
# conclusion

I research dataset, find anomalies in user and items. I conducted simple experiments and made prototype.

Things that I can do if have more time:
* Better fit models
* Make web app with FastApi
* Add sql database for web app
* Make docker


Unfortunately I did not have time to study the question of how adding unsuccessful surveys to data will change the result
I didn't have time to make a recommender system that would pay more attention to new surveys, but I have ideas.
Each surveys can be assigned an expiration date and will simply be excluded from the recommendation results.
For each item, you can assign a weight from 0 to 1 (where 1 is the newest item, 0 is the oldest item) and
multiply the received from model scores for items by the given vector.

