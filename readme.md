Create a recommendation system based on the data in recsys_data.csv file. The goal
is to recommend surveys with the highest completion probability to users (up to 3
surveys to each user).

The columns are:
● user - the id of the user
● survey - the id of the survey
● status - whether the completion of the survey was successful or not (1 -
successful, 0 - unsuccessful)
● date - the date of the survey completion attempt.

Note: Surveys have a limited lifetime but exact lifetime information is not available
(it can be 20 minutes, a week, a month, etc.). If you can create a recommendation
system that pays more attention to newer surveys, it would be better. You can
disregard this part if it makes the task harder for you.