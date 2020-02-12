# R2DE: a NLP approach to estimating IRT parameters of newly generated questions

This folder contains the code used to obtain the results of the paper "R2DE: a NLP approach to estimating IRT parameters of newly generated questions" (https://arxiv.org/abs/2001.07569), presented at the LAK20 conference.

The scripts that were used to obtain the results are:

- _lb-cv-for-model-choice.py_: cross validation for model choice and tests on latent traits estimation
- _lb-analysis-effects-nw.py_: for obtaining the plots showing the effects of the Nw threshold
- _lb-performance-prediction.py_: for the performance prediction part

All the scripts require the data to be stored in a folder whose path is saved in the DATA_PATH constant (in our case it was './data/', thus that is the value of the variable in constants.py)

This code uses data stored in three .csv files, as shown in the constants.py file:

- _answers\_texts.csv_
- _detailed\_quiz\_session_answer.csv_
- _questions\_counts.csv_

Below, a description of the type of information contained in each file and their structure. It should be sufficient to format your data in the same way as presented here to use these scripts without any changes.

- _detailed\_quiz\_session_answer.csv_ contains the list of interactions between students and items (i.e. the answers given by the students to the questions they encountered). The following columns are used:
    - `user\_id`: the ID of the student (str)
    - `time_stamp`: the time of the interaction (datetime)
    - `correct`: whether the student answered correctly (boolean)
    - `question_id`: ID of the question (str)
    - `question_text`: the text of the question (str)
- _answers\_texts.csv_ contains for each question the text of the possible choices
    - `question_id`: the ID of the question (str)
    - `id`: the ID of the choice (str)
    - `correct`: whether it is the correct choice (boolean)
    - `description`: the text of the choice (str)
- _questions\_counts.csv_ is a file that contains the count of occurrences for each question, it could also easily be obtained from detailed_quiz_session_answer.csv
    - `question\_id`: (str)
    - `count`: (int)

If you have conda, in order to run this code it should be sufficient to follow these steps (provided that you have data in the required format):
```
conda create -n venv-r2de-public-code python=3 pip
conda activate venv-r2de-public-code
python setup.py install
python lb-cv-for-model-choice.py
python lb-analysis-effects-nw.py
python lb-performance-prediction.py
```
