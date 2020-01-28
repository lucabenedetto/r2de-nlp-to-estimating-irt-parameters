"""
This script contain the code for performing the first data split, the IRT estimation of the difficulty and
discrimination of each question (which are used as ground truth in the paper), the second data split (in train questions
and test questions) and the cross validation for choosing the model and the encoding.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from src.models.data_preparation import train_test_split_by_column
from src.utils.estimators import question_irt_estimation
from src.utils.text_processing import text_preprocessor
from src.utils.evaluation_metrics import evaluation_metrics_to_string
from src.constants import (
    DATA_PATH,
    ANSWERS_TEXT_FILENAME,
    DETAILED_QS_ANSWERS_FILENAME,
    DS_GTE_FILENAME,
    DS_VAL_FILENAME,
    QUESTION_COUNT_FILENAME,
    COUNT_HEADER,
    DIFFICULTY_KEY,
    DISCRIMINATION_KEY,
    DISCRIMINATION_MIN,
    DISCRIMINATION_MAX,
    FEATURES_HEADER,
    QUESTION_ID_HEADER,
    QUESTION_TEXT_HEADER,
    TARGET_DISCRIMINATION_HEADER,
    TARGET_DIFFICULTY_HEADER,
)
from src.utils.data_manager import concatenate_answers_text_into_question_text_df
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pickle

# variables declaration
output_filename = 'cv-for-model-choice.txt'
irt_train_nlp_test_ratio = 0.7
question_train_test_split = 0.8
question_cnt_threshold = 100

vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),
    stop_words='english',
    preprocessor=text_preprocessor,
    analyzer='word'
)
vectorizer_params = {
    'tfidf__max_features': list(np.arange(100, 2001, 100)),
}
models_to_evaluate = [RandomForestRegressor(), DecisionTreeRegressor(), LinearRegression(), SVR()]
model_names = ['RF Regressor', 'DT Regressor', 'Linear Regressor', 'SVR']
parameters_for_model_evaluation = [
    {
        'regressor__n_estimators': [10, 25, 50, 100, 150, 200, 250],
        'regressor__max_depth': [2, 5, 10, 15, 25, 50]
    },  # RF
    {
        'regressor__max_features': [1, 2, 3, 4, 5, None],
        'regressor__max_depth': [2, 5, 10, 20, 50]
    },  # DT
    {
        'regressor__normalize': [True, False]
    },  # LR
    {
        'regressor__kernel': ['linear', 'poly', 'rbf'],
        'regressor__gamma': ['auto', 'scale'],
        'regressor__shrinking': [True, False],
        'regressor__degree': [1, 2, 3, 4]
    },  # SVR
]

output_file = open(output_filename, 'w')

# dataset collection + filtering questions below occurrence threshold
df = pd.read_csv(DATA_PATH + DETAILED_QS_ANSWERS_FILENAME)
question_cnt_df = pd.read_csv(DATA_PATH + QUESTION_COUNT_FILENAME)
list_question_ids = question_cnt_df[question_cnt_df[COUNT_HEADER] >= question_cnt_threshold][QUESTION_ID_HEADER].values
df = df[df[QUESTION_ID_HEADER].isin(list_question_ids)]

# first data split
df_GTE, df_VAL = train_test_split_by_column(df, QUESTION_ID_HEADER, ratio=irt_train_nlp_test_ratio)
df_GTE.to_csv(DATA_PATH + DS_GTE_FILENAME, index=False)
df_VAL.to_csv(DATA_PATH + DS_VAL_FILENAME, index=False)

# irt estimation + storing variables
question_dict = question_irt_estimation(df_GTE, discrimination_range=[DISCRIMINATION_MIN, DISCRIMINATION_MAX])
difficulty_dict = question_dict[DIFFICULTY_KEY]
discrimination_dict = question_dict[DISCRIMINATION_KEY]
pickle.dump(difficulty_dict, open(DATA_PATH + 'true-difficulty-dict.p', 'wb'))
pickle.dump(discrimination_dict, open(DATA_PATH + 'true-discrimination-dict.p', 'wb'))

# collection of the text of the correct answers - used for the second and the third encoding
answers_text_df = pd.read_csv(DATA_PATH + ANSWERS_TEXT_FILENAME)

# collection of dataset for training the NLP model, which is made of qID - text pairs
df_train_nlp = df_GTE[[QUESTION_ID_HEADER, QUESTION_TEXT_HEADER]].drop_duplicates(QUESTION_ID_HEADER).copy()
train_questions, test_questions = train_test_split(list_question_ids, test_size=(1-question_train_test_split))
pickle.dump(train_questions, open(DATA_PATH + 'train-questions.p', 'wb'))
pickle.dump(test_questions, open(DATA_PATH + 'test-questions.p', 'wb'))

# second data split
df_train_nlp[TARGET_DIFFICULTY_HEADER] = df_train_nlp.apply(lambda r: difficulty_dict[r[QUESTION_ID_HEADER]], axis=1)
df_train_nlp[TARGET_DISCRIMINATION_HEADER] = df_train_nlp.apply(
    lambda r: discrimination_dict[r[QUESTION_ID_HEADER]], axis=1)
df_train = df_train_nlp[df_train_nlp[QUESTION_ID_HEADER].isin(train_questions)].copy()
df_test = df_train_nlp[df_train_nlp[QUESTION_ID_HEADER].isin(test_questions)].copy()

# collect target values
y_train_difficulty = list(df_train[TARGET_DIFFICULTY_HEADER].values)
y_test_difficulty = list(df_test[TARGET_DIFFICULTY_HEADER].values)
y_train_discrimination = list(df_train[TARGET_DISCRIMINATION_HEADER].values)
y_test_discrimination = list(df_test[TARGET_DISCRIMINATION_HEADER].values)

# I will need the original ones to compute the 2nd and the 3rd encoding
original_df_train = df_train.copy()
original_df_test = df_test.copy()
for encoding_idx in range(3):
    # data preparation is different depending on the encoding
    if encoding_idx == 0:
        output_file.write('ENCODING 1: "question only"\n')
        df_train[FEATURES_HEADER] = df_train.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_train = list(df_train[FEATURES_HEADER].values)
        df_test[FEATURES_HEADER] = df_test.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_test = list(df_test[FEATURES_HEADER].values)
    elif encoding_idx == 1:
        output_file.write('\nENCODING 2: "question correct"\n')
        df_train[QUESTION_TEXT_HEADER] = concatenate_answers_text_into_question_text_df(
            original_df_train, answers_text_df, correct=True, wrong=False)
        df_train[FEATURES_HEADER] = df_train.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_train = list(df_train[FEATURES_HEADER].values)
        df_test[QUESTION_TEXT_HEADER] = concatenate_answers_text_into_question_text_df(
            original_df_test, answers_text_df, correct=True, wrong=False)
        df_test[FEATURES_HEADER] = df_test.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_test = list(df_test[FEATURES_HEADER].values)
    else:  # encoding_idx == 2
        output_file.write('\nENCODING 3: "question full"\n')
        df_train[QUESTION_TEXT_HEADER] = concatenate_answers_text_into_question_text_df(
            original_df_train, answers_text_df, correct=True, wrong=True)
        df_train[FEATURES_HEADER] = df_train.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_train = list(df_train[FEATURES_HEADER].values)
        df_test[QUESTION_TEXT_HEADER] = concatenate_answers_text_into_question_text_df(
            original_df_test, answers_text_df, correct=True, wrong=True)
        df_test[FEATURES_HEADER] = df_test.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_test = list(df_test[FEATURES_HEADER].values)

    for idx, regressor in enumerate(models_to_evaluate):
        output_file.write(model_names[idx] + "\n")
        pipe = Pipeline(steps=[('tfidf', vectorizer), ('regressor', regressor)], verbose=False)
        param_grid = vectorizer_params.copy()
        param_grid.update(parameters_for_model_evaluation[idx])
        search = GridSearchCV(pipe, param_grid, iid=False, cv=5, n_jobs=10)

        # Difficulty
        search.fit(x_train, y_train_difficulty)
        output_file.write("DIFFICULTY - best parameter (CV score=%0.3f):\n" % search.best_score_)
        output_file.write(str(search.best_params_) + '\n')
        best_estimator = search.best_estimator_
        output_file.write("training error (used for model's and encoding's choice):\n")
        y_pred_train_difficulty = best_estimator.predict(x_train)
        output_file.write(evaluation_metrics_to_string(y_train_difficulty, y_pred_train_difficulty) + '\n')
        output_file.write("test error:\n")
        y_pred_difficulty = best_estimator.predict(x_test)
        output_file.write(evaluation_metrics_to_string(y_test_difficulty, y_pred_difficulty) + '\n')
        # store the predicted difficulties in a dictionary (used for the performance prediction estimation)
        predicted_difficulty_dict = dict()
        for q_idx, q_id in enumerate(test_questions):
            predicted_difficulty_dict[q_id] = y_pred_difficulty[q_idx]
        pickle.dump(predicted_difficulty_dict, open(DATA_PATH+'predicted-difficulty-enc%d-dict.p' % encoding_idx, 'wb'))

        # Discrimination
        search.fit(x_train, y_train_discrimination)
        output_file.write("DISCRIMINATION - best parameter (CV score=%0.3f):\n" % search.best_score_)
        output_file.write(str(search.best_params_) + '\n')
        best_estimator = search.best_estimator_
        output_file.write("training error (used for model's and encoding's choice):\n")
        y_pred_train_discrimination = best_estimator.predict(x_train)
        output_file.write(
            evaluation_metrics_to_string(y_train_discrimination, y_pred_train_discrimination) + '\n')
        output_file.write("test error:\n")
        y_pred_discrimination = best_estimator.predict(x_test)
        output_file.write(evaluation_metrics_to_string(y_test_discrimination, y_pred_discrimination) + '\n')
        # store the predicted discrimination in a dictionary (used for the performance prediction estimation)
        predicted_discrimination_dict = dict()
        for q_idx, q_id in enumerate(test_questions):
            predicted_discrimination_dict[q_id] = y_pred_discrimination[q_idx]
        pickle.dump(
            predicted_discrimination_dict, open(DATA_PATH+'predicted-discrimination-enc%d-dict.p' % encoding_idx, 'wb'))

output_file.close()
