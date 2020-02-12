"""
This script contains the code to recreate the plots representing the effects of Nw on the latent traits estimation.
As described in the paper, Nw is the size of the feature arrays that are generated from the text and given as input to
the regression modules. Those features are the most frequent TF-IDF features in the corpus.
It has to be run *after* cv-for-model-choice, as it uses some data generated in that script.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from r2de.utils.text_processing import text_preprocessor
from r2de.utils.data_manager import concatenate_answers_text_into_question_text_df
from r2de.constants import (
    QUESTION_ID_HEADER,
    QUESTION_TEXT_HEADER,
    FEATURES_HEADER,
    DATA_PATH,
    ANSWERS_TEXT_FILENAME,
    DS_GTE_FILENAME,
    TARGET_DIFFICULTY_HEADER,
    TARGET_DISCRIMINATION_HEADER,
)
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import mean_squared_error

plt.rcParams.update({'text.usetex': True, 'font.size': 14})

# the models as obtained in the -for-model-choice script
model_for_difficulty = RandomForestRegressor(n_estimators=250, max_depth=50)
model_for_discrimination = RandomForestRegressor(n_estimators=200, max_depth=50)

# the values of Nw to test
list_nw = np.arange(100, 2001, 100)

# load the same data split as in the CV script
df_GTE = pd.read_csv(DATA_PATH + DS_GTE_FILENAME)
df_train_nlp = df_GTE[[QUESTION_ID_HEADER, QUESTION_TEXT_HEADER]].drop_duplicates(QUESTION_ID_HEADER).copy()

# concatenate choices' text to question text: in this script only the "question full" encoding is used
answers_text_df = pd.read_csv(DATA_PATH + ANSWERS_TEXT_FILENAME)
df_train_nlp[QUESTION_TEXT_HEADER] = concatenate_answers_text_into_question_text_df(
    df_train_nlp, answers_text_df, correct=True, wrong=True
)

# collect the data as stored in lb-cv-for-model-choice (question split and estimated latent traits)
train_questions = pickle.load(open(DATA_PATH + 'train-questions.p', "rb"))
test_questions = pickle.load(open(DATA_PATH + 'test-questions.p', "rb"))
difficulty_dict = pickle.load(open(DATA_PATH + 'true-difficulty-dict.p', "rb"))
discrimination_dict = pickle.load(open(DATA_PATH + 'true-discrimination-dict.p', "rb"))

df_train_nlp[FEATURES_HEADER] = df_train_nlp.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
df_train_nlp[TARGET_DIFFICULTY_HEADER] = df_train_nlp.apply(
    lambda r: difficulty_dict[r[QUESTION_ID_HEADER]], axis=1)
df_train_nlp[TARGET_DISCRIMINATION_HEADER] = df_train_nlp.apply(
    lambda r: discrimination_dict[r[QUESTION_ID_HEADER]], axis=1)

# second data split
df_train = df_train_nlp[df_train_nlp[QUESTION_ID_HEADER].isin(train_questions)]
df_test = df_train_nlp[df_train_nlp[QUESTION_ID_HEADER].isin(test_questions)]

# collect features
x_train = list(df_train[FEATURES_HEADER].values)
x_test = list(df_test[FEATURES_HEADER].values)

# collect target values
y_train_difficulty = list(df_train[TARGET_DIFFICULTY_HEADER].values)
y_test_difficulty = list(df_test[TARGET_DIFFICULTY_HEADER].values)
y_train_discrimination = list(df_train[TARGET_DISCRIMINATION_HEADER].values)
y_test_discrimination = list(df_test[TARGET_DISCRIMINATION_HEADER].values)

difficulty_mse_list = []
discrimination_mse_list = []
for nw in list_nw:
    vectorizer = TfidfVectorizer(
        max_features=nw,
        ngram_range=(1, 1),
        stop_words='english',
        preprocessor=text_preprocessor,
        analyzer='word'
    )

    # difficulty
    pipe = Pipeline(steps=[('tfidf', vectorizer), ('regressor', model_for_difficulty)], verbose=False)
    y_pred_difficulty = pipe.predict(x_test)
    difficulty_mse_list.append(mean_squared_error(y_test_difficulty, y_pred_difficulty))

    # discrimination
    pipe = Pipeline(steps=[('tfidf', vectorizer), ('regressor', model_for_discrimination)], verbose=False)
    y_pred_discrimination = pipe.predict(x_test)
    discrimination_mse_list.append(mean_squared_error(y_test_discrimination, y_pred_discrimination))

X = np.arange(len(list_nw))

# plot the effects of Nw on difficulty estimation
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(X, difficulty_mse_list, linewidth=3, label='RF', c='#16607A')
ax.set_ylabel('MSE', fontsize=16)
ax.set_xlabel(r'$N_W$', fontsize=16)
ax.set_xticks(X)
ax.set_xticklabels([x if idx % 2 == 0 else '' for idx, x in enumerate(list_nw)])
plt.legend()
plt.show()

# plot the effects of Nw on discrimination estimation
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(X, discrimination_mse_list, linewidth=3, label='RF', c='#16607A')
ax.set_ylabel(r'MSE', fontsize=16)
ax.set_xlabel(r'$N_W$')
ax.set_xticks(X)
ax.set_xticklabels([x if idx % 2 == 0 else '' for idx, x in enumerate(list_nw)])
plt.legend()
plt.show()
