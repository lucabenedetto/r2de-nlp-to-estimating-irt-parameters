import pandas as pd
from r2de.constants import FEATURES_HEADER, QUESTION_TEXT_HEADER


def concatenate_answers_text_into_question_text_df(
        question_text_df: pd.DataFrame,
        correct: bool = True,
        wrong: bool = True,
) -> list:
    if correct:
        question_text_df[QUESTION_TEXT_HEADER] = question_text_df.apply(
            lambda r: r[QUESTION_TEXT_HEADER] + ' '.join(r['text_correct_choices']), axis=1)
    if wrong:
        question_text_df[QUESTION_TEXT_HEADER] = question_text_df.apply(
            lambda r: r[QUESTION_TEXT_HEADER] + ' '.join(r['text_wrong_choices']), axis=1)
    return question_text_df[QUESTION_TEXT_HEADER]


def get_encoded_texts(mode, df_train, df_test):
    if mode == 0:
        print("DOING ENCODING 1")
        # output_file.write('ENCODING 1: "question only"\n')
        df_train[FEATURES_HEADER] = df_train.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_train = list(df_train[FEATURES_HEADER].values)
        df_test[FEATURES_HEADER] = df_test.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_test = list(df_test[FEATURES_HEADER].values)

    elif mode == 1:
        print("DOING ENCODING 2")
        # output_file.write('\nENCODING 2: "question correct"\n')
        df_train[QUESTION_TEXT_HEADER] = concatenate_answers_text_into_question_text_df(
            df_train, correct=True, wrong=False)
        df_train[FEATURES_HEADER] = df_train.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_train = list(df_train[FEATURES_HEADER].values)
        df_test[QUESTION_TEXT_HEADER] = concatenate_answers_text_into_question_text_df(
            df_test, correct=True, wrong=False)
        df_test[FEATURES_HEADER] = df_test.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_test = list(df_test[FEATURES_HEADER].values)

    else:  # encoding_idx == 2
        print("DOING ENCODING 3")
        # output_file.write('\nENCODING 3: "question full"\n')
        df_train[QUESTION_TEXT_HEADER] = concatenate_answers_text_into_question_text_df(
            df_train, correct=True, wrong=True)
        df_train[FEATURES_HEADER] = df_train.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_train = list(df_train[FEATURES_HEADER].values)
        df_test[QUESTION_TEXT_HEADER] = concatenate_answers_text_into_question_text_df(
            df_test, correct=True, wrong=True)
        df_test[FEATURES_HEADER] = df_test.apply(lambda r: r[QUESTION_TEXT_HEADER].lower(), axis=1)
        x_test = list(df_test[FEATURES_HEADER].values)

    return x_train, x_test
