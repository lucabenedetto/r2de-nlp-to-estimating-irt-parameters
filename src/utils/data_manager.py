import pandas as pd
from src.constants import (
    CORRECT_HEADER,
    DESCRIPTION_HEADER,
    QUESTION_ID_HEADER,
    QUESTION_TEXT_HEADER,
)


def generate_correct_answers_dictionary(answers_text_df: pd.DataFrame) -> dict:
    """
    Given the dataframe containing the text of the answers (i.e. possible choices), it returns a dictionary containing
    for each question (the question IDs are the keys) a string made of the concatenation of all the correct answers to
    that question.
    """
    correct_answers_text_df = answers_text_df[answers_text_df[CORRECT_HEADER]]
    return generate_answers_dictionary(correct_answers_text_df)


def generate_wrong_answers_dictionary(answers_text_df: pd.DataFrame) -> dict:
    """
    Given the dataframe containing the text of the answers (i.e. possible choices), it returns a dictionary containing
    for each question (the question IDs are the keys) a string made of the concatenation of all the wrong answers to
    that question.
    """
    wrong_answers_text_df = answers_text_df[~answers_text_df[CORRECT_HEADER]]
    return generate_answers_dictionary(wrong_answers_text_df)


def generate_answers_dictionary(answers_text_df: pd.DataFrame) -> dict:
    """
    Given the dataframe containing the text of the answers (i.e. possible choices), it returns a dictionary containing
    for each question (the question IDs are the keys) a string made of the concatenation of all the answers to that q.
    """
    answers_text_dict = dict()
    for q_id, text in answers_text_df[[QUESTION_ID_HEADER, DESCRIPTION_HEADER]].values:
        if q_id not in answers_text_dict.keys():
            answers_text_dict[q_id] = str(text)
        else:
            answers_text_dict[q_id] = answers_text_dict[q_id] + ' ' + str(text)
    return answers_text_dict


def concatenate_answers_text_into_question_text_df(
        question_text_df: pd.DataFrame,
        answers_text_df: pd.DataFrame,
        correct: bool = True,
        wrong: bool = True
) -> list:
    """
    creates a unique text made by concatenating to the text of the question the text of the correct choices and/or the
    text of the wrong choices (which ones to concatenate depends on the aguments).
    """
    if correct:
        correct_answers_text_dict = generate_correct_answers_dictionary(answers_text_df)
        question_text_df[QUESTION_TEXT_HEADER] = question_text_df.apply(
            lambda r:
            r[QUESTION_TEXT_HEADER]
            + ' ' + correct_answers_text_dict[r[QUESTION_ID_HEADER]]
            if type(correct_answers_text_dict[r[QUESTION_ID_HEADER]]) == str
            else r[QUESTION_TEXT_HEADER],
            axis=1
        )
    if wrong:
        wrong_answers_text_dict = generate_wrong_answers_dictionary(answers_text_df)
        question_text_df[QUESTION_TEXT_HEADER] = question_text_df.apply(
            lambda r:
            r[QUESTION_TEXT_HEADER]
            + ' ' + wrong_answers_text_dict[r[QUESTION_ID_HEADER]]
            if type(wrong_answers_text_dict[r[QUESTION_ID_HEADER]]) == str
            else r[QUESTION_TEXT_HEADER],
            axis=1
        )
    return question_text_df[QUESTION_TEXT_HEADER]
