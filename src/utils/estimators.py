import pandas as pd
from pyirt import irt

from src.constants import (
    DIFFICULTY_MIN,
    DIFFICULTY_MAX,
    DEFAULT_DISCRIMINATION,
    DEFAULT_GUESS,
    DIFFICULTY_KEY,
    DISCRIMINATION_KEY,
    USER_ID_HEADER,
    CORRECT_HEADER,
    QUESTION_ID_HEADER,
)


def question_irt_estimation(
        interactions_df: pd.DataFrame,
        difficulty_range=(DIFFICULTY_MIN, DIFFICULTY_MAX),
        discrimination_range=(DEFAULT_DISCRIMINATION, DEFAULT_DISCRIMINATION),
        guess=DEFAULT_GUESS
) -> dict:
    """
    Calls the method for IRT estimation and returns only the dictionary containing the difficulty of the questions.
    :param interactions_df:
    :param difficulty_range:
    :param discrimination_range:
    :param guess:
    :return:
    """
    return irt_estimation(interactions_df, difficulty_range, discrimination_range, guess)[1]


def irt_estimation(
        interactions_df: pd.DataFrame,
        difficulty_range=(DIFFICULTY_MIN, DIFFICULTY_MAX),
        discrimination_range=(DEFAULT_DISCRIMINATION, DEFAULT_DISCRIMINATION),
        guess=DEFAULT_GUESS
) -> (dict, dict):
    """
    Given the input interactions between a set of students and a set of questions, performs with the irt method from
    pyirt the IRT estimation of the latent traits of students and questions. It returns the dictionaries mapping
    from the studentID or itemID to the corresponding latent traits.
    """
    interactions_list = [
        (user, item, correctness)
        for user, item, correctness in interactions_df[[USER_ID_HEADER, QUESTION_ID_HEADER, CORRECT_HEADER]].values
    ]
    # if there are some items with only correct or only wrong answers, pyirt crashes
    question_count_per_correctness = interactions_df.groupby([QUESTION_ID_HEADER, CORRECT_HEADER])\
        .size().reset_index().groupby(QUESTION_ID_HEADER).size().reset_index().rename(columns={0: 'cnt'})
    list_q_to_add = list(question_count_per_correctness[question_count_per_correctness['cnt'] == 1][QUESTION_ID_HEADER])
    print('[INFO] %d questions filled in' % len(list_q_to_add))
    interactions_list.extend([('p_good', itemID, True) for itemID in list_q_to_add])
    interactions_list.extend([('p_bad', itemID, False) for itemID in list_q_to_add])

    try:
        item_params, user_params = irt(
            interactions_list,
            theta_bnds=difficulty_range,
            beta_bnds=difficulty_range,
            alpha_bnds=discrimination_range,
            in_guess_param={q: guess for q in interactions_df[QUESTION_ID_HEADER].unique()},
            max_iter=100
        )
    except Exception:
        raise ValueError("Problem in irt_estimation. Check if there are items with only correct/wrong answers.")
    question_dict = dict()
    question_dict[DIFFICULTY_KEY] = dict()
    question_dict[DISCRIMINATION_KEY] = dict()
    for question, question_params in item_params.items():
        question_dict[DIFFICULTY_KEY][question] = -question_params['beta']
        question_dict[DISCRIMINATION_KEY][question] = question_params["alpha"]
    user_dict = {x[0]: x[1] for x in user_params.items()}
    return user_dict, question_dict
