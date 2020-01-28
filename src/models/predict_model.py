import numpy as np
import pandas as pd
from src.utils.math import (
    inverse_item_response_function,
    item_response_function,
    information_function,
)
from src.constants import (
    USER_ID_HEADER,
    CORRECT_HEADER,
    QUESTION_ID_HEADER,
    DEFAULT_GUESS,
    DEFAULT_DISCRIMINATION,
    DEFAULT_SLIP,
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
)


def perform_user_irt_prediction(
        interactions_df: pd.DataFrame,
        difficulty_dict: dict,
        discrimination_dict: dict,
        difficulty_range: (DIFFICULTY_MIN, DIFFICULTY_MAX),
        theta_increment=0.1,
        initial_theta=(DIFFICULTY_MAX+DIFFICULTY_MIN)/2,
        guess=DEFAULT_GUESS,
        slip=DEFAULT_SLIP,
) -> list:
    """
    :param interactions_df: dataframe containing all the interactions between users and items
    :param difficulty_dict: dictionary containing the difficulty of each item
    :param discrimination_dict: dictionary containing the discrimination of each item
    :param difficulty_range: tuple containing min and max difficulty
    :param theta_increment: the granularity of the skill level we are interested in
    :param initial_theta: starting skill level for the estimation
    :param guess: guess factor to use in the IRT model
    :param slip: slip factor to use in the IRT model
    :return: the list containing the predicted results for the interactions in the input dataframe
    """
    predicted_result = []
    estimated_theta = [initial_theta]
    thetas = np.arange(difficulty_range[0], difficulty_range[1] + theta_increment, theta_increment)
    log_likelihood = np.zeros(len(thetas), dtype=np.float)
    information_func = np.zeros(len(thetas), dtype=np.float)
    list_loglikelihood = np.zeros(len(thetas), dtype=object)
    list_information_function = np.zeros(len(thetas), dtype=object)
    for idx, theta in enumerate(thetas):
        list_loglikelihood[idx] = []
        list_information_function[idx] = []

    for true_result, item_id in interactions_df[[CORRECT_HEADER, QUESTION_ID_HEADER]].values:
        if item_id in difficulty_dict.keys() and item_id in discrimination_dict.keys():
            difficulty = [difficulty_dict[item_id]]
            discrimination = discrimination_dict[item_id]
        else:
            difficulty = [(DIFFICULTY_MAX+DIFFICULTY_MIN)/2]
            discrimination = DEFAULT_DISCRIMINATION
            print("[INFO] Question with ID %s was not known. Manually set latent traits" % item_id)

        predicted_result.append(item_response_function(difficulty, estimated_theta, discrimination, guess, slip))

        func = item_response_function if true_result == 1 else inverse_item_response_function
        for idx, theta in enumerate(thetas):
            item_log_likelihood = np.log(func(difficulty, [theta], discrimination, guess, slip))
            list_loglikelihood[idx].append(item_log_likelihood)
            log_likelihood[idx] = np.sum(list_loglikelihood[idx])

            item_information = information_function(difficulty, [theta], discrimination, guess, slip)
            list_information_function[idx].append(item_information)
            information_func[idx] = np.sum(list_information_function[idx])

        estimated_theta = [thetas[np.argmax(log_likelihood)]]

    return predicted_result


def irt_prediction_with_update(
        interactions_df: pd.DataFrame,
        difficulty_dict: dict,
        discrimination_dict: dict,
        user_id_list: list,
        difficulty_range=(DIFFICULTY_MIN, DIFFICULTY_MAX),
        theta_increment=0.1,
        initial_theta=(DIFFICULTY_MAX + DIFFICULTY_MIN) / 2,
        guess=DEFAULT_GUESS,
        slip=DEFAULT_SLIP,
) -> list:
    """
    :param interactions_df: dataframe containing all the interactions between users and items
    :param difficulty_dict: dictionary containing the difficulty of each item
    :param discrimination_dict: dictionary containing the discrimination of each item
    :param user_id_list:
    :param difficulty_range: tuple containing min and max difficulty
    :param theta_increment: the granularity of the skill level we are interested in
    :param initial_theta: starting skill level for the estimation
    :param guess: guess factor to use in the IRT model
    :param slip: slip factor to use in the IRT model
    :return: the list containing the predicted results for all the interactions and students in the input dataframe
    """
    predicted_result = []
    for user_id in user_id_list:  # performance prediction is done for all the students, one at a time
        predicted_result.extend(
            perform_user_irt_prediction(
                interactions_df=interactions_df[interactions_df[USER_ID_HEADER] == user_id],
                difficulty_dict=difficulty_dict,
                discrimination_dict=discrimination_dict,
                difficulty_range=difficulty_range,
                theta_increment=theta_increment,
                initial_theta=initial_theta,
                guess=guess,
                slip=slip,
            )
        )
    return predicted_result
