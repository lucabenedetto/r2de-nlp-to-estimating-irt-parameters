import numpy as np

DISCRIMINATION_COEFFICIENT = 1.7


def item_response_function(difficulty, skill, discrimination, guess, slip) -> float:
    """
    Computes the logistic function for the given arguments and returns a float. The initial np.product is necessary for
    the multidimensional case.
    """
    return np.product(
        np.add(
            guess,
            np.divide(
                1.0 - np.add(guess, slip),
                1.0 + np.exp(-DISCRIMINATION_COEFFICIENT * np.multiply(discrimination, np.subtract(skill, difficulty)))
            )
        )
    )


def inverse_item_response_function(difficulty, skill, discrimination, guess, slip) -> float:
    """
    Computes 1 - logistic function for the given arguments and returns a float.
    """
    return 1.0 - item_response_function(difficulty, skill, discrimination, guess, slip)


def information_function(b, theta, discrimination, guess=0, slip=0) -> float:
    """
    Information function of a question: I(theta) = (P'(theta))**2/(P(theta)*Q(theta)), where Q(theta) = 1 - P(theta)
    """
    return np.divide(
        np.square(derivative_item_response_function(b, theta, discrimination, guess, slip)),
        (
                item_response_function(b, theta, discrimination, guess, slip)
                * inverse_item_response_function(b, theta, discrimination, guess, slip)
        )
    )


def derivative_item_response_function(b, theta, discrimination, guess, slip) -> float:
    """
    Computes the derivative of the item_response_function.
    """
    x = np.exp(-DISCRIMINATION_COEFFICIENT * discrimination * (theta[0]-b[0]))
    return np.divide((1.-guess-slip) * x * (-DISCRIMINATION_COEFFICIENT) * discrimination, np.square(1 + x**2))
