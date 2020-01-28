import pandas as pd
from sklearn import model_selection


def train_test_split_by_column(
        data: pd.DataFrame,
        column: str,
        ratio: float = 0.7,
        random_state: int = None,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the input dataframe in train set and test set. The split is stratified on the column passed as parameter
    (only one column name is accepted) and the ratio can be given as parameter.
    :param data: input dataframe, to split
    :param column: the column to stratify the split
    :param ratio: fraction of data to be used as train set
    :param random_state:
    :return: the train df and the test df
    """
    if column not in data.columns:
        raise ValueError("The column parameter must be one of the columns of the 'data' dataframe")
    return model_selection.train_test_split(data, test_size=1.0-ratio, stratify=data[column], random_state=random_state)
