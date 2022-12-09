from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from r2de.utils.text_processing import text_preprocessor


def get_model():
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1), stop_words='english', preprocessor=text_preprocessor, analyzer='word')
    regressor = RandomForestRegressor()
    return Pipeline(steps=[('tfidf', vectorizer), ('regressor', regressor)], verbose=False)
