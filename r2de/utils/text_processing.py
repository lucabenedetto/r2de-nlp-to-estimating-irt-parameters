import re
import string
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def text_preprocessor(
        text,
        uncased=True,
        remove_stop_words=True,
        remove_html_tags=False,
        remove_numbers=True,
        remove_punctuation=True,
        perform_stemming=True,
):
    if uncased:
        text = text.lower()
    if remove_stop_words:
        text = stop_words_removal(text)
    if remove_html_tags:
        text = html_tags_removal(text)
    if remove_numbers:
        text = numbers_removal(text)
    if remove_punctuation:
        text = punctuation_removal(text)
    if perform_stemming:
        text = stemming(text)
    return text


def html_tags_removal(text):
    return re.sub("<.*?>", " ", text)


def numbers_removal(text):
    return ' '.join([x for x in text.split(' ') if not x.isdigit()])


def punctuation_removal(text):
    return ''.join([char if char not in string.punctuation else ' ' for char in text])


def stop_words_removal(text, stop_words=ENGLISH_STOP_WORDS):
    return ' '.join([word for word in text.split() if word not in stop_words])


def stemming(text, stemmer=PorterStemmer()):
    return ' '.join([stemmer.stem(word) for word in text.split()])

