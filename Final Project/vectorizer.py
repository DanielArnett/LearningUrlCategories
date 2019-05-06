from sklearn.feature_extraction.text import TfidfVectorizer


def word_vectorizer(text):
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=10000)
    vectorizer.fit(text)
    return vectorizer


def char_vectorizer(text):
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        max_features=50000)
    vectorizer.fit(text)
    return vectorizer


def test_debug(x, y):
    return x + y