from sklearn.feature_extraction.text import CountVectorizer

class TextVectorizer:
    def __init__(self, max_features=5000):
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english'
        )

    def fit_transform(self, texts):
        """
        Fits the vectorizer and transforms text into vectors.
        Args:
            texts (pd.Series or list): Text data
        Returns:
            np.ndarray: Vectorized representation
        """
        return self.vectorizer.fit_transform(texts).toarray()
