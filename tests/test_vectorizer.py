import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion.data_loader import load_data
from src.feature_engineering.feature_builder import FeatureBuilder
from src.ml.vectorizer import TextVectorizer

def test_vectorizer():
    movies, credits = load_data()
    if movies is None:
        print("Failed to load data")
        return
        
    df = movies.merge(credits, left_on='id', right_on='movie_id')

    builder = FeatureBuilder()
    processed_df = builder.build_features(df)

    vectorizer = TextVectorizer()
    print("Vectorizing data... (this might take a moment)")
    vectors = vectorizer.fit_transform(processed_df['soup'])

    print("SUCCESS: Vectorization completed")
    print("Vector shape:", vectors.shape)

if __name__ == "__main__":
    test_vectorizer()
