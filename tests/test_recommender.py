import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion.data_loader import load_data
from src.feature_engineering.feature_builder import FeatureBuilder
from src.ml.vectorizer import TextVectorizer
from src.ml.similarity import SimilarityEngine
from src.ml.recommender import Recommender

def test_recommender():
    print("Loading data...")
    movies, credits = load_data()
    if movies is None:
        print("Dataset load failed")
        return

    df = movies.merge(credits, left_on='id', right_on='movie_id')

    print("Building features...")
    builder = FeatureBuilder()
    processed_df = builder.build_features(df)

    print("Vectorizing...")
    vectorizer = TextVectorizer()
    vectors = vectorizer.fit_transform(processed_df['soup'])

    print("Computing similarity...")
    similarity_engine = SimilarityEngine()
    similarity = similarity_engine.compute_similarity(vectors)

    print("Initializing Recommender...")
    recommender = Recommender(processed_df, similarity)

    test_movie = "Batman Begins"
    print(f"\nRecommendations for '{test_movie}':")
    try:
        recs = recommender.recommend(test_movie)
        for movie in recs:
            print("-", movie)
        
        # Simple assertion
        if len(recs) == 5:
            print("\nSUCCESS: Received 5 recommendations.")
        else:
            print(f"\nFAILURE: Expected 5 recommendations, got {len(recs)}")

    except ValueError as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test_recommender()
