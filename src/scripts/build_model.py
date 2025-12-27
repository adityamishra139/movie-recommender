import os
import sys

# Ensure src modules are found
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data_ingestion.data_loader import load_data
from src.feature_engineering.feature_builder import FeatureBuilder
from src.ml.vectorizer import TextVectorizer
from src.ml.similarity import SimilarityEngine
from src.utils.model_persistence import ModelPersistence

def build():
    print("🔨 Starting model build process...")
    
    # 1. Load data
    movies, credits = load_data()
    if movies is None:
        print("❌ Error loading data")
        sys.exit(1)

    # 2. Merge
    print("Pre-processing data...")
    df = movies.merge(credits, left_on="id", right_on="movie_id")
    
    # Using full dataset (5000 movies) as per user request
    print(f"Dataset size: {len(df)} movies.")

    # 3. Feature Engineering
    print("Building features...")
    builder = FeatureBuilder()
    processed_df = builder.build_features(df)

    # 4. Vectorization
    print("Vectorizing...")
    vectorizer = TextVectorizer()
    vectors = vectorizer.fit_transform(processed_df['soup'])

    # 5. Similarity
    print("Computing similarity...")
    similarity_engine = SimilarityEngine()
    similarity = similarity_engine.compute_similarity(vectors)

    # 6. Save
    print("Saving artifacts...")
    persistence = ModelPersistence()
    persistence.save(processed_df, "movies.pkl")
    persistence.save(similarity, "similarity.pkl")
    # We might not strictly need the vectorizer for inference if we only lookup by ID
    # but saving it is good practice if we allow new user input text later.
    persistence.save(vectorizer, "vectorizer.pkl")

    print("✅ Build complete.")

if __name__ == "__main__":
    build()
