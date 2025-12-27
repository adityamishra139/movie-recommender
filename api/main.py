from fastapi import FastAPI, Query
import os

from src.data_ingestion.data_loader import load_data
from src.feature_engineering.feature_builder import FeatureBuilder
from src.ml.vectorizer import TextVectorizer
from src.ml.similarity import SimilarityEngine
from src.utils.model_persistence import ModelPersistence
from src.ml.recommender import Recommender


app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommender using cosine similarity",
    version="1.0.0"
)

print("🔄 Initializing Movie Recommender backend...")

persistence = ModelPersistence()
ARTIFACT_DIR = "artifacts"
SIM_PATH = os.path.join(ARTIFACT_DIR, "similarity.pkl")
MOVIES_PATH = os.path.join(ARTIFACT_DIR, "movies.pkl")

# --------------------------------------------------
# BUILD OR LOAD MODEL (RUNS ONCE AT SERVER START)
# --------------------------------------------------
if os.path.exists(SIM_PATH) and os.path.exists(MOVIES_PATH):
    print("⚡ Loading existing artifacts...")
    processed_df = persistence.load("movies.pkl")
    similarity = persistence.load("similarity.pkl")
else:
    print("🔨 Artifacts not found. Building model on server...")

    # 1. Load raw data
    movies, credits = load_data()
    if movies is None or credits is None:
        raise RuntimeError("Failed to load movie data")

    # 2. Merge
    df = movies.merge(credits, left_on="id", right_on="movie_id")

    # 3. Feature engineering
    builder = FeatureBuilder()
    processed_df = builder.build_features(df)

    # 4. Vectorization
    vectorizer = TextVectorizer()
    vectors = vectorizer.fit_transform(processed_df["soup"])

    # 5. Similarity
    similarity_engine = SimilarityEngine()
    similarity = similarity_engine.compute_similarity(vectors)

    # 6. Save artifacts (for future restarts)
    persistence.save(processed_df, "movies.pkl")
    persistence.save(similarity, "similarity.pkl")

print("✅ Model ready. Initializing recommender...")
recommender = Recommender(processed_df, similarity)


# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------
@app.get("/")
def home():
    return {"message": "Movie Recommender API is running"}


@app.get("/recommend")
def recommend(
    movie: str = Query(..., description="Movie name"),
    top_n: int = Query(5, description="Number of recommendations")
):
    candidates = recommender.search_movies(movie)

    if not candidates:
        return {
            "error": f"Movie '{movie}' not found",
            "suggestions": []
        }

    exact_match = next(
        (c for c in candidates if c.lower() == movie.lower()),
        None
    )

    target_movie = exact_match if exact_match else candidates[0]
    recommendations = recommender.recommend(target_movie, top_n)

    return {
        "input_movie": target_movie,
        "recommendations": recommendations
    }
