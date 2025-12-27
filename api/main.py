from fastapi import FastAPI, Query
from src.utils.model_persistence import ModelPersistence
from src.ml.recommender import Recommender

app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommender using cosine similarity",
    version="1.0.0"
)

print("🔄 Loading model artifacts...")

# Load artifacts once at startup
try:
    persistence = ModelPersistence()
    processed_df = persistence.load("movies.pkl")
    similarity = persistence.load("similarity.pkl")
    recommender = Recommender(processed_df, similarity)
    print("✅ Model loaded and ready.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Make sure to run 'python app.py' with BUILD_MODEL=True at least once.")
    processed_df = None
    similarity = None
    recommender = None

@app.get("/")
def home():
    return {"message": "Movie Recommender API is running"}

@app.get("/recommend")
def recommend(
    movie: str = Query(..., description="Movie name to get recommendations for"),
    top_n: int = 5
):
    if recommender is None:
        return {"error": "Model not loaded. Please ensure artifacts are generated."}

    try:
        candidates = recommender.search_movies(movie)

        if not candidates:
            return {
                "error": f"Movie '{movie}' not found",
                "suggestions": []
            }

        # Exact match logic
        exact_match = next(
            (c for c in candidates if c.lower() == movie.lower()),
            None
        )

        target_movie = None
        if exact_match:
            target_movie = exact_match
        elif len(candidates) == 1:
            target_movie = candidates[0]
        else:
            return {
                "error": "Multiple matches found",
                "suggestions": candidates[:10]
            }

        recommendations = recommender.recommend(target_movie, top_n=top_n)

        return {
            "input_movie": target_movie,
            "recommendations": recommendations
        }

    except Exception as e:
        return {"error": str(e)}
