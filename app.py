from src.data_ingestion.data_loader import load_data
from src.feature_engineering.feature_builder import FeatureBuilder
from src.ml.vectorizer import TextVectorizer
from src.ml.similarity import SimilarityEngine
from src.utils.model_persistence import ModelPersistence
from src.ml.recommender import Recommender

# 🔴 CHANGE THIS FLAG ONLY ONCE
BUILD_MODEL = False   # True only the first time


def main():

    persistence = ModelPersistence()

    if BUILD_MODEL:
        print("🔨 Building model from scratch...", flush=True)

        # 1. Load data
        movies, credits = load_data()
        if movies is None:
            print("❌ Error: Could not load data.")
            return

        # 2. Merge
        df = movies.merge(credits, left_on="id", right_on="movie_id")

        # 3. Feature Engineering
        print("🧠 Building features...", flush=True)
        builder = FeatureBuilder()
        processed_df = builder.build_features(df)

        # 4. Vectorization
        print("📊 Vectorizing...", flush=True)
        vectorizer = TextVectorizer()
        vectors = vectorizer.fit_transform(processed_df['soup'])

        # 5. Similarity
        print("📐 Computing similarity...", flush=True)
        similarity_engine = SimilarityEngine()
        similarity = similarity_engine.compute_similarity(vectors)

        # 6. Save artifacts (ONLY ONCE)
        persistence.save(processed_df, "movies.pkl")
        persistence.save(similarity, "similarity.pkl")
        persistence.save(vectorizer, "vectorizer.pkl")

        print("✅ Model built and saved successfully.")

    else:
        print("⚡ Loading model artifacts...", flush=True)

        processed_df = persistence.load("movies.pkl")
        similarity = persistence.load("similarity.pkl")

        print("✅ Model loaded successfully.")

    # 7. Initialize recommender
    print("🚀 Initializing recommender...", flush=True)
    recommender = Recommender(processed_df, similarity)

    print("\n🎬 Movie Recommender Ready!")

    # 8. User loop
    while True:
        try:
            movie_input = input("\nEnter a movie name (or 'exit'): ").strip()

            if not movie_input:
                continue

            if movie_input.lower() == "exit":
                print("👋 Exiting application.")
                break

            # Search candidates
            candidates = recommender.search_movies(movie_input)

            if not candidates:
                print(f"❌ Movie '{movie_input}' not found.")
                continue

            # Exact / smart match
            exact_match = next(
                (c for c in candidates if c.lower() == movie_input.lower()),
                None
            )

            if exact_match:
                target_movie = exact_match
            elif len(candidates) == 1:
                target_movie = candidates[0]
                print(f"✔ Found: {target_movie}")
            else:
                print("🔍 Did you mean:")
                for m in candidates[:10]:
                    print("-", m)
                print("Please type the exact movie name.")
                continue

            # Recommend
            recommendations = recommender.recommend(target_movie)
            print(f"\n🎯 Recommended movies for '{target_movie}':")
            for m in recommendations:
                print("-", m)

        except ValueError as e:
            print(f"⚠ Error: {e}")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting application.")
            break


if __name__ == "__main__":
    main()
