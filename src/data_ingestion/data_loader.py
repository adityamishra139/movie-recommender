import pandas as pd
import os


def load_data():
    """
    Loads the movie and credits datasets using project-relative paths.
    Works both locally and on cloud platforms like Render.
    
    Returns:
        movies (pd.DataFrame)
        credits (pd.DataFrame)
    """
    try:
        # Absolute path to the project root
        # src/data_ingestion/data_loader.py -> src -> project root
        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )

        movies_path = os.path.join(
            base_dir, "data", "raw", "tmdb_5000_movies.csv"
        )
        credits_path = os.path.join(
            base_dir, "data", "raw", "tmdb_5000_credits.csv"
        )

        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)

        return movies, credits

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None
