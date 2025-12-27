import pandas as pd
import os

def load_data():
    """
    Loads the movie and credits datasets from raw csv files.
    Returns:
        movies (pd.DataFrame): The movies dataset.
        credits (pd.DataFrame): The credits dataset.
    """
    # Define absolute paths using os.path.join for better compatibility
    # Assuming the script is run from the project root or we can rely on relative paths from the workspace root
    # Ideally, we should find the project root dynamically, but for now we'll assumes a standard structure
    # or use absolute paths if provided by the environment, but here we'll use paths relative to the project root
    # given the workspace is /Users/rentomojo/Desktop/movie-recommender
    
    project_root = '/Users/rentomojo/Desktop/movie-recommender'
    movies_path = os.path.join(project_root, 'data', 'raw', 'tmdb_5000_movies.csv')
    credits_path = os.path.join(project_root, 'data', 'raw', 'tmdb_5000_credits.csv')

    try:
        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)
        return movies, credits
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None
