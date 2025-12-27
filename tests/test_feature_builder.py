import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion.data_loader import load_data
from src.feature_engineering.feature_builder import FeatureBuilder

def test_feature_engineering():
    # Mocking data loading for faster/isolated testing or loading real data if preferred
    # For integration testing, we use the real loader
    print("Loading data...")
    movies, credits = load_data()
    if movies is None or credits is None:
        print("Failed to load data")
        return

    # Merge dataset as expected before builder
    df = movies.merge(credits, left_on='id', right_on='movie_id')

    print("Building features...")
    builder = FeatureBuilder()
    processed_df = builder.build_features(df)

    # Verifications
    required_columns = ['genres', 'keywords', 'cast', 'director', 'soup']
    missing_cols = [col for col in required_columns if col not in processed_df.columns]
    
    if missing_cols:
        print(f"FAILURE: Missing columns: {missing_cols}")
        sys.exit(1)

    # Check content types
    if not isinstance(processed_df['genres'].iloc[0], list):
        print("FAILURE: 'genres' column does not contain lists")
        sys.exit(1)
        
    if not isinstance(processed_df['soup'].iloc[0], str):
        print("FAILURE: 'soup' column does not contain strings")
        print(f"Got type: {type(processed_df['soup'].iloc[0])}")
        sys.exit(1)

    print("SUCCESS: Feature engineering completed")
    print("\nSample 'soup' entry:")
    print(processed_df['soup'].iloc[0][:100] + "...")
    print("\nColumns:", processed_df.columns.tolist())

if __name__ == "__main__":
    test_feature_engineering()
