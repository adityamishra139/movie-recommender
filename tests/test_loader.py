import sys
import os
# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion.data_loader import load_data

def test_loading():
    print("Attempting to load data...")
    movies, credits = load_data()
    
    if movies is not None and credits is not None:
        print("SUCCESS: Data loaded successfully.")
        print(f"Movies shape: {movies.shape}")
        print(f"Credits shape: {credits.shape}")
        print("Movies columns:", movies.columns.tolist()[:5])
    else:
        print("FAILURE: Data returned as None.")

if __name__ == "__main__":
    test_loading()
