import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_ingestion.data_loader import load_data

def check_titles():
    movies, _ = load_data()
    titles = movies['title'].unique()
    
    queries = ["spider", "avenger", "conjuring"]
    
    print("Checking for similar titles...")
    for q in queries:
        matches = [t for t in titles if q.lower() in str(t).lower()]
        print(f"\nMatches for '{q}':")
        for m in matches:
            print("-", m)

if __name__ == "__main__":
    check_titles()
