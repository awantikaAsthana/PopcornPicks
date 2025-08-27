import pandas as pd
import joblib
import warnings
from .train import MovieRecommender   
import os
warnings.filterwarnings("ignore")

# Get the backend folder path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/

# CSV path
CSV_PATH = os.path.join(BASE_DIR, "data", "merged_final.csv")
# Model path
MODEL_PATH = os.path.join(BASE_DIR, "data", "movie_recommender_latest.joblib")

# Load CSV
df = pd.read_csv(CSV_PATH)

# Load model bundle
bundle = joblib.load(MODEL_PATH)
# Recreate the MovieRecommender instance and populate its attributes

rec_loaded = MovieRecommender()
rec_loaded.df = bundle["df"]
rec_loaded.X = bundle["X"]
rec_loaded.art = bundle["art"]
rec_loaded.nn = bundle["nn"]
rec_loaded._title_to_idx = {}
for i, t in enumerate(rec_loaded.df["title_clean"].str.lower()):
    rec_loaded._title_to_idx.setdefault(t, []).append(i)
# #Example usage
# print(rec_loaded.recommend(["scarface"], k=5, min_rating=3.0))
