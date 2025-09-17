import joblib
from Pop import MovieRecommender

bundle = joblib.load("movie_recommender_latest.joblib")
rec_loaded = MovieRecommender().load(bundle)

print(rec_loaded.recommend(["scarface"], k=5, min_rating=3.0))
