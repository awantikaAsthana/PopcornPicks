import pandas as pd
import joblib
from model import MovieRecommender
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("./data/merged_final.csv")
rec = MovieRecommender().fit(df)


#Loading the model already created

bundle = joblib.load("./data/movie_recommender_latest.joblib")
rec_loaded = MovieRecommender()
rec_loaded.df = bundle["df"]
rec_loaded.X = bundle["X"]
rec_loaded.art = bundle["art"]
rec_loaded.nn = bundle["nn"]

#Example usage
print(rec.recommend(["scarface"], k=5, min_rating=3.0))