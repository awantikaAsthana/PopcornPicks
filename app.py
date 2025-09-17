from flask import Flask, jsonify, request
from Pop import MovieRecommender
import joblib
import pandas as pd
from flask_cors import CORS

# Flask app create
app = Flask(__name__)
CORS(app)
# Load trained recommender from joblib
bundle = joblib.load("movie_recommender_latest.joblib")
rec_loaded = MovieRecommender().load(bundle)

@app.route('/')
def home():
    return "🎬 Movie Recommender API is running!"

@app.route('/recommend', methods=['POST'])
def recommend():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    print("Received:", data)

    title = data.get("title")
    count = int(data.get("count", 5))
    rating = float(data.get("rating", None))

    # ✅ Normalize: always make title a list
    if isinstance(title, str):
        titles = [title]
    elif isinstance(title, list):
        titles = title
    else:
        return jsonify({"error": "Title must be a string or list of strings"}), 400

    try:
        result = rec_loaded.recommend(titles, k=count, min_rating=rating)
        result_json = result.to_dict(orient="records")
        return jsonify({"query": titles, "recommendations": result_json}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

 

if __name__ == "__main__":
    app.run(debug=True)
