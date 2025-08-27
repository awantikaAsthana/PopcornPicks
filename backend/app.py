from flask import Flask, request, jsonify
from MoviePickModel.index import rec_loaded

app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True)

    # Get movies (accepts string or list)
    movies = data.get("movies", [])
    if isinstance(movies, str):
        movies = [movies]

    if not movies:
        return jsonify({"error": "Please provide at least one movie"}), 400

    # Get params with defaults
    k = int(data.get("count", 5))
    min_rating = float(data.get("min_rating", 0.0))

    # Get recommendations
    recs = rec_loaded.recommend(movies, k=k, min_rating=min_rating)

 
    results = []
    for _, row in recs.iterrows():
        results.append({
            "title": row.get("title"),
            "genres": row.get("genres"),
            "similarity": round(float(row.get("similarity", 0.0))*100, 3),  # 0-1
            "year": int(row["year"]) if "year" in row and row["year"] else None,
            "movieId": int(row["movieId"]) if "movieId" in row and row["movieId"] else None,
            "imdb_search": row.get("imdb_search")
        })

    return jsonify({
        "input_movies": movies,
        "count": k,
        "min_rating": min_rating,
        "recommendations": results
    })


if __name__ == "__main__":
    app.run(debug=True)
