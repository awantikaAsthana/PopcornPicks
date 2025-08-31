from flask import Flask, request, jsonify
from MoviePickModel.index import rec_loaded
from flask_cors import CORS
app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

# ---------- Global Error Handlers ----------
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "message": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "message": str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server Error", "message": str(e)}), 500


# ---------- Recommendation API ----------
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        print(data)
        movies = data.get("movies")
        if not movies:
            return jsonify({"error": "Please provide 'movies'"}), 400

        if isinstance(movies, str):
            movies = [movies]

        k = int(data.get("count", 5))
        min_rating = float(data.get("min_rating", 0.0))

        # Call model
        recs = rec_loaded.recommend(movies, k=k, min_rating=min_rating)

        results = []
        for _, row in recs.iterrows():
            imdb_link = f"https://www.imdb.com/find?q={str(row.get('title','')).replace(' ', '+')}"
            results.append({
                "title": row.get("title"),
                "genres": row.get("genres"),
                "similarity": round(float(row.get("similarity", 0.0)), 3),
                "year": int(row["year"]) if "year" in row and row["year"] else None,
                "movieId": int(row["movieId"]) if "movieId" in row and row["movieId"] else None,
                "imdb_search": imdb_link
            })
        print(results)
        return jsonify({
            "input_movies": movies,
            "count": k,
            "min_rating": min_rating,
            "recommendations": results
        })

    except Exception as e:
        # Any unexpected error
        print(e)
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
