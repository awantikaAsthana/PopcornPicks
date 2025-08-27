import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
 
 

#Build Feature
def build_feature_space(
    df: pd.DataFrame,
    alpha_genres=0.5,
    alpha_tags=0.3,
    alpha_year=0.1,
    alpha_rating=0.1
):
    # Genres
    genres_vec = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False)
    X_genres = genres_vec.fit_transform(df["genres_tokens"])

    # Tags
    tags_vec = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False)
    X_tags = tags_vec.fit_transform(df["tags"])

    # Title n-grams
    title_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))
    X_title = title_vec.fit_transform(df["title_clean"])

    # Rating
    scaler = MinMaxScaler()
    rating_scaled = scaler.fit_transform(df[["rating"]].values)
    X_rating = csr_matrix(rating_scaled)

    # Year
    year_scaled = scaler.fit_transform(df[["year"]].fillna(df["year"].median()))
    X_year = csr_matrix(year_scaled)

    # Combine
    X = hstack([
        X_genres * alpha_genres,
        X_tags   * alpha_tags,
        X_title  * 0.2,          # keep some weight for title
        X_year   * alpha_year,
        X_rating * alpha_rating
    ]).tocsr()

    artifacts = {
        "genres_vec": genres_vec,
        "tags_vec": tags_vec,
        "title_vec": title_vec,
        "scaler": scaler,
        "weights": (alpha_genres, alpha_tags, alpha_year, alpha_rating)
    }
    return X, artifacts

