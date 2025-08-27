from sklearn.neighbors import NearestNeighbors
from rapidfuzz import process, fuzz
from scipy.sparse import vstack
import urllib.parse
import re
import numpy as np
import pandas as pd
from feature import build_feature_space 

#Movie Recommender

class MovieRecommender:
    def __init__(self):
        self.df = None
        self.X = None
        self.art = None
        self.nn = None

    def fit(self, df: pd.DataFrame, alpha_genres=0.5, alpha_tags=0.3, alpha_year=0.1, alpha_rating=0.1):
        self.df = df.copy()
        self.X, self.art = build_feature_space(
            self.df,
            alpha_genres=alpha_genres,
            alpha_tags=alpha_tags,
            alpha_year=alpha_year,
            alpha_rating=alpha_rating
        )
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(self.X)

        # Build lookup
        self._title_to_idx = {}
        for i, t in enumerate(self.df["title_clean"].str.lower()):
            self._title_to_idx.setdefault(t, []).append(i)
        return self

    def _encode_texts(self, titles: list[str]):
        # Convert list of raw titles to a single seed vector
        titles_norm = [re.sub(r"\s*\(\d{4}\)\s*$","", t).strip().lower() for t in titles]
        idxs = []
        for t in titles_norm:
            if t in self._title_to_idx:
                idxs.append(self._title_to_idx[t][0])
            else:
                # Fuzzy match top candidate
                cand, score, _ = process.extractOne(
                    t,
                    self.df["title_clean"].str.lower().tolist(),
                    scorer=fuzz.WRatio
                )
                if score >= 80:
                    idxs.append(self.df[self.df["title_clean"].str.lower()==cand].index[0])
        if not idxs:
            raise ValueError("None of the provided titles were found (even after fuzzy matching).")
        return idxs

    def _make_seed_vector_from_indices(self, idxs: list[int]):
        # Average the feature rows of the seed movies
        seed_mat = vstack([self.X[i] for i in idxs])
        seed_vec = np.asarray(seed_mat.mean(axis=0)).reshape(1, -1)
        return seed_vec

    def _post_rank(self, indices, sims, min_rating=None,
               must_have_genres=None, year_range=None,
               tags=None, exclude_idxs=None):
      rows = []
      exclude_set = set(exclude_idxs or [])
      for idx, sim in zip(indices, sims):
          if idx in exclude_set:
              continue
          row = self.df.iloc[idx]

          # Filters
          if min_rating is not None and row["rating"] < min_rating:
              continue
          if must_have_genres:
              movie_genres = set(row["genres"].split("|"))
              if not set(must_have_genres).issubset(movie_genres):
                  continue
          if year_range and not pd.isna(row["year"]):
              y0, y1 = year_range
              if not (y0 <= row["year"] <= y1):
                  continue

          # ✅ Tag-based boost using seed movie tags
          tag_boost = 0.0
          if tags and isinstance(row["tags"], str):
              movie_tags = set(row["tags"].split())
              overlap = len(set(tags) & movie_tags)
              tag_boost = overlap / max(len(tags), 1)

          rows.append((idx, float(1.0 - sim), tag_boost))

      # Blended score
      out = []
      for idx, sim, tag_boost in rows:
          r = float(self.df.iloc[idx]["rating"])
          blended = (0.8 * sim) + (0.1 * (r / 5.0)) + (0.1 * tag_boost)
          out.append((idx, sim, blended))

      out.sort(key=lambda x: x[2], reverse=True)
      return out


    def recommend(
      self,
      titles: list[str],
      k: int = 10,
      min_rating: float | None = None,
      must_have_genres: list[str] | None = None,
      year_range: tuple[int,int] | None = None
  ) -> pd.DataFrame:
      seed_idxs = self._encode_texts(titles)
      seed_vec = self._make_seed_vector_from_indices(seed_idxs)

      # ✅ Collect all tags from the seed movies
      seed_tags = set()
      for idx in seed_idxs:
          if isinstance(self.df.iloc[idx]["tags"], str):
              seed_tags.update(self.df.iloc[idx]["tags"].split())

      # Query more than k to allow for filtering and re-ranking
      n_query = min(self.X.shape[0], max(50, k*5))
      distances, indices = self.nn.kneighbors(seed_vec, n_neighbors=n_query)
      indices = indices.ravel().tolist()
      distances = distances.ravel().tolist()

      # Post-filter & rerank (pass in seed_tags)
      scored = self._post_rank(indices, distances, min_rating, must_have_genres,
                              year_range, seed_tags, exclude_idxs=seed_idxs)
      scored = scored[:k]

      rec_df = self.df.loc[[i for i,_,_ in scored]].copy()
      rec_df["similarity"] = [s for _,s,_ in scored]
      rec_df["score"] = [sc for *_, sc in scored]

      # IMDb search URL
      def imdb_search_url(title, year):
          q = title if pd.isna(year) else f"{title} ({int(year)})"
          return "https://www.imdb.com/find/?q=" + urllib.parse.quote(q)
      rec_df["imdb_search"] = rec_df.apply(lambda r: imdb_search_url(r["title_clean"], r["year"]), axis=1)

      return rec_df[["movieId","title","year","genres","tags","rating","similarity","score","imdb_search"]].reset_index(drop=True)
