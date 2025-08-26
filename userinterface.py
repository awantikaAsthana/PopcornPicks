import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from pop import MovieRecommender
# assume you already have: from your_code import MovieRecommender
# and recommender = MovieRecommender().fit(df)

class PopcornPicksUI:
    def __init__(self, root, rec):
        self.root = root
        self.rec = rec
        self.root.title("🎬 PopcornPicks - Movie Recommender")
        self.root.geometry("800x600")
        
        # --- Input frame ---
        frame_top = tk.Frame(root, padx=10, pady=10)
        frame_top.pack(fill="x")

        tk.Label(frame_top, text="Enter Movie Title:", font=("Arial", 12)).grid(row=0, column=0, sticky="w")
        self.entry_title = tk.Entry(frame_top, width=40, font=("Arial", 12))
        self.entry_title.grid(row=0, column=1, padx=5)

        tk.Label(frame_top, text="Min Rating:", font=("Arial", 12)).grid(row=1, column=0, sticky="w")
        self.entry_rating = tk.Entry(frame_top, width=10, font=("Arial", 12))
        self.entry_rating.grid(row=1, column=1, sticky="w")

        tk.Label(frame_top, text="Genres (comma separated):", font=("Arial", 12)).grid(row=2, column=0, sticky="w")
        self.entry_genres = tk.Entry(frame_top, width=40, font=("Arial", 12))
        self.entry_genres.grid(row=2, column=1, padx=5)

        # tk.Label(frame_top, text="Tags (comma separated):", font=("Arial", 12)).grid(row=3, column=0, sticky="w")
        # self.entry_tags = tk.Entry(frame_top, width=40, font=("Arial", 12))
        # self.entry_tags.grid(row=3, column=1, padx=5)

        # tk.Label(frame_top, text="Year Range (e.g. 1990-2010):", font=("Arial", 12)).grid(row=4, column=0, sticky="w")
        # self.entry_year = tk.Entry(frame_top, width=20, font=("Arial", 12))
        # self.entry_year.grid(row=4, column=1, sticky="w")

        tk.Button(frame_top, text="Recommend 🎥", font=("Arial", 12, "bold"), 
                  command=self.show_recommendations).grid(row=5, columnspan=2, pady=10)

        # --- Results frame ---
        frame_results = tk.Frame(root, padx=10, pady=10)
        frame_results.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(frame_results, columns=("title","year","genres","rating","score"), show="headings", height=15)
        self.tree.pack(fill="both", expand=True)

        self.tree.heading("title", text="Title")
        self.tree.heading("year", text="Year")
        self.tree.heading("genres", text="Genres")
        self.tree.heading("rating", text="Rating")
        self.tree.heading("score", text="Score")

        self.tree.column("title", width=250)
        self.tree.column("year", width=70)
        self.tree.column("genres", width=200)
        self.tree.column("rating", width=70)
        self.tree.column("score", width=70)

    def show_recommendations(self):
        title = self.entry_title.get().strip()
        if not title:
            messagebox.showerror("Error", "Please enter a movie title.")
            return
        
        try:
            min_rating = float(self.entry_rating.get()) if self.entry_rating.get() else None
        except:
            min_rating = None
        
        genres = [g.strip() for g in self.entry_genres.get().split(",") if g.strip()]
        tags = [t.strip() for t in self.entry_tags.get().split(",") if t.strip()]
        
        year_range = None
        if self.entry_year.get():
            try:
                y0, y1 = map(int, self.entry_year.get().split("-"))
                year_range = (y0, y1)
            except:
                messagebox.showerror("Error", "Year range must be like 1990-2010")
                return

        try:
            recs = self.rec.recommend(
                [title],
                k=10,
                min_rating=min_rating,
                must_have_genres=genres or None,
                tags=tags or None,
                year_range=year_range
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        
        # clear previous
        for i in self.tree.get_children():
            self.tree.delete(i)

        # insert new
        for _, row in recs.iterrows():
            self.tree.insert("", "end", values=(
                row["title"],
                row["year"],
                row["genres"],
                f"{row['rating']:.2f}",
                f"{row['score']:.3f}"
            ))

# --- Example usage ---
if __name__ == "__main__":
    df = pd.read_csv("merged_final.csv")  # <-- replace with your enriched df
    rec = MovieRecommender().fit(df)

    root = tk.Tk()
    app = PopcornPicksUI(root, rec)
    root.mainloop()
