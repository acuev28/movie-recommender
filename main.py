import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Convert the 'genres' and 'keywords' JSON fields into a single space-separated string
def genres_and_keywords_to_string(row):
    # Parse and clean genres
    genres = ' '.join(g["name"].replace(" ", "") for g in json.loads(row["genres"]))
    # Parse and clean keywords
    keywords = ' '.join(k["name"].replace(" ", "") for k in json.loads(row["keywords"]))
    # Return combined string
    return f"{genres} {keywords}"


# Recommend similar movies based on cosine similarity of TF-IDF vectors
def recommend(title, data, df, movie_2_id):
    # Get index of the movie title in the dataset
    idx = movie_2_id.get(title)
    if idx is None:
        print(f"Movie '{title}' not found.")
        return []

    # If there are multiple entries, take the first one
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # Extract the TF-IDF vector for the given movie
    query = data[idx]

    # Compute cosine similarity between the query vector and all others
    scores = cosine_similarity(query, data).flatten()

    # Sort scores in descending order and get indices of top 5 matches (excluding the movie itself)
    recommend_idx = (-scores).argsort()[1:6]

    # Return the titles of the recommended movies
    return df["title"].iloc[recommend_idx]


def main():
    # Load the movie dataset
    df = pd.read_csv("tmdb_5000_movies.csv")

    # Create a new column with a processed string of genres and keywords
    df["string"] = df.apply(genres_and_keywords_to_string, axis=1)

    # Initialize the TF-IDF vectorizer and transform the string column
    tfidf = TfidfVectorizer(max_features=2000)
    data = tfidf.fit_transform(df["string"])

    # Create a mapping from movie title to its index in the DataFrame
    movie_2_id = pd.Series(df.index, index=df["title"])

    title = input("Enter a movie title (case-sensitive, e.g., 'Scream 3'): ")
    recommendations = recommend(title, data, df, movie_2_id) # Get recommendations for other movies

    print(f"Recommendations for '{title}':\n{recommendations}")


if __name__ == "__main__":
    main()
