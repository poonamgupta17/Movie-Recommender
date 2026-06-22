import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('movies.csv')
df.reset_index(inplace=True)

# Features to combine for content-based recommendation
features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Fill NaN values with empty string
for feature in features:
    df[feature] = df[feature].fillna('')

# Combine all selected features into a single string
combinedFeatures = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']

# Vectorize the combined features
vectorizer = TfidfVectorizer()
vectorizedData = vectorizer.fit_transform(combinedFeatures)

# Compute cosine similarity between movies
similarity = cosine_similarity(vectorizedData)

# Streamlit UI with Emojis and Styling
st.set_page_config(page_title="🎬 Movie Recommendation System", page_icon="🎥", layout="centered")

st.title("🎬 Movie Recommendation System 🎥")
st.markdown(
    """
    Welcome to the Movie Recommendation System! 🎉
    - Enter a movie name 🎥
    - We'll find the closest match and recommend similar movies based on genres, cast, and more! 🎬
    """
)

# Ask the user to input a movie name
movie_name = st.text_input("🔍 Enter the name of a movie:")

if movie_name:
    # List of all movie titles
    titles = df['title'].tolist()

    # Get close matches using difflib
    closeMatch = difflib.get_close_matches(movie_name, titles, n=3, cutoff=0.5)

    if not closeMatch:
        st.write("❌ No matching movie found. Please check the spelling. ✨")
        st.write("Here are some suggestions you might like: ✨")
        
        # Show top 5 titles that contain the input as a substring
        suggestions = [title for title in titles if movie_name.lower() in title.lower()]
        suggestions = suggestions[:5]

        if suggestions:
            for s in suggestions:
                st.write(f"🔹 {s}")
        else:
            st.write("🚫 No suggestions available. Try using simpler or more accurate movie names. 📜")

    else:
        closestMatch = closeMatch[0]
        st.write(f"✅ Closest match: {closestMatch}")

        # Find the index of the matched movie
        idx = df[df.title == closestMatch]['index'].values[0]

        # Get similarity scores for the matched movie
        similarityScore = list(enumerate(similarity[idx]))
        sortedSimilarityScore = sorted(similarityScore, key=lambda x: x[1], reverse=True)

        st.write("🌟 Movies Suggested for you:")

        # Display the top 30 most similar movies
        for i, movie in enumerate(sortedSimilarityScore[:30]):
            index = movie[0]
            titleIdx = df[df.index == index]['title'].values[0]
            st.write(f"🎬 {i+1}. {titleIdx}")
        
        st.write("✨ Enjoy your movie discovery! 🍿")

# Footer Section
st.markdown(
    """
    ---
    Created with ❤️ by **Poonam** | [GitHub](https://github.com/poonamgupta17) | [LinkedIn](https://www.linkedin.com/in/poonamgupta17/)
    """
)

# To run the app, use the command: streamlit run app.py