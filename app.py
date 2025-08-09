import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import zipfile
import os

# --- UNZIP AND LOAD DATA ---
#with zipfile.ZipFile("archive (7).zip", 'r') as zip_ref:"
    #"zip_ref.extractall("data")"

# Load the CSV (adjust the actual name if different)
df = pd.read_csv("/Users/manvijacherukuri/Downloads/songs_normalize.csv")

# --- DATA PREPROCESSING ---
# Drop duplicates
df = df.drop_duplicates(subset='song').reset_index(drop=True)

# Extract numeric features
numeric_features = ['tempo', 'valence', 'energy', 'danceability', 'acousticness']

# Handle genre (first genre only)
df['main_genre'] = df['genre'].apply(lambda x: x.split(',')[0].strip())

# One-hot encode genres
genre_encoded = pd.get_dummies(df['main_genre'], prefix='genre')

# Final feature set
features = pd.concat([genre_encoded, df[numeric_features]], axis=1)

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Fit KNN
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(features_scaled)

# --- STREAMLIT APP UI ---
st.title("🎧 Song Recommender System")
st.write("Get similar songs based on audio features + genre")

# Dropdown to select song
song_list = df['song'].tolist()
selected_song = st.selectbox("Choose a song:", song_list)

if selected_song:
    song_index = df[df['song'] == selected_song].index[0]
    distances, indices = knn.kneighbors([features_scaled[song_index]])

    input_song = df.iloc[song_index][['song', 'artist', 'main_genre']]
    recommended = df.iloc[indices[0][1:]][['song', 'artist', 'main_genre']]

    st.subheader("🎵 Input Song")
    st.write(f"**{input_song['song']}** by {input_song['artist']} ({input_song['main_genre']})")

    st.subheader("🎶 Recommended Songs")
    for i, row in recommended.iterrows():
        st.markdown(f"**{row['song']}** by {row['artist']}  \n*Genre:* {row['main_genre']}")