import streamlit as st
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
encoder_path = os.path.join(BASE_DIR, "models", "target_encoder.pkl")

model = pickle.load(open(model_path, "rb"))
encoder = pickle.load(open(encoder_path, "rb"))

st.title("Perfume Recommendation System")

# Inputs
personality = st.selectbox("Personality", ["Confident", "Calm", "Adventurous"])
mood = st.selectbox("Mood", ["Happy", "Focused", "Relaxed"])
lifestyle = st.selectbox("Lifestyle", ["Corporate", "Socialite", "Minimalist", "Entrepreneur"])
season = st.selectbox("Season", ["Spring", "Summer", "Winter", "Autumn"])

# Encoding
personality_map = {"Confident": 0, "Calm": 1, "Adventurous": 2}
mood_map = {"Happy": 0, "Focused": 1, "Relaxed": 2}
lifestyle_map = {"Corporate": 0, "Socialite": 1, "Minimalist": 2, "Entrepreneur": 3}
season_map = {"Spring": 0, "Summer": 1, "Winter": 2, "Autumn": 3}

# Input DataFrame
input_data = pd.DataFrame([[
    personality_map[personality],
    mood_map[mood],
    lifestyle_map[lifestyle],
    season_map[season]
]], columns=["personality", "mood", "lifestyle", "season"])

# Predict
if st.button("Recommend Perfume"):
    prediction = model.predict(input_data)

    # IMPORTANT FIX HERE
    result = encoder.inverse_transform(prediction)

    st.success(f"Recommended Fragrance Family: {result[0]}")
