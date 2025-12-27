import streamlit as st
import requests

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Content-based recommender using cosine similarity")

API_URL = "http://127.0.0.1:8000/recommend"

movie_name = st.text_input("Enter a movie name")

if st.button("Recommend"):
    if not movie_name.strip():
        st.warning("Please enter a movie name.")
    else:
        with st.spinner("Finding recommendations..."):
            response = requests.get(
                API_URL,
                params={"movie": movie_name}
            )

            data = response.json()

            if "error" in data:
                st.error(data["error"])
                if "suggestions" in data and data["suggestions"]:
                    st.write("Did you mean:")
                    for s in data["suggestions"]:
                        st.write("-", s)
            else:
                st.success(f"Recommendations for **{data['input_movie']}**")
                for rec in data["recommendations"]:
                    st.write("👉", rec)
