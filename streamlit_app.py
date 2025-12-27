import streamlit as st
import requests

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Content-based recommender using cosine similarity")

API_URL = "http://127.0.0.1:8000/recommend"

# Initialize session state for auto-run
if "auto_run" not in st.session_state:
    st.session_state.auto_run = False

def set_movie_and_run(movie):
    st.session_state.search_term = movie
    st.session_state.auto_run = True

# Main input
# key="search_term" binds this input to st.session_state.search_term
movie_name = st.text_input("Enter a movie name", key="search_term")
search_clicked = st.button("Recommend")

if search_clicked or st.session_state.auto_run:
    # Reset auto_run flag immediately so it doesn't loop forever
    st.session_state.auto_run = False
    
    if not movie_name.strip():
        st.warning("Please enter a movie name.")
    else:
        with st.spinner("Finding recommendations..."):
            try:
                response = requests.get(
                    API_URL,
                    params={"movie": movie_name}
                )
                data = response.json()

                if "error" in data:
                    st.error(data["error"])
                    if "suggestions" in data and data["suggestions"]:
                        st.info("Did you mean one of these? Click to search:")
                        # Display suggestions as buttons
                        cols = st.columns(min(3, len(data["suggestions"])))
                        for idx, suggestion in enumerate(data["suggestions"]):
                            col = cols[idx % 3]
                            # When clicked, this calls set_movie_and_run, triggering a rerun
                            col.button(
                                suggestion, 
                                key=f"sugg_{idx}", 
                                on_click=set_movie_and_run, 
                                args=(suggestion,)
                            )
                else:
                    st.success(f"Recommendations for **{data['input_movie']}**")
                    for rec in data["recommendations"]:
                        st.write("👉", rec)
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
