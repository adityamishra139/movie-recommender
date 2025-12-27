# 🎬 CineMatch: Content-Based Movie Recommender

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> **Don't just watch movies. Experience them.**
> CineMatch uses advanced natural language processing (NLP) and vector similarity to find your next favorite film based on genres, keywords, cast, and plot descriptions.

---

## 📜 Table of Contents
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🏗 Architecture & Tech Stack](#-architecture--tech-stack)
- [🔌 API Documentation](#-api-documentation)
- [📂 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)

---

## ✨ Features

- **🔍 Smart Search**: Handles typos and partial matches (e.g., searches for "avenger" suggest "The Avengers").
- **🧠 Advanced Recommendation**: Uses **Cosine Similarity** on a "soup" of metadata (Director + Top 3 Actors + Genres + Keywords).
- **⚡ Fast Inference**: Pre-computed model artifacts allow for millisecond-latency recommendations.
- **🖥 Dual Interface**:
    - **FastAPI**: Robust REST API for backend integration.
    - **Streamlit**: Beautiful, interactive web UI for end-users.

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/adityamishra139/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
```

### 2. Build Model Artifacts
Generate the optimized similarity matrix and vectorizer models.
```bash
python -m src.scripts.build_model
```
> *Output: artifacts/movies.pkl, artifacts/similarity.pkl*

### 3. Run the Application
You can run both the API and User Interface simultaneously.

**Backend (API):**
```bash
uvicorn api.main:app --reload --port 8000
```
*Visit http://127.0.0.1:8000/docs for interactive API swagger.*

**Frontend (UI):**
```bash
streamlit run streamlit_app.py
```
*Opens in your browser at http://localhost:8501*

---

## 🏗 Architecture & Tech Stack

The system follows a modular ML pipeline architecture:

| Component | Tech | Description |
|-----------|------|-------------|
| **Data Ingestion** | `pandas` | Loads TMDB 5000 Movies dataset. |
| **Feature Eng.** | `python` | Cleans text, merges cast/crew, creates "soup". |
| **Vectorization** | `scikit-learn` | `CountVectorizer` (Bag of Words) to convert text to numbers. |
| **Similarity** | `scikit-learn` | `Cosine Similarity` to calculate distance between vectors. |
| **API** | `FastAPI` | Serves recommendations via REST endpoints. |
| **UI** | `Streamlit` | Interactive frontend with session management. |

<details>
<summary><b>Click to see Recommendation Logic</b></summary>

1.  **Input**: User selects a movie (e.g., "The Dark Knight").
2.  **Lookup**: System retrieves the pre-computed vector for this movie.
3.  **Distance**: Calculates cosine similarity against all 4800+ movies in the database.
4.  **Sort**: Ranks top 5 closest vectors.
5.  **Output**: Returns titles of the most similar movies.
</details>

---

## 🔌 API Documentation

### `GET /recommend`

Returns a list of recommended movies or search suggestions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `movie` | `string` | The movie title to search for. |
| `top_n` | `int` | (Optional) Number of results. Default: 5. |

**Example Response:**
```json
{
  "input_movie": "The Avengers",
  "recommendations": [
    "Avengers: Age of Ultron",
    "Captain America: Civil War",
    "Captain America: The Winter Soldier",
    "Ant-Man",
    "Iron Man 2"
  ]
}
```

---

## 📂 Project Structure

```bash
movie-recommender/
├── api/
│   └── main.py              # FastAPI application entry point
├── artifacts/               # Saved model files (.pkl)
├── data/                    # Raw CSV datasets
├── src/
│   ├── data_ingestion/      # Data loading logic
│   ├── feature_engineering/ # Text processing & soup creation
│   ├── ml/                  # ML models (Vectorizer, Similarity)
│   ├── scripts/             # Build scripts for artifacts
│   └── utils/               # Persistence utilities
├── streamlit_app.py         # Frontend UI
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please fork this repository and submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
