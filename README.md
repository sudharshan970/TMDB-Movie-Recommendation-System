# 🎬 TMDB Movie Universe — Movie Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)

### 🚀 [Live Demo](https://tmdb-movie-recommendation-system-vtg9inbcsnjrmkg3twgrmw.streamlit.app/) &nbsp;|&nbsp; 📊 [Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) &nbsp;|&nbsp; ☁️ [Dataset Google Drive](https://drive.google.com/file/d/1xm86kNhMV77blxYhERMO_tr0UQnMOr5V/view?usp=sharing)

**A stunning dark-neon cinema dashboard with 1M+ movies, live TMDB API posters,
ML-powered recommendations, and rich analytics — built with Streamlit.**



</div>

---

## ✨ Features

| Tab | What's Inside |
|-----|--------------|
| 🏆 **Top Movies** | Bayesian-weighted ranking, poster cards from dataset, live TMDB feed |
| 🎯 **Recommender** | TF-IDF + cosine similarity, match % badges, poster previews |
| 📊 **Analytics** | Timeline, genre heatmap, rating distribution, language breakdown |
| 💰 **Box Office** | Budget vs revenue scatter, profit rankings, genre profitability |
| 🌍 **Explore** | Full-text search, filter by genre/year/rating, poster grid |

### Key Highlights
- 🎨 **Dark neon cinema UI** — Cinzel + Inter fonts, purple/pink/cyan palette
- 🖼️ **Dataset posters** — reads `poster_path` column directly, no extra API calls
- 🌐 **Live TMDB API** — real-time top-rated movies fetched fresh
- 🤖 **Smart Recommender** — TF-IDF on overview + genres, cosine similarity
- 📊 **Interactive Plotly charts** — dark-themed, hover-enabled, fully responsive
- ⚡ **Streamlit caching** — fast re-renders with `@st.cache_data`
- 🔧 **Auto-fallback** — works with CSV or TMDB API alone

---

## 📦 Dataset

> ⚠️ The dataset is ~200MB and cannot be uploaded to GitHub directly.
> **Download it from one of the links below and place it in the project folder.**

| Source | Link |
|--------|------|
| 🏆 Kaggle (Official) | [TMDB Movies Dataset 2024](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) |
| ☁️ Google Drive (Mirror) | [Download TMDB_movie_dataset_v11.csv](https://drive.google.com/file/d/1xm86kNhMV77blxYhERMO_tr0UQnMOr5V/view?usp=sharing) |

**After downloading, place the file here:**
```
Movie Recommendation System/
└── TMDB_movie_dataset_v11.csv   ← place here
```

> **No dataset?** The app automatically fetches ~500 popular movies
> from the TMDB API as a demo — no setup needed.

---

## 🛠️ Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/tmdb-movie-universe.git
cd tmdb-movie-universe
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset *(optional but recommended)*
Download from Kaggle or Google Drive link above and place
`TMDB_movie_dataset_v11.csv` in the project folder.

### 4. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. 🎉

---

## 🚀 Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy**

Your live URL will be:
```
https://YOUR-USERNAME-tmdb-movie-universe.streamlit.app
```

> **Dataset on Streamlit Cloud:** Upload `TMDB_movie_dataset_v11.csv`
> to Google Drive, make it public, and update the Google Drive link
> in `app.py` (see section below).

---

## ☁️ Google Drive Dataset Setup

If the CSV is too large for GitHub, host it on Google Drive:

1. Upload `TMDB_movie_dataset_v11.csv` to Google Drive
2. Right-click → **Share** → **Anyone with the link can view**
3. Copy the file ID from the URL:
```
   https://drive.google.com/file/d/FILE_ID_HERE/view
```
4. Update `app.py` — the `load_and_clean()` function already
   supports Google Drive. Just replace the link:
```python
   GDRIVE_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID_HERE"
```

---

## 📁 Project Structure
```
Movie Recommendation System/
├── app.py                        # 🎯 Main Streamlit application
├── requirements.txt              # 📦 Python dependencies
├── README.md                     # 📖 This file
└── TMDB_movie_dataset_v11.csv    # 📊 Dataset (download separately)
```

---

## 🔑 TMDB API Key

The TMDB API key is included for live poster fetching:
```
559a9cd882b40c3e18995bb93a5c49b3
```
Get your own free key at [developers.themoviedb.org](https://developers.themoviedb.org)

---

## 🎨 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.32 | Web framework + UI |
| `pandas` | ≥2.0 | Data manipulation |
| `numpy` | ≥1.24 | Numerical operations |
| `plotly` | ≥5.18 | Interactive charts |
| `scikit-learn` | ≥1.3 | TF-IDF + cosine similarity |
| `requests` | ≥2.31 | TMDB API calls |
| `Pillow` | ≥10.0 | Image handling |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built with ❤️ using Streamlit + TMDB API<br>
⭐ Star this repo if you found it useful!
</div>
