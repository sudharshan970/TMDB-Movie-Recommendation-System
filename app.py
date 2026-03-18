import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import ast, warnings, os, re
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="🎬 TMDB Movie Universe",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#05050f;color:#e2d9f3;}
.stApp{background:#05050f;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2rem 2rem;max-width:100%;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0a1a,#0f0a1f);border-right:1px solid rgba(124,58,237,0.25);}
[data-testid="metric-container"]{background:linear-gradient(135deg,rgba(124,58,237,0.12),rgba(236,72,153,0.08));border:1px solid rgba(124,58,237,0.3);border-radius:14px;padding:1rem 1.2rem;}
[data-testid="metric-container"] label{color:#7b7a9a!important;font-size:0.78rem!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#a78bfa!important;font-size:1.6rem!important;font-weight:600!important;}
.stSelectbox>div>div,.stTextInput>div>div>input{background:#0f0f24!important;border:1px solid rgba(124,58,237,0.35)!important;border-radius:10px!important;color:#e2d9f3!important;}
.stButton>button{background:linear-gradient(135deg,#7c3aed,#ec4899);color:white!important;border:none;border-radius:10px;font-weight:600;padding:0.5rem 1.5rem;box-shadow:0 4px 20px rgba(124,58,237,0.4);}
.stTabs [data-baseweb="tab-list"]{background:#0f0f24;border-radius:12px;padding:4px;gap:4px;border:1px solid rgba(124,58,237,0.25);}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:9px;color:#7b7a9a;font-weight:500;padding:8px 20px;border:none;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#7c3aed,#ec4899)!important;color:white!important;}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:#0a0a1a;}
::-webkit-scrollbar-thumb{background:#7c3aed;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

TMDB_API_KEY  = "559a9cd882b40c3e18995bb93a5c49b3"
TMDB_BASE     = "https://api.themoviedb.org/3"
TMDB_IMG_W500 = "https://image.tmdb.org/t/p/w500"
TMDB_IMG_W185 = "https://image.tmdb.org/t/p/w185"
COLORS = ["#7c3aed","#ec4899","#06b6d4","#f59e0b","#10b981",
          "#ef4444","#8b5cf6","#f97316","#14b8a6","#e879f9"]
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,13,34,0.85)",
    font=dict(family="Inter,sans-serif", color="#e2d9f3"),
    colorway=COLORS,
)

def make_poster_url(poster_path, size="w500"):
    if not poster_path or pd.isna(poster_path): return None
    p = str(poster_path).strip()
    if p in ("","nan","None"): return None
    if p.startswith("http"): return p
    if not p.startswith("/"): p = "/" + p
    return (TMDB_IMG_W185 if size=="w185" else TMDB_IMG_W500) + p

def banner(title, subtitle="", emoji="🎬"):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f0a1f,#0a1a2e);
                padding:20px 28px;border-radius:16px;margin-bottom:20px;
                border:1px solid rgba(124,58,237,0.35);
                box-shadow:0 4px 32px rgba(124,58,237,0.15);">
        <h2 style="margin:0;color:#c4b5fd;font-family:'Cinzel',serif;
                   font-size:1.4em;letter-spacing:1px;">{emoji} {title}</h2>
        <p style="margin:6px 0 0;color:#7b7a9a;font-size:0.88em;">{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def genre_list_from_str(s):
    if not s or pd.isna(s): return []
    return [g for g in str(s).split("|") if g.strip()]

def movie_card(title, rating, year, overview, genres, poster_url,
               border="#7c3aed", sim=None, vote_count=None):
    gc = ["#7c3aed","#ec4899","#06b6d4","#f59e0b","#10b981","#ef4444"]
    gh = "".join(
        f'<span style="background:{gc[i%6]}33;color:{gc[i%6]};border:1px solid {gc[i%6]}66;padding:2px 9px;border-radius:20px;font-size:11px;margin:2px;display:inline-block;">{g}</span>'
        for i,g in enumerate(genres[:3])
    )
    if poster_url:
        ph = (f'<div style="width:100%;aspect-ratio:2/3;border-radius:10px;overflow:hidden;margin-bottom:10px;background:#1a0a2e;">' +
              f'<img src="{poster_url}" style="width:100%;height:100%;object-fit:cover;display:block;" ' +
              f'onerror="this.style.display=\'none\';this.parentElement.innerHTML=\'<div style=&quot;height:200px;display:flex;align-items:center;justify-content:center;font-size:48px;background:%231a0a2e;&quot;>🎬</div>\'" /></div>')
    else:
        ph = '<div style="width:100%;aspect-ratio:2/3;background:linear-gradient(135deg,#1a0a2e,#0a1a2e);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:48px;margin-bottom:10px;">🎬</div>'
    sb = (f'<div style="background:rgba(124,58,237,0.18);border-radius:6px;padding:3px 8px;margin:5px 0;text-align:center;"><span style="color:#a78bfa;font-size:11px;font-weight:600;">🎯 {float(sim)*100:.1f}% match</span></div>' if sim is not None else "")
    vc = ""
    if vote_count:
        try: vc = f'<span style="background:rgba(124,58,237,0.15);color:#a78bfa;padding:2px 8px;border-radius:20px;font-size:11px;border:1px solid rgba(124,58,237,0.3);">🗳️ {int(float(vote_count)):,}</span>'
        except: pass
    try:    yr = str(int(float(year))) if year and not pd.isna(year) else "?"
    except: yr = str(year) if year else "?"
    try:    rv = float(rating)
    except: rv = 0.0
    pct = max(0, min(100, rv/10*100))
    ts  = str(title)[:28].replace("<","&lt;").replace(">","&gt;")
    ov  = str(overview)[:90].replace("<","&lt;").replace(">","&gt;") + ("..." if len(str(overview))>90 else "")
    components.html(f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>*{{margin:0;padding:0;box-sizing:border-box;}}body{{background:transparent;font-family:'Inter',sans-serif;}}
.card{{background:linear-gradient(135deg,rgba(13,13,34,0.97),rgba(10,26,46,0.97));border:1px solid {border}55;border-radius:16px;padding:13px;box-shadow:0 6px 24px rgba(0,0,0,0.6);}}
.t{{color:#e2d9f3;font-weight:600;font-size:13px;margin:0 0 5px;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;}}
.r{{color:#f59e0b;font-size:13px;font-weight:700;}}.y{{color:#7b7a9a;font-size:11px;}}
.bg{{height:4px;background:rgba(255,255,255,0.08);border-radius:3px;margin:4px 0 6px;}}
.bf{{width:{pct:.1f}%;height:100%;border-radius:3px;background:linear-gradient(90deg,#7c3aed,#ec4899,#f59e0b);}}
.ov{{color:#7b7a9a;font-size:11px;margin-top:6px;line-height:1.5;}}</style></head><body>
<div class="card">{ph}<p class="t" title="{ts}">{ts}</p>
<div class="row"><span class="r">⭐ {rv:.1f}</span><span class="y">{yr}</span></div>
<div class="bg"><div class="bf"></div></div>{sb}
<div style="margin:5px 0;">{gh}</div>{vc}
<p class="ov">{ov}</p></div></body></html>""", height=480, scrolling=False)

@st.cache_data(show_spinner=False)
def load_and_clean():
    df = None
    source = "api"
    for fname in ["tmdb_top5k.csv","tmdb_top50k.csv","TMDB_movie_dataset_v11.csv","movies.csv"]:
        if os.path.exists(fname):
            try:
                df = pd.read_csv(fname, low_memory=False)
                source = fname
                break
            except Exception:
                pass
    if df is None:
        rows = []
        for page in range(1, 51):
            try:
                r = requests.get(f"{TMDB_BASE}/movie/popular",
                    params={"api_key":TMDB_API_KEY,"page":page}, timeout=8)
                if r.status_code == 200:
                    rows.extend(r.json().get("results",[]))
            except Exception:
                pass
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        source = "tmdb_api"
    st.session_state["data_source"] = source
    if df is None or len(df) == 0:
        df = pd.DataFrame({
            "title":["Sample Movie"],"overview":["A great movie."],
            "vote_average":[7.0],"vote_count":[1000],"genres":["Action"],
            "release_date":["2020-01-01"],"original_language":["en"],
            "popularity":[10.0],"budget":[0],"revenue":[0],
            "runtime":[120],"poster_path":[""]
        })
    df.columns = [str(c).lower().strip() for c in df.columns]
    for col in ["title","genres","vote_average","vote_count","release_date",
                "original_language","overview","budget","revenue","runtime",
                "popularity","poster_path"]:
        if col not in df.columns: df[col] = np.nan
    for col in ["vote_average","vote_count","budget","revenue","runtime","popularity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["budget"]  = df["budget"].replace(0, np.nan)
    df["revenue"] = df["revenue"].replace(0, np.nan)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"]   = df["release_date"].dt.year
    df["decade"] = (df["year"]//10*10).astype("Int64")
    def parse_genres(val):
        if pd.isna(val): return []
        val = str(val)
        if "[" in val:
            try:
                p = ast.literal_eval(val)
                if p and isinstance(p[0],dict): return [x["name"] for x in p]
                return [str(x).strip() for x in p if x]
            except Exception: pass
        return [x.strip() for x in val.split(",") if x.strip()]
    df["genre_str"]     = df["genres"].apply(lambda v:"|".join(parse_genres(v)))
    df["primary_genre"] = df["genre_str"].apply(lambda s:s.split("|")[0] if s else "Unknown")
    df["genre_count"]   = df["genre_str"].apply(lambda s:len(s.split("|")) if s else 0)
    df["profit"]        = df["revenue"] - df["budget"]
    df["overview"]      = df["overview"].fillna("").astype(str)
    df["title"]         = df["title"].fillna("Unknown").astype(str)
    df["poster_path"]   = df["poster_path"].fillna("").astype(str).replace("nan","")
    C = df["vote_average"].median()
    m = df["vote_count"].quantile(0.80)
    def ws(row):
        v = row["vote_count"] if pd.notna(row["vote_count"]) else 0
        R = row["vote_average"] if pd.notna(row["vote_average"]) else 0
        return 0.0 if v==0 else (v/(v+m))*R+(m/(v+m))*C
    df["score"] = df.apply(ws, axis=1)
    df.drop_duplicates(subset=["title"],keep="first",inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

@st.cache_data(show_spinner=False)
def build_recommender(_df, sample=5000):
    df_r = _df.copy()
    for col in ["title","overview","genre_str","primary_genre"]:
        df_r[col] = df_r[col].fillna("").astype(str).str.strip() if col in df_r.columns else ""
    if "vote_count" in df_r.columns and "score" in df_r.columns:
        vc   = pd.to_numeric(df_r["vote_count"],errors="coerce").fillna(0)
        df_r = df_r[vc>=10].sort_values("score",ascending=False).head(sample).reset_index(drop=True)
    else:
        df_r = df_r.head(sample).reset_index(drop=True)
    def make_corpus(row):
        try:
            ov = str(row["overview"]).strip()
            gs = " ".join(str(row["genre_str"]).split("|"))*2
            tt = str(row["title"]).strip()
            c  = f"{ov} {gs} {tt}".strip()
            return c if c else tt if tt else "movie"
        except Exception: return "movie"
    df_r["corpus"] = df_r.apply(make_corpus, axis=1).astype(str)
    df_r["corpus"] = df_r["corpus"].replace("","movie").fillna("movie")
    if len(df_r) < 10:
        df_r = _df.head(1000).reset_index(drop=True).copy()
        df_r["title"]  = df_r["title"].fillna("movie").astype(str)
        df_r["corpus"] = df_r["title"].str.lower()
    corpus_list = [str(c) if c and str(c).strip() else "movie" for c in df_r["corpus"].tolist()]
    tfidf = TfidfVectorizer(max_features=5000,stop_words=None,ngram_range=(1,1),
                            min_df=1,max_df=1.0,analyzer="word",token_pattern=r"(?u)\b\w+\b")
    mat = tfidf.fit_transform(corpus_list)
    idx = pd.Series(df_r.index, index=df_r["title"].str.lower())
    return df_r, mat, idx

@st.cache_data(show_spinner=False)
def fetch_live_top_rated():
    movies = []
    for page in range(1,5):
        try:
            r = requests.get(f"{TMDB_BASE}/movie/top_rated",
                params={"api_key":TMDB_API_KEY,"page":page},timeout=8)
            if r.status_code==200: movies.extend(r.json().get("results",[]))
        except Exception: pass
    return movies[:16]

with st.spinner("🎬 Loading TMDB Movie Universe..."):
    df = load_and_clean()
    df_rec, tfidf_mat, title_idx = build_recommender(df)

all_genres     = [g for s in df["genre_str"] for g in genre_list_from_str(s)]
all_genre_opts = sorted(set(all_genres))
src = st.session_state.get("data_source","")
st.markdown(f'<p style="color:#10b981;font-size:0.78em;text-align:right;margin:0 0 4px;">📂 {src} · {len(df):,} movies</p>',unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#05050f,#0f0a1f,#0a1a2e);
            padding:48px 40px 36px;border-radius:0 0 24px 24px;
            border-bottom:1px solid rgba(124,58,237,0.3);
            box-shadow:0 8px 48px rgba(124,58,237,0.18);text-align:center;margin-bottom:28px;">
    <div style="font-size:3.2em;margin-bottom:8px;">🎬</div>
    <h1 style="font-family:'Cinzel',serif;font-size:2.6em;
               background:linear-gradient(135deg,#c4b5fd,#f9a8d4,#67e8f9);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin:0;letter-spacing:3px;">TMDB MOVIE UNIVERSE</h1>
    <p style="color:#7b7a9a;font-size:1em;margin-top:12px;">
        1,000,000+ Movies · Analytics · ML Recommendations · Live Posters</p>
    <div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-top:20px;">
        <span style="background:rgba(124,58,237,0.2);color:#c4b5fd;padding:5px 16px;border-radius:20px;border:1px solid rgba(124,58,237,0.4);font-size:0.82em;">📊 Full EDA</span>
        <span style="background:rgba(236,72,153,0.2);color:#f9a8d4;padding:5px 16px;border-radius:20px;border:1px solid rgba(236,72,153,0.4);font-size:0.82em;">🏆 Top Rated</span>
        <span style="background:rgba(6,182,212,0.2);color:#67e8f9;padding:5px 16px;border-radius:20px;border:1px solid rgba(6,182,212,0.4);font-size:0.82em;">🎯 Recommender</span>
        <span style="background:rgba(245,158,11,0.2);color:#fcd34d;padding:5px 16px;border-radius:20px;border:1px solid rgba(245,158,11,0.4);font-size:0.82em;">🖼️ Dataset Posters</span>
    </div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:10px 0 6px;">
        <div style="font-size:2em;">🎬</div>
        <h3 style="font-family:'Cinzel',serif;color:#c4b5fd;margin:6px 0 2px;font-size:1em;">TMDB Universe</h3>
        <p style="color:#7b7a9a;font-size:0.75em;margin:0;">Movie Recommendation System</p>
    </div><hr style="border:none;border-top:1px solid rgba(124,58,237,0.3);margin:12px 0;">""",unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa;font-size:0.82em;font-weight:600;margin-bottom:6px;">🎭 GENRE</p>',unsafe_allow_html=True)
    sel_genre = st.selectbox("Genre",["All"]+all_genre_opts,label_visibility="collapsed")
    yr_min = int(df["year"].min()) if df["year"].notna().any() else 1950
    yr_max = int(df["year"].max()) if df["year"].notna().any() else 2025
    st.markdown('<p style="color:#a78bfa;font-size:0.82em;font-weight:600;margin:14px 0 6px;">📅 YEAR RANGE</p>',unsafe_allow_html=True)
    year_range = st.slider("Year",yr_min,yr_max,(2000,yr_max),label_visibility="collapsed")
    st.markdown('<p style="color:#a78bfa;font-size:0.82em;font-weight:600;margin:14px 0 6px;">⭐ MIN RATING</p>',unsafe_allow_html=True)
    min_rating = st.slider("Rating",0.0,10.0,5.0,0.1,label_visibility="collapsed")
    st.markdown('<p style="color:#a78bfa;font-size:0.82em;font-weight:600;margin:14px 0 6px;">🗳️ MIN VOTES</p>',unsafe_allow_html=True)
    min_votes  = st.selectbox("Votes",[0,100,500,1000,5000,10000],index=2,label_visibility="collapsed")
    has_p = df["poster_path"].str.strip().ne("").sum()
    st.markdown(f"""<div style="background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.25);border-radius:12px;padding:12px;text-align:center;margin-top:14px;">
        <div style="color:#7b7a9a;font-size:0.75em;">Movies Loaded</div>
        <div style="color:#a78bfa;font-size:1.5em;font-weight:700;">{len(df):,}</div></div>
    <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);border-radius:10px;padding:10px;text-align:center;margin-top:8px;">
        <div style="color:#7b7a9a;font-size:0.73em;">With Posters</div>
        <div style="color:#10b981;font-size:1.2em;font-weight:700;">{has_p:,}</div></div>""",unsafe_allow_html=True)

dff = df.copy()
if sel_genre!="All": dff=dff[dff["genre_str"].apply(lambda s:sel_genre in genre_list_from_str(s))]
if dff["year"].notna().any(): dff=dff[dff["year"].between(year_range[0],year_range[1])]
dff=dff[dff["vote_average"]>=min_rating]
dff=dff[dff["vote_count"]>=min_votes]

k1,k2,k3,k4,k5,k6=st.columns(6)
k1.metric("🎬 Movies",f"{len(dff):,}")
k2.metric("⭐ Avg Rating",f"{dff['vote_average'].mean():.2f}" if dff["vote_average"].notna().any() else "N/A")
k3.metric("🗳️ Avg Votes",f"{int(dff['vote_count'].mean()):,}" if dff["vote_count"].notna().any() else "N/A")
k4.metric("🎭 Genres",str(len(set(all_genres))))
k5.metric("📅 Years",f"{year_range[0]}–{year_range[1]}")
k6.metric("🌍 Languages",str(dff["original_language"].nunique()))
st.markdown("<br>",unsafe_allow_html=True)

tabs=st.tabs(["🏆 Top Movies","🎯 Recommender","📊 Analytics","💰 Box Office","🌍 Explore"])

with tabs[0]:
    banner("Top Rated Movies","Bayesian-weighted score · posters from dataset","🏆")
    col_l,col_r=st.columns([3,2])
    with col_l: top_n=st.slider("Number of top movies",5,50,20,key="topn")
    with col_r: sort_by=st.selectbox("Sort by",["Bayesian Score","Vote Average","Vote Count","Popularity"],key="sort_top")
    sort_col={"Bayesian Score":"score","Vote Average":"vote_average","Vote Count":"vote_count","Popularity":"popularity"}.get(sort_by,"score")
    mn_v=int(dff["vote_count"].quantile(0.75)) if len(dff)>10 else 0
    top_movies=dff[dff["vote_count"]>=mn_v].nlargest(top_n,sort_col).reset_index(drop=True)
    fig_top=go.Figure(go.Bar(y=top_movies["title"],x=top_movies[sort_col],orientation="h",
        marker=dict(color=top_movies["vote_average"],
            colorscale=[[0,"#7c3aed"],[0.35,"#ec4899"],[0.7,"#f59e0b"],[1,"#10b981"]],
            showscale=True,colorbar=dict(title="Rating",tickfont=dict(color="#e2d9f3"),len=0.6),
            line=dict(width=0)),
        text=top_movies[sort_col].round(2),textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score:%{x:.3f}<extra></extra>"))
    fig_top.update_layout(**PLOTLY_BASE,title=f"Top {top_n} Movies — {sort_by}",
        yaxis=dict(autorange="reversed",tickfont=dict(size=11)),
        xaxis=dict(showgrid=True,gridcolor="rgba(124,58,237,0.1)"),
        height=max(400,top_n*26),margin=dict(l=0,r=90,t=44,b=20))
    st.plotly_chart(fig_top,use_container_width=True)
    st.markdown("<hr>",unsafe_allow_html=True)
    banner("🖼️ Poster Cards","Posters from your TMDB dataset","🎬")
    for row_i in range(0,min(12,len(top_movies)),4):
        cols=st.columns(4)
        for col,(_,m) in zip(cols,top_movies.iloc[row_i:row_i+4].iterrows()):
            with col:
                movie_card(m["title"],m["vote_average"] if pd.notna(m["vote_average"]) else 0,
                    m.get("year"),m.get("overview",""),
                    genre_list_from_str(m.get("genre_str","")),
                    make_poster_url(m.get("poster_path",""),"w500"),
                    border="#7c3aed",vote_count=m.get("vote_count"))
    st.markdown("<br>",unsafe_allow_html=True)
    banner("🌟 Live Top Rated","Real-time from TMDB API","🌐")
    with st.spinner("Fetching..."):
        live_movies=fetch_live_top_rated()
    if live_movies:
        live_df=pd.DataFrame(live_movies).drop_duplicates("id").head(8)
        cols2=st.columns(4)
        for i,(_,m) in enumerate(live_df.iterrows()):
            with cols2[i%4]:
                movie_card(m.get("title","Unknown"),float(m.get("vote_average",0)),
                    str(m.get("release_date",""))[:4],m.get("overview",""),[],
                    make_poster_url(m.get("poster_path",""),"w500"),
                    border="#06b6d4",vote_count=m.get("vote_count"))

with tabs[1]:
    banner("Movie Recommendation System","TF-IDF + Cosine Similarity Engine","🎯")
    def get_recs(movie_title,n=10):
        key=movie_title.lower().strip()
        if key not in title_idx:
            matches=[t for t in title_idx.index if key in t]
            if not matches: return None
            key=matches[0]
        sims=cosine_similarity(tfidf_mat[title_idx[key]],tfidf_mat).flatten()
        top_i=sims.argsort()[::-1][1:n+1]
        recs=df_rec.iloc[top_i][["title","primary_genre","vote_average","score","year","overview","genre_str","poster_path"]].copy()
        recs["similarity"]=sims[top_i].round(4)
        return recs.reset_index(drop=True)
    inp_col,btn_col=st.columns([5,1])
    with inp_col: query=st.text_input("Type a movie title:",placeholder="e.g. The Dark Knight, Inception, Parasite...",key="rec_input")
    with btn_col:
        st.markdown("<br>",unsafe_allow_html=True)
        search=st.button("🔍 Search",use_container_width=True)
    num_recs=st.slider("Number of recommendations",5,20,10,key="nrecs")
    st.markdown('<p style="color:#7b7a9a;font-size:0.8em;margin:6px 0 4px;">⚡ Quick picks:</p>',unsafe_allow_html=True)
    demo_cols=st.columns(6)
    for dc,dm in zip(demo_cols,["The Dark Knight","Inception","Interstellar","The Godfather","Parasite","Avengers"]):
        if dc.button(dm,key=f"qp_{dm}"): query=dm; search=True
    if query:
        recs=get_recs(query,num_recs)
        if recs is None:
            st.error(f"❌ '{query}' not found. Try another title.")
        else:
            seed_rows=df_rec[df_rec["title"].str.lower()==query.lower().strip()]
            if len(seed_rows)==0:
                m2=[t for t in df_rec["title"].str.lower() if query.lower() in t]
                if m2: seed_rows=df_rec[df_rec["title"].str.lower()==m2[0]]
            if len(seed_rows):
                seed=seed_rows.iloc[0]
                sp=make_poster_url(seed.get("poster_path",""),"w185")
                p_html=(f"<img src='{sp}' style='width:90px;height:135px;object-fit:cover;border-radius:10px;flex-shrink:0;'>" if sp
                        else "<div style='width:90px;height:135px;background:#1a0a2e;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:2.5em;'>🎬</div>")
                try: yr_s=int(seed["year"]) if pd.notna(seed.get("year")) else "?"
                except: yr_s="?"
                st.markdown(f"""<div style="background:linear-gradient(135deg,rgba(124,58,237,0.1),rgba(236,72,153,0.05));
                    border:1px solid rgba(124,58,237,0.35);border-radius:18px;padding:22px;margin-bottom:22px;
                    display:flex;gap:20px;align-items:flex-start;">{p_html}
                    <div style="flex:1;"><h3 style="color:#c4b5fd;font-family:'Cinzel',serif;margin:0 0 6px;">🎬 {seed['title']}</h3>
                    <div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;">
                    <span style="color:#f59e0b;font-weight:700;">⭐ {float(seed['vote_average']):.1f}/10</span>
                    <span style="color:#7b7a9a;">📅 {yr_s}</span>
                    <span style="color:#7b7a9a;">🎭 {seed['primary_genre']}</span></div>
                    <p style="color:#9ca3af;font-size:0.84em;margin:0;line-height:1.5;">{str(seed.get('overview',''))[:220]}...</p>
                    </div></div>""",unsafe_allow_html=True)
            st.markdown(f'<h4 style="color:#a78bfa;margin-bottom:14px;">✨ {len(recs)} recommendations for <em style="color:#f9a8d4;">{query}</em></h4>',unsafe_allow_html=True)
            fig_sim=go.Figure(go.Bar(y=recs["title"],x=recs["similarity"],orientation="h",
                marker=dict(color=recs["similarity"],colorscale=[[0,"#7c3aed"],[0.5,"#ec4899"],[1,"#f59e0b"]],line=dict(width=0)),
                text=recs["similarity"].apply(lambda x:f"{x:.3f}"),textposition="outside",
                hovertemplate="<b>%{y}</b><br>Similarity:%{x:.4f}<extra></extra>"))
            fig_sim.update_layout(**PLOTLY_BASE,title="Cosine Similarity Scores",
                yaxis=dict(autorange="reversed"),height=min(500,50+num_recs*30),
                margin=dict(l=0,r=80,t=44,b=20))
            st.plotly_chart(fig_sim,use_container_width=True)
            card_cols=st.columns(4)
            for i,(_,rec) in enumerate(recs.head(8).iterrows()):
                with card_cols[i%4]:
                    movie_card(rec["title"],rec["vote_average"] if pd.notna(rec["vote_average"]) else 0,
                        rec.get("year"),rec.get("overview",""),
                        genre_list_from_str(rec.get("genre_str","")),
                        make_poster_url(rec.get("poster_path",""),"w500"),
                        border="#ec4899",sim=rec["similarity"])

with tabs[2]:
    banner("Analytics Dashboard","Comprehensive data insights","📊")
    c1,c2=st.columns(2)
    with c1:
        yc=dff[dff["year"].between(1950,2025)].groupby("year").size().reset_index(name="count")
        fig_yr=go.Figure(go.Scatter(x=yc["year"],y=yc["count"],fill="tozeroy",mode="lines",
            line=dict(color="#7c3aed",width=2.5),fillcolor="rgba(124,58,237,0.12)",
            hovertemplate="<b>%{x}</b><br>Movies:%{y:,}<extra></extra>"))
        fig_yr.update_layout(**PLOTLY_BASE,title="📅 Movies Per Year",height=320,margin=dict(l=20,r=20,t=44,b=20))
        st.plotly_chart(fig_yr,use_container_width=True)
    with c2:
        top12=dict(Counter(all_genres).most_common(12))
        fig_g=go.Figure(go.Bar(x=list(top12.keys()),y=list(top12.values()),
            marker=dict(color=list(top12.values()),colorscale=[[0,"#7c3aed"],[0.5,"#ec4899"],[1,"#f59e0b"]],line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y:,}<extra></extra>"))
        fig_g.update_layout(**PLOTLY_BASE,title="🎭 Top 12 Genres",xaxis_tickangle=-30,height=320,margin=dict(l=20,r=20,t=44,b=20))
        st.plotly_chart(fig_g,use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        rats=dff["vote_average"].dropna(); rats=rats[rats>0]
        fig_r=go.Figure(go.Histogram(x=rats,nbinsx=40,marker=dict(color="#7c3aed",opacity=0.85,line=dict(width=0))))
        if len(rats): fig_r.add_vline(x=float(rats.mean()),line_dash="dash",line_color="#f59e0b",
            annotation_text=f"Mean:{rats.mean():.2f}",annotation_font_color="#f59e0b")
        fig_r.update_layout(**PLOTLY_BASE,title="⭐ Rating Distribution",height=320,margin=dict(l=20,r=20,t=44,b=20))
        st.plotly_chart(fig_r,use_container_width=True)
    with c4:
        lc=dff["original_language"].value_counts().head(12)
        fig_l=go.Figure(go.Pie(labels=lc.index,values=lc.values,hole=0.45,
            marker=dict(colors=COLORS,line=dict(color="#05050f",width=2))))
        fig_l.update_layout(**PLOTLY_BASE,title="🌍 Languages",height=320,margin=dict(l=20,r=20,t=44,b=20))
        st.plotly_chart(fig_l,use_container_width=True)
    df_ex=dff[dff["decade"].notna()&(dff["decade"]>=1950)].copy()
    df_ex["gl"]=df_ex["genre_str"].apply(genre_list_from_str)
    df_ex=df_ex.explode("gl"); df_ex=df_ex[df_ex["gl"].str.strip()!=""]
    top12g=[g for g,_ in Counter(df_ex["gl"].dropna()).most_common(12)]
    if top12g:
        pivot=df_ex[df_ex["gl"].isin(top12g)].groupby(["decade","gl"]).size().unstack(fill_value=0)
        fig_h=px.imshow(pivot.T,color_continuous_scale=[[0,"#05050f"],[0.3,"#2d1b69"],[0.6,"#7c3aed"],[0.85,"#ec4899"],[1,"#f59e0b"]],
            aspect="auto",title="🔥 Genre × Decade Heatmap",text_auto=True)
        fig_h.update_layout(**PLOTLY_BASE,height=400,margin=dict(l=20,r=20,t=44,b=20))
        st.plotly_chart(fig_h,use_container_width=True)
    df_tr=dff.copy(); df_tr["gl"]=df_tr["genre_str"].apply(genre_list_from_str)
    df_tr=df_tr.explode("gl"); df_tr=df_tr[df_tr["gl"].notna()&(df_tr["gl"].str.strip()!="")]
    if dff["year"].notna().any(): df_tr=df_tr[df_tr["year"].between(1970,2025)]
    top8=[g for g,_ in Counter(df_tr["gl"]).most_common(8)]
    if top8:
        trend=df_tr[df_tr["gl"].isin(top8)].groupby(["year","gl"])["vote_average"].mean().reset_index()
        fig_t=px.line(trend,x="year",y="vote_average",color="gl",color_discrete_sequence=COLORS,
            title="📈 Rating by Genre Over Time",labels={"vote_average":"Rating","year":"Year","gl":"Genre"})
        fig_t.update_traces(line=dict(width=2))
        fig_t.update_layout(**PLOTLY_BASE,height=380,margin=dict(l=20,r=20,t=44,b=20))
        st.plotly_chart(fig_t,use_container_width=True)

with tabs[3]:
    banner("Box Office Analysis","Budget, revenue & profitability","💰")
    df_fin=dff[dff["budget"].notna()&dff["revenue"].notna()].copy()
    df_fin=df_fin[(df_fin["budget"]>10000)&(df_fin["revenue"]>10000)]
    if len(df_fin)<5:
        st.info("⚠️ Not enough financial data. Try removing filters.")
    else:
        ck1,ck2,ck3,ck4=st.columns(4)
        ck1.metric("💰 Avg Budget",f"${df_fin['budget'].mean()/1e6:.1f}M")
        ck2.metric("📦 Avg Revenue",f"${df_fin['revenue'].mean()/1e6:.1f}M")
        ck3.metric("💵 Avg Profit",f"${df_fin['profit'].mean()/1e6:.1f}M")
        ck4.metric("✅ Profitable",f"{(df_fin['profit']>0).mean()*100:.1f}%")
        sample=df_fin.sample(min(1500,len(df_fin)),random_state=42)
        mx=float(sample[["budget","revenue"]].max().max())
        fig_sc=px.scatter(sample,x="budget",y="revenue",color="profit",
            color_continuous_scale="RdYlGn",size="vote_count",size_max=24,
            hover_name="title",hover_data={"budget":":.2s","revenue":":.2s","profit":":.2s"},
            title="💰 Budget vs Revenue",log_x=True,log_y=True)
        fig_sc.add_trace(go.Scatter(x=[1e4,mx],y=[1e4,mx],mode="lines",
            line=dict(color="#ef4444",dash="dash",width=1.5),name="Break Even"))
        fig_sc.update_layout(**PLOTLY_BASE,height=460,margin=dict(l=20,r=20,t=44,b=20))
        st.plotly_chart(fig_sc,use_container_width=True)
        cp1,cp2=st.columns(2)
        with cp1:
            tp10=df_fin.nlargest(10,"profit")[["title","profit"]].copy(); tp10["pm"]=tp10["profit"]/1e6
            fig_pr=go.Figure(go.Bar(y=tp10["title"],x=tp10["pm"],orientation="h",
                marker=dict(color=tp10["pm"],colorscale=[[0,"#06b6d4"],[1,"#10b981"]],line=dict(width=0)),
                text=tp10["pm"].apply(lambda x:f"${x:.0f}M"),textposition="outside"))
            fig_pr.update_layout(**PLOTLY_BASE,title="Top 10 Profitable",
                yaxis=dict(autorange="reversed"),height=360,margin=dict(l=0,r=80,t=44,b=20))
            st.plotly_chart(fig_pr,use_container_width=True)
        with cp2:
            df_fin2=df_fin.copy(); df_fin2["gl"]=df_fin2["genre_str"].apply(genre_list_from_str)
            gp=df_fin2.explode("gl").groupby("gl")["profit"].mean().sort_values(ascending=False).head(10)/1e6
            fig_gp=go.Figure(go.Bar(x=gp.index,y=gp.values,
                marker=dict(color=gp.values,colorscale=[[0,"#7c3aed"],[1,"#f59e0b"]],line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>$%{y:.1f}M<extra></extra>"))
            fig_gp.update_layout(**PLOTLY_BASE,title="Avg Profit by Genre",
                xaxis_tickangle=-30,height=360,margin=dict(l=20,r=20,t=44,b=20))
            st.plotly_chart(fig_gp,use_container_width=True)

with tabs[4]:
    banner("Explore Movies","Search & browse with poster grid","🌍")
    ec1,ec2,ec3=st.columns(3)
    with ec1: search_q=st.text_input("Search title",placeholder="e.g. Batman...",key="exp_search")
    with ec2: exp_genre=st.selectbox("Filter genre",["All"]+all_genre_opts,key="exp_genre")
    with ec3: exp_sort=st.selectbox("Sort by",["score","vote_average","vote_count","popularity","year"],key="exp_sort")
    exp_df=dff.copy()
    if search_q: exp_df=exp_df[exp_df["title"].str.contains(search_q,case=False,na=False)]
    if exp_genre!="All": exp_df=exp_df[exp_df["genre_str"].apply(lambda s:exp_genre in genre_list_from_str(s))]
    exp_df=exp_df.sort_values(exp_sort,ascending=False).head(20)
    st.markdown(f'<p style="color:#7b7a9a;font-size:0.85em;">Showing {len(exp_df)} results</p>',unsafe_allow_html=True)
    for row_i in range(0,len(exp_df),5):
        cols=st.columns(5)
        for col,(_,m) in zip(cols,exp_df.iloc[row_i:row_i+5].iterrows()):
            with col:
                movie_card(m["title"],m["vote_average"] if pd.notna(m["vote_average"]) else 0,
                    m.get("year"),m.get("overview",""),
                    genre_list_from_str(m.get("genre_str","")),
                    make_poster_url(m.get("poster_path",""),"w185"),
                    border="#06b6d4",vote_count=m.get("vote_count"))

st.markdown("""<div style="background:linear-gradient(135deg,#0a0a1a,#0f0a1f);padding:24px 40px;
    border-radius:16px;border:1px solid rgba(124,58,237,0.2);text-align:center;margin-top:32px;">
    <p style="color:#7b7a9a;font-size:0.82em;margin:0;">🎬 TMDB Movie Universe ·
    <a href="https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies"
       style="color:#7c3aed;text-decoration:none;">Kaggle Dataset</a> ·
    <a href="https://www.themoviedb.org/" style="color:#ec4899;text-decoration:none;">TMDB API</a></p>
</div>""",unsafe_allow_html=True)
