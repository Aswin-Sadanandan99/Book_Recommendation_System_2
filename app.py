import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="📚 Book Recommendation System",
    layout="wide"
)
# -----------------------------
# Custom CSS for Styling
# -----------------------------
st.markdown("""
<style>

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* App background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff !important;
}

/* ALL TEXT DEFAULT */
.stApp, p, span, div, label {
    color: #eaeaea !important;
}

/* Titles */
h1, h2, h3 {
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Input labels (Enter description..., Select Category...) */
label {
    color: #f5f5f5 !important;
    font-weight: 600 !important;
}

/* Textarea & Text input */
textarea, input {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #ff4b4b !important;
}


/* Placeholder text */
textarea::placeholder,
input::placeholder {
    color: #bdbdbd !important;
    opacity: 1 !important;
}

/* Selectbox & Multiselect container */
div[data-baseweb="select"] > div {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
}

/* Selectbox & Multiselect selected text */
div[data-baseweb="select"] span {
    color: #ffffff !important;
}

/* Multiselect pills */
div[data-baseweb="tag"] {
    background-color: #ff4b4b !important;
    color: #ffffff !important;
}

/* Slider label & value */
.stSlider label, .stSlider span {
    color: #ffffff !important;
}

/* Button */
div.stButton > button {
    background-color: #ff4b4b;
    color: #ffffff !important;
    border-radius: 25px;
    padding: 0.6em 1.5em;
    font-weight: bold;
    border: none;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    background-color: #ff1f1f;
    transform: scale(1.05);
}

/* Success box */
.stSuccess {
    background-color: rgba(0, 255, 127, 0.15);
    border-radius: 10px;
    color: #ffffff !important;
}

/* Book cards */
.book-card {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 15px;
    margin: 10px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    text-align: center;
    color: #ffffff !important;
}

.book-title {
    font-size: 18px;
    font-weight: bold;
    margin-top: 10px;
    color: #ffffff !important;
}

/* Force dropdown list container */
div[role="listbox"],
ul[role="listbox"],
div[data-baseweb="popover"] {
    background-color: #232323 !important;
}

/* Each dropdown option */
div[role="option"],
li[role="option"],
div[data-baseweb="menu"] > div {
    background-color: #232323 !important;
    color: #ffffff !important;
}

/* Text inside options */
div[role="option"] span,
li[role="option"] span {
    color: #ffffff !important;
}

/* Hover / selected option */
div[role="option"]:hover,
li[role="option"]:hover,
div[data-baseweb="menu"] > div:hover {
    background-color: #ff4b4b !important;
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)


st.title("📚 Book Recommendation System")
st.write("Get book recommendations based on your description!")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/content/drive/MyDrive/Book_Recommendation_System_2/books.csv")   # <-- change to your file
    df['tags'] = df['tags'].fillna("").str.lower()
    return df

df = load_data()

# -----------------------------
# TF-IDF Model
# -----------------------------
@st.cache_resource
def build_model(tags):
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )
    tfidf_matrix = tfidf.fit_transform(tags)
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = build_model(df['tags'])

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_books_by_description(user_description, df, tfidf, tfidf_matrix):
    
    user_description = user_description.lower()
    
    user_vec = tfidf.transform([user_description])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)[0]
    
    top_indices = similarity_scores.argsort()[::-1]
    
    recommendation = df.iloc[top_indices][['isbn13', 'title','thumbnail', 'authors', 'average_rating','simple_categories',
    'anger','disgust','fear', 'joy', 'neutral', 'sadness', 'surprise']]
    
    return recommendation

# -----------------------------
# UI Inputs
# -----------------------------
st.subheader("🔍 Describe the book you want or Search by Category and Mood")

col1,col2,col3 = st.columns(3)
with col1:
    user_input = st.text_area("Enter description (genre, mood, author, story type, etc.)",
    placeholder="e.g. emotional love story, romance, drama, heartbreak...")
with col2:
    category = st.selectbox('Select Category',["All"]+[x for x in df['simple_categories'].unique()])
with col3:
    mood = st.multiselect('Select mood (Multi)',['anger','disgust','fear','joy','neutral','sadness','surprise'])

top_n = st.slider("Number of recommendations", 3, 10, 5)

# -----------------------------
# Button Action
# -----------------------------
if st.button("📖 Get Recommendations"):
    results = recommend_books_by_description(user_input, df, tfidf, tfidf_matrix)
    if category == "All":
        filter_df = results
    else:
        filter_df = results[results['simple_categories'].str.lower() == category.lower()]
    filter_df['mood_avg'] = filter_df[mood].mean(axis=1)
    filter_df = filter_df.sort_values(ascending=False,by='mood_avg')
    filter_df = filter_df[:top_n].reset_index()
    st.success("Here are some books you might like:")
    cols = st.columns(len(filter_df))
    cols = st.columns(len(filter_df))

    for i in range(len(filter_df)):
        with cols[i]:
            st.markdown(f"""
            <div class="book-card">
                <img src="{filter_df['thumbnail'][i]}" width="160" />
                <div class="book-title">{filter_df['title'][i]}</div>
                <div style="font-size:14px;">⭐ {filter_df['average_rating'][i]}</div>
            </div>
            """, unsafe_allow_html=True)
