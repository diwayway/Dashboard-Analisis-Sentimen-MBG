import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="MBG Sentiment Dashboard",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: #FFFFFF;
}

/* HEADER */
.main-title{
    font-size:64px;
    font-weight:700;
    line-height:72px;

    background: linear-gradient(
        90deg,
        #1C7651 0%,
        #914CD5 35%,
        #FB3679 70%,
        #7CC1F2 100%
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sub-title{
    font-size:24px;
    color:#1C7651;
    font-weight:500;
    margin-bottom:30px;
}

/* METRIC CARD */
.metric-card{
    padding:22px;
    border-radius:24px;
    color:white;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    min-height:140px;
    transition:0.3s;
}

.metric-card:hover{
    transform:translateY(-4px);
}

/* Gradient colors */
.card-purple{
    background: linear-gradient(135deg,#914CD5,#B889F8);
}

.card-green{
    background: linear-gradient(135deg,#1C7651,#2FAF77);
}

.card-pink{
    background: linear-gradient(135deg,#FB3679,#FF78A8);
}

.card-blue{
    background: linear-gradient(135deg,#7CC1F2,#A9DBFF);
}

.card-white{
    background: linear-gradient(135deg,#FFFFFF,#F8F8F8);
    border:1px solid #EEEEEE;
    color:#111;
}

/* Text */
.metric-title{
    font-size:18px;
    font-weight:500;
    opacity:0.95;
}

.metric-value{
    font-size:40px;
    font-weight:700;
    margin-top:15px;
}

/* Tabs */
.stTabs [data-baseweb="tab"]{
    font-size:18px;
    font-weight:600;
}

.stTabs [aria-selected="true"]{
    color:#914CD5;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("datambgbersih_labeling.csv")

# =====================================
# METRICS
# =====================================
total_data = len(df)
netral = len(df[df["sentimen"] == "netral"])
negatif = len(df[df["sentimen"] == "negatif"])
positif = len(df[df["sentimen"] == "positif"])
accuracy = 93.29

# =====================================
# HEADER
# =====================================
st.markdown(
    '<div class="main-title">Dashboard Analisis Sentimen MBG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">IndoBERT Labeling + IndoBERT Embedding + Decision Tree</div>',
    unsafe_allow_html=True
)

# =====================================
# TABS
# =====================================
tab1, tab2 = st.tabs(["Overview", "Visualisasi"])

# =====================================
# TAB 1
# =====================================
with tab1:

    col1, col2, col3, col4, col5 = st.columns(5)

    cards = [
        ("Total Data", total_data, "card-purple"),
        ("Netral", netral, "card-green"),
        ("Negatif", negatif, "card-pink"),
        ("Positif", positif, "card-blue"),
        ("Accuracy", f"{accuracy}%", "card-white")
    ]

    for col, (title, value, color) in zip(
        [col1, col2, col3, col4, col5], cards
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        pie_fig = px.pie(
            df,
            names="sentimen",
            hole=0.65
        )
        st.plotly_chart(pie_fig, width="stretch")

    with colB:
        bar_fig = px.histogram(
            df,
            x="sentimen"
        )
        st.plotly_chart(bar_fig, width="stretch")

    st.subheader("Dataset Preview")

    st.dataframe(
        df[["full_text", "tweet_processed", "sentimen"]].head(20),
        width="stretch"
    )

# =====================================
# TAB 2
# =====================================
with tab2:

    st.subheader("Confusion Matrix")
    st.image("asset/cm.png")

    st.subheader("WordCloud")
    st.image("asset/wordcloud.png")

    st.subheader("Top Words")
    st.image("asset/topwords.png")

    st.subheader("Decision Tree")
    st.image("asset/decisiontree.png")
