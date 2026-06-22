import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="MBG Sentiment Dashboard",
    layout="wide"
)

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.main-title{
    font-size:64px;
    font-weight:700;
    color:#1C7651;
}

.sub-title{
    font-size:24px;
    color:#914CD5;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# LOAD CSV
df = pd.read_csv("datambgbersih_labeling.csv")

# METRICS
total_data = len(df)
netral = len(df[df["sentimen"] == "netral"])
negatif = len(df[df["sentimen"] == "negatif"])
positif = len(df[df["sentimen"] == "positif"])
accuracy = 93.29

st.markdown(
    '<div class="main-title">Dashboard Analisis Sentimen MBG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">IndoBERT Labeling + IndoBERT Embedding + Decision Tree</div>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["Overview", "Visualisasi"])

with tab1:
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Data", total_data)
    col2.metric("Netral", netral)
    col3.metric("Negatif", negatif)
    col4.metric("Positif", positif)
    col5.metric("Accuracy", f"{accuracy}%")

    colA, colB = st.columns(2)

    with colA:
        pie_fig = px.pie(df, names="sentimen", hole=0.6)
        st.plotly_chart(pie_fig, width="stretch")

    with colB:
        bar_fig = px.histogram(df, x="sentimen")
        st.plotly_chart(bar_fig, width="stretch")

    st.dataframe(
        df[["full_text", "tweet_processed", "sentimen"]].head(20),
        width="stretch"
    )

with tab2:
    st.subheader("Confusion Matrix")
    st.image("asset/cm.png")

    st.subheader("WordCloud")
    st.image("asset/wordcloud.png")

    st.subheader("Top Words")
    st.image("asset/topwords.png")

    st.subheader("Decision Tree")
    st.image("asset/decisiontree.png")
