import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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
    color:#111111;
    font-weight:500;
    margin-bottom:30px;
}

/* CARD */
.metric-card{
    padding:22px;
    border-radius:24px;
    color:white;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    min-height:140px;
    transition:0.3s;
}

.metric-card:hover{
    transform:translateY(-6px);
}

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

.metric-title{
    font-size:18px;
    font-weight:500;
}

.metric-value{
    font-size:40px;
    font-weight:700;
    margin-top:15px;
}

/* INSIGHT BOX */
.insight-box{
    background:#F8F8F8;
    padding:20px;
    border-radius:20px;
    border:1px solid #EEEEEE;
    margin-top:10px;
    color:#111;
    line-height:1.7;
    text-align:justify;
}

/* TABS */
.stTabs [data-baseweb="tab-list"]{
    gap: 20px;
}

.stTabs [data-baseweb="tab"]{
    font-size:22px;
    font-weight:700;
    padding:16px 28px;
    height:70px;
    border-radius:14px 14px 0px 0px;
}

.stTabs [aria-selected="true"]{
    color:#FB3679;
    border-bottom:3px solid #FB3679;
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
accuracy = 94.90

# =====================================
# HEADER
# =====================================
st.markdown(
    '<div class="main-title">Dashboard Analisis Sentimen MBG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">IndoBERT Labeling + IndoBERT Embedding + SMOTE + Decision Tree</div>',
    unsafe_allow_html=True
)

# =====================================
# TABS
# =====================================
tab1, tab2, tab3 = st.tabs([
    "Dataset",
    "Balancing (SMOTE)",
    "Visualisasi"
])

# =====================================
# TAB 1 - DATASET
# =====================================
with tab1:

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    cards = [
        ("Total Data", total_data, "card-purple"),
        ("Netral", netral, "card-green"),
        ("Negatif", negatif, "card-pink"),
        ("Positif", positif, "card-blue")
    ]

    for col, (title, value, color) in zip(
        [col1, col2, col3, col4], cards
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Distribusi Sentimen Dataset Asli")

    colA, colB = st.columns(2)

    sentiment_count = df["sentimen"].value_counts()

    with colA:
        pie_fig = go.Figure(data=[go.Pie(
            labels=sentiment_count.index,
            values=sentiment_count.values,
            hole=0.65
        )])
        st.plotly_chart(pie_fig, use_container_width=True)

    with colB:
        bar_fig = go.Figure(data=[go.Bar(
            x=sentiment_count.index,
            y=sentiment_count.values
        )])
        st.plotly_chart(bar_fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(
        df[["full_text", "tweet_processed", "sentimen"]].head(20),
        use_container_width=True
    )

# =====================================
# TAB 2 - SMOTE
# =====================================
with tab2:

    st.subheader("Distribusi Data Sebelum dan Sesudah SMOTE")

    before_smote = {
        "Netral": 1526,
        "Negatif": 1054,
        "Positif": 623
    }

    after_smote = {
        "Netral": 1068,
        "Negatif": 1068,
        "Positif": 1068
    }

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Sebelum SMOTE")
        st.bar_chart(before_smote)

    with col2:
        st.markdown("### Sesudah SMOTE")
        st.bar_chart(after_smote)

    st.markdown("""
    <div class="insight-box">
    Sebelum dilakukan SMOTE, distribusi data menunjukkan ketidakseimbangan antar kelas sentimen, di mana kelas netral mendominasi data, sedangkan kelas positif memiliki jumlah paling sedikit. Kondisi ini dapat menyebabkan model lebih cenderung mempelajari pola dari kelas mayoritas dan kurang optimal dalam mengenali kelas minoritas. Oleh karena itu, SMOTE digunakan pada data training untuk menambah data pada kelas yang jumlahnya lebih sedikit agar distribusi data menjadi lebih seimbang. Setelah proses SMOTE dilakukan, jumlah data pada masing-masing kelas menjadi 1068 data, sehingga distribusi data lebih seimbang dan model dapat belajar dengan lebih baik serta mengurangi bias terhadap kelas mayoritas.
    </div>
    """, unsafe_allow_html=True)

# =====================================
# TAB 3 - VISUALISASI
# =====================================
with tab3:

    st.subheader("Model Performance")

    st.markdown(f"""
    <div class="metric-card card-white">
        <div class="metric-title">Final Accuracy</div>
        <div class="metric-value">{accuracy}%</div>
    </div>
    """, unsafe_allow_html=True)

    visualizations = [
        ("Confusion Matrix", "asset/cm.png", """Berdasarkan confusion matrix di atas, model Decision Tree yang telah melalui proses SMOTE menunjukkan hasil klasifikasi yang cukup baik pada ketiga kelas sentimen. Nilai pada diagonal utama menunjukkan jumlah prediksi yang benar, yaitu sebanyak 302 data untuk sentimen negatif, 437 data untuk sentimen netral, dan 173 data untuk sentimen positif."""),
        
        ("WordCloud", "asset/wordcloud.png", """Berdasarkan visualisasi wordcloud di atas, terlihat beberapa kata yang paling sering muncul dalam data tweet terkait Program Makan Bergizi Gratis (MBG), seperti “gizi”, “gratis”, “mbg”, “makan”, dan “program”."""),
        
        ("Top Words", "asset/topwords.png", """Berdasarkan grafik frekuensi kata di atas, terlihat bahwa kata “mbg”, “gizi”, “makan”, “gratis”, dan “program” menjadi kata yang paling sering muncul dalam dataset."""),
        
        ("Decision Tree", "asset/decisiontree.png", """Berdasarkan visualisasi Decision Tree di atas, proses klasifikasi dimulai dari node paling atas sebagai titik awal pengambilan keputusan. Pada node tersebut terdapat aturan x[603] <= 0.625 yang menunjukkan bahwa model menggunakan fitur ke-603 dari hasil word embedding IndoBERT.""")
    ]

    for title, image_path, desc in visualizations:
        st.subheader(title)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image_path, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div class="insight-box">
            {desc}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
