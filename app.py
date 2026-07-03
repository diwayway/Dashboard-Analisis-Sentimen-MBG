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
    margin-top:20px;
    color:#111;
    line-height:1.8;
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
            hole=0.65,
            marker=dict(
                colors=["#2FAF77", "#FF78A8", "#A9DBFF"]
            )
        )])

        st.plotly_chart(pie_fig, width="stretch")

    with colB:
        bar_fig = go.Figure(data=[go.Bar(
            x=sentiment_count.index,
            y=sentiment_count.values,
            marker_color=["#2FAF77", "#FF78A8", "#A9DBFF"]
        )])

        st.plotly_chart(bar_fig, width="stretch")

    st.subheader("Dataset Preview")

    st.dataframe(
        df[["full_text", "tweet_processed", "sentimen"]].head(20),
        width="stretch"
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

        before_fig = go.Figure(data=[go.Bar(
            x=list(before_smote.keys()),
            y=list(before_smote.values()),
            marker_color=["#2FAF77", "#FF78A8", "#A9DBFF"]
        )])

        st.plotly_chart(before_fig, width="stretch")

    with col2:
        st.markdown("### Sesudah SMOTE")

        after_fig = go.Figure(data=[go.Bar(
            x=list(after_smote.keys()),
            y=list(after_smote.values()),
            marker_color=["#2FAF77", "#FF78A8", "#A9DBFF"]
        )])

        st.plotly_chart(after_fig, width="stretch")

    st.markdown("""
    <div class="insight-box">
    Sebelum dilakukan SMOTE, distribusi data menunjukkan ketidakseimbangan antar kelas sentimen, di mana kelas netral mendominasi data, 
    sedangkan kelas positif memiliki jumlah paling sedikit. Kondisi ini dapat menyebabkan model lebih cenderung mempelajari pola dari kelas 
    mayoritas dan kurang optimal dalam mengenali kelas minoritas. Oleh karena itu, SMOTE digunakan pada data training untuk menambah data 
    pada kelas yang jumlahnya lebih sedikit agar distribusi data menjadi lebih seimbang. Setelah proses SMOTE dilakukan, jumlah data pada 
    masing-masing kelas menjadi 1068 data, sehingga distribusi data lebih seimbang dan model dapat belajar dengan lebih baik serta mengurangi 
    bias terhadap kelas mayoritas.
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

    st.markdown("<br>", unsafe_allow_html=True)

    # CONFUSION MATRIX
    st.subheader("Confusion Matrix")
    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.image("asset/cm.png", width=800)

    with col2:
        st.markdown("""
        <div class="insight-box">
        Berdasarkan confusion matrix di samping, model Decision Tree yang telah melalui proses SMOTE 
        menunjukkan hasil klasifikasi yang cukup baik pada ketiga kelas sentimen. Nilai pada diagonal 
        utama menunjukkan jumlah prediksi yang benar, yaitu sebanyak 302 data untuk sentimen negatif, 
        437 data untuk sentimen netral, dan 173 data untuk sentimen positif. Sementara itu, nilai di 
        luar diagonal menunjukkan kesalahan klasifikasi, seperti data negatif yang diprediksi sebagai 
        netral sebanyak 11 data dan positif sebanyak 3 data. Pada kelas netral, terdapat 10 data yang 
        diprediksi sebagai negatif dan 11 data sebagai positif. Sedangkan pada kelas positif, terdapat 
        masing-masing 7 data yang salah diprediksi sebagai negatif dan netral. Hasil tersebut menunjukkan 
        bahwa model mampu mengenali pola sentimen dengan cukup baik, terutama pada kelas netral yang memiliki 
        jumlah prediksi benar paling tinggi, meskipun masih terdapat beberapa kesalahan klasifikasi antar kelas.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # WORDCLOUD
    st.subheader("WordCloud")
    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.image("asset/wordcloud.png", width=800)

    with col2:
        st.markdown("""
        <div class="insight-box">
        Berdasarkan visualisasi wordcloud di atas, terlihat beberapa kata yang paling sering muncul dalam data 
        tweet terkait Program Makan Bergizi Gratis (MBG), seperti “gizi”, “gratis”, “mbg”, “makan”, dan “program”. 
        Ukuran kata yang lebih besar menunjukkan frekuensi kemunculan yang lebih tinggi dalam dataset. Visualisasi 
        ini memberikan gambaran umum mengenai topik utama yang banyak dibahas masyarakat terkait program MBG.
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TOP WORDS
    st.subheader("Top Words")
    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.image("asset/topwords.png", width=800)

    with col2:
        st.markdown("""
        <div class="insight-box">
        Berdasarkan grafik frekuensi kata di atas, terlihat bahwa kata “mbg”, “gizi”, “makan”, “gratis”, dan “program” 
        menjadi kata yang paling sering muncul dalam dataset. Visualisasi ini membantu memperlihatkan kata-kata dominan 
        yang sering digunakan dalam opini masyarakat terkait program MBG.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # DECISION TREE
    st.subheader("Decision Tree")
    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.image("asset/decisiontree.png", width=1000)

    with col2:
        st.markdown("""
        <div class="insight-box">
        Berdasarkan visualisasi Decision Tree di atas, proses klasifikasi dimulai dari node paling atas sebagai titik awal 
        pengambilan keputusan. Pada node tersebut terdapat aturan **x[603] <= 0.625**, yang menunjukkan bahwa model menggunakan 
        fitur ke-603 dari hasil *word embedding* IndoBERT untuk membagi data. Nilai *entropy* sebesar **1.585** menunjukkan bahwa 
        data pada node awal masih memiliki tingkat campuran yang tinggi. Semakin kecil nilai *entropy*, maka data pada node tersebut 
        semakin seragam dan keputusan model menjadi lebih jelas. Nilai *samples* sebesar **3204** menunjukkan jumlah data yang diproses, 
        sedangkan nilai *value* sebesar **[1068, 1068, 1068]** menunjukkan jumlah data pada masing-masing kelas sentimen, yaitu negatif, 
        netral, dan positif yang telah diseimbangkan menggunakan SMOTE. Meskipun jumlah data pada ketiga kelas sama, nilai *class* ditampilkan 
        sebagai **negatif** karena urutan kelas dimulai dari negatif, netral, lalu positif, sehingga kelas pertama dipilih sebagai kelas dominan. 
        Percabangan pada bagian bawah merupakan lanjutan dari proses pembagian data sebelumnya dapat menentukan hasil klasifikasi sentimen akhir 
        dengan lebih spesifik.
        </div>
        """, unsafe_allow_html=True)
