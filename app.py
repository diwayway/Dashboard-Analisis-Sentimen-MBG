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

/* CARD BASE STYLE */
.metric-card{
    padding:22px;
    border-radius:24px;
    color:white;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    min-height:140px;
    transition: 0.3s;
    display: block;
}

.metric-card:hover{
    transform:translateY(-6px);
}

.card-purple{ background: linear-gradient(135deg,#914CD5,#B889F8); }
.card-green{ background: linear-gradient(135deg,#1C7651,#2FAF77); }
.card-pink{ background: linear-gradient(135deg,#FB3679,#FF78A8); }
.card-blue{ background: linear-gradient(135deg,#7CC1F2,#A9DBFF); }

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

/* INTERACTIVE DROPDOWN CARD ESSENTIALS */
details.metric-card {
    list-style: none;
}
details.metric-card summary {
    list-style: none;
    outline: none;
}
details.metric-card summary::-webkit-details-marker {
    display: none; /* Sembunyikan panah bawaan browser */
}

.card-summary-layout {
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    width: 100%;
}

.card-caret {
    font-size: 20px;
    transition: transform 0.3s ease;
    opacity: 0.8;
}

details[open] .card-caret {
    transform: rotate(180deg); /* Panah berputar saat dropdown terbuka */
}

.card-dropdown-content {
    margin-top: 15px;
    padding-top: 15px;
    border-top: 1px solid rgba(255, 255, 255, 0.25);
    font-size: 14px;
    line-height: 1.6;
    text-align: justify;
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
accuracy = 94.38

# =====================================
# HEADER
# =====================================
st.markdown(
    '<div class="main-title">Dashboard Analimen Sentimen MBG</div>',
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

    # ==========================================
    # TOTAL DATA
    # ==========================================
    with col1:
        st.markdown(f"""
        <div class="metric-card card-purple">
            <div class="metric-title">Total Data</div>
            <div class="metric-value">{total_data}</div>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # NEGATIF
    # ==========================================
    with col2:
        st.markdown(f"""
        <details class="metric-card card-pink">
            <summary>
                <div class="card-summary-layout">
                    <div>
                        <div class="metric-title">Negatif</div>
                        <div class="metric-value">{negatif}</div>
                    </div>
                    <div class="card-caret">▼</div>
                </div>
            </summary>

            <div class="card-dropdown-content">
                Sentimen <b>negatif</b> menunjukkan opini masyarakat yang berisi kritik, penolakan,
                ketidakpuasan, atau tanggapan yang kurang mendukung terhadap Program
                Makan Bergizi Gratis (MBG).
            </div>
        </details>
        """, unsafe_allow_html=True)

    # ==========================================
    # NETRAL
    # ==========================================
    with col3:
        st.markdown(f"""
        <details class="metric-card card-green">
            <summary>
                <div class="card-summary-layout">
                    <div>
                        <div class="metric-title">Netral</div>
                        <div class="metric-value">{netral}</div>
                    </div>
                    <div class="card-caret">▼</div>
                </div>
            </summary>

            <div class="card-dropdown-content">
                Sentimen <b>netral</b> menunjukkan opini masyarakat yang tidak secara jelas
                mendukung maupun menolak Program Makan Bergizi Gratis (MBG). Tweet pada
                kategori ini umumnya berupa penyampaian informasi, berita, pertanyaan,
                atau komentar yang tidak mengandung kecenderungan sentimen positif
                maupun negatif.
            </div>
        </details>
        """, unsafe_allow_html=True)

    # ==========================================
    # POSITIF
    # ==========================================
    with col4:
        st.markdown(f"""
        <details class="metric-card card-blue">
            <summary>
                <div class="card-summary-layout">
                    <div>
                        <div class="metric-title">Positif</div>
                        <div class="metric-value">{positif}</div>
                    </div>
                    <div class="card-caret">▼</div>
                </div>
            </summary>

            <div class="card-dropdown-content">
                Sentimen <b>positif</b> menunjukkan opini masyarakat yang mendukung,
                mengapresiasi, atau memberikan tanggapan yang baik terhadap Program
                Makan Bergizi Gratis (MBG).
            </div>
        </details>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.subheader("Distribusi Sentimen Dataset Asli")

    colA, colB = st.columns(2)

    sentiment_count = (df["sentimen"].value_counts().reindex(["negatif", "netral", "positif"]))

    with colA:
        pie_fig = go.Figure(data=[go.Pie(
            labels=sentiment_count.index,
            values=sentiment_count.values,
            hole=0.65,
            marker=dict(
                colors=[
                    "#FF78A8",   # negatif
                    "#2FAF77",   # netral
                    "#A9DBFF"    # positif
                ]
            )
        )])

        st.plotly_chart(pie_fig, width="stretch")
    
    with colB:
        bar_fig = go.Figure(data=[go.Bar(
            x=sentiment_count.index,
            y=sentiment_count.values,
            marker_color=[
                "#FF78A8",
                "#2FAF77",
                "#A9DBFF"
            ]
        )])

        st.plotly_chart(bar_fig, width="stretch")

# =====================================
# TAB 2 - SMOTE
# =====================================
with tab2:

    st.subheader("Distribusi Data Sebelum dan Sesudah SMOTE")

    before_smote = {
        "Negatif":741,
        "Netral":1065,
        "Positif":436
    }

    after_smote = {
        "Negatif":1065,
        "Netral":1065,
        "Positif":1065
    }

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Sebelum SMOTE")

        before_fig = go.Figure(data=[go.Bar(
            x=list(before_smote.keys()),
            y=list(before_smote.values()),
            marker_color=[
                "#FF78A8",
                "#2FAF77",
                "#A9DBFF"
            ]
        )])

        st.plotly_chart(before_fig, width="stretch")

    with col2:
        st.markdown("### Sesudah SMOTE")

        after_fig = go.Figure(data=[go.Bar(
            x=list(after_smote.keys()),
            y=list(after_smote.values()),
            marker_color=[
                "#FF78A8",
                "#2FAF77",
                "#A9DBFF"
            ]
        )])

        st.plotly_chart(after_fig, width="stretch")

    st.markdown("""
    <div class="insight-box">
    Sebelum dilakukan SMOTE, data training hasil proses split data dengan rasio 70:30 menunjukkan distribusi kelas yang tidak seimbang, 
    yaitu sebanyak 1065 data pada kelas netral, 741 data pada kelas negatif, dan 436 data pada kelas positif. Kondisi ini dapat menyebabkan 
    model lebih cenderung mempelajari pola dari kelas mayoritas sehingga kurang optimal dalam mengenali kelas minoritas. Oleh karena itu, 
    metode SMOTE diterapkan hanya pada data training untuk menambah jumlah data pada kelas yang memiliki jumlah lebih sedikit agar distribusi 
    kelas menjadi lebih seimbang. Setelah proses SMOTE dilakukan, jumlah data pada masing-masing kelas menjadi 1065 data sehingga total data 
    training meningkat menjadi 3195 data. Dengan distribusi kelas yang seimbang, model dapat mempelajari pola dari setiap kelas secara lebih 
    proporsional serta mengurangi bias terhadap kelas mayoritas.

    Sebelum dilakukan SMOTE, data training hasil pembagian data dengan rasio 70:30 menunjukkan distribusi kelas yang tidak seimbang, yaitu sebanyak
    1065 data pada kelas netral, 741 data pada kelas negatif, dan 436 data pada kelas positif. Kondisi ini berpotensi menyebabkan model lebih cenderung 
    mempelajari pola dari kelas mayoritas sehingga kemampuan dalam mengklasifikasikan kelas minoritas menjadi kurang optimal. Oleh karena itu, metode SMOTE 
    diterapkan hanya pada data training untuk menghasilkan data sintetis pada kelas minoritas hingga jumlah setiap kelas menjadi seimbang. Setelah proses SMOTE dilakukan, 
    jumlah data pada masing-masing kelas menjadi 1065 data sehingga total data training meningkat menjadi 3195 data. Dengan distribusi kelas yang seimbang, model diharapkan 
    dapat mempelajari karakteristik setiap kelas secara lebih proporsional serta mengurangi kecenderungan bias terhadap kelas mayoritas.
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
        Berdasarkan confusion matrix di samping, model Decision Tree yang telah melalui proses SMOTE mampu melakukan klasifikasi dengan sangat 
        baik pada ketiga kelas sentimen. Nilai pada diagonal utama menunjukkan jumlah prediksi yang benar, yaitu sebanyak 301 data pada kelas negatif, 
        435 data pada kelas netral, dan 171 data pada kelas positif. Sementara itu, masih terdapat beberapa kesalahan klasifikasi. Pada kelas negatif, 
        sebanyak 14 data diprediksi sebagai netral dan 2 data diprediksi sebagai positif. Pada kelas netral, terdapat 8 data yang diprediksi sebagai 
        negatif dan 14 data yang diprediksi sebagai positif. Adapun pada kelas positif, terdapat 7 data yang diprediksi sebagai negatif dan 9 data yang diprediksi sebagai netral. 
        Hasil tersebut menunjukkan bahwa sebagian besar data berhasil diklasifikasikan dengan benar sehingga model memiliki kemampuan yang baik dalam 
        membedakan ketiga kelas sentimen.
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
        Berdasarkan visualisasi wordcloud yang ditampilkan, terlihat beberapa kata yang paling sering muncul dalam data 
        tweet terkait Program Makan Bergizi Gratis (MBG), seperti “makan”, “gizi”, “gratis”, “mbg”, dan “program”. 
        Ukuran kata yang lebih besar menunjukkan frekuensi kemunculan yang lebih tinggi dalam dataset. Visualisasi 
        ini memberikan gambaran umum mengenai topik utama yang banyak dibahas masyarakat terkait program MBG.
        </div>
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
        Berdasarkan grafik frekuensi kata di samping, terlihat bahwa kata “mbg”, “gizi”, “makan”, “gratis”, dan “program” 
        menjadi kata yang paling sering muncul dalam dataset. Visualisasi ini membantu memperlihatkan kata-kata dominan 
        yang sering digunakan dalam opini masyarakat terkait Program Makan Bergizi Gratis (MBG).
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
        Berdasarkan visualisasi <b>Decision Tree</b> yang ditampilkan, proses klasifikasi dimulai dari <b>node akar (root node)</b> sebagai titik 
        awal pengambilan keputusan. Pada node tersebut terdapat aturan <b>x[603] ≤ 0.698</b>, yang menunjukkan bahwa model menggunakan <b>fitur ke-603</b> 
        hasil <b>feature extraction</b> menggunakan <b>IndoBERT Embedding</b> sebagai dasar pemisahan data. Nilai <b>entropy</b> sebesar <b>1.585</b> menunjukkan 
        bahwa distribusi kelas pada node awal masih bercampur sehingga proses pemisahan data masih perlu dilakukan pada percabangan berikutnya. Sementara itu, nilai 
        <b>samples</b> sebesar <b>3195</b> menunjukkan jumlah data latih yang digunakan dalam proses pembentukan Decision Tree setelah melalui proses <b>oversampling</b> 
        menggunakan metode <b>SMOTE</b>. Jumlah tersebut lebih besar dibandingkan jumlah data latih awal sebanyak <b>2242</b> data karena kelas minoritas telah 
        ditambahkan hingga seluruh kelas memiliki jumlah yang sama. Hal tersebut ditunjukkan oleh nilai <b>value = [1065, 1065, 1065]</b>, yang menyatakan bahwa 
        masing-masing kelas sentimen, yaitu <b>negatif</b>, <b>netral</b>, dan <b>positif</b>, masing-masing memiliki <b>1065</b> data setelah proses SMOTE. Meskipun 
        distribusi ketiga kelas telah seimbang, nilai <b>class = Negatif</b> ditampilkan karena kelas negatif merupakan kelas pertama berdasarkan urutan label yang digunakan 
        oleh model. Selanjutnya, setiap percabangan akan memisahkan data berdasarkan nilai fitur tertentu hingga mencapai <b>node akhir (leaf node)</b>, yang digunakan sebagai 
        dasar dalam menentukan hasil klasifikasi sentimen.
        </div>
        """, unsafe_allow_html=True)
