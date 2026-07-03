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

    # =====================================
    # CONFUSION MATRIX
    # =====================================
    st.subheader("Confusion Matrix")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image("asset/cm.png", width=800)

    with col2:
        st.markdown("""
        <div class="insight-box">
        <p>
        Berdasarkan confusion matrix di atas, model Decision Tree yang telah melalui proses SMOTE menunjukkan hasil klasifikasi yang cukup baik pada ketiga kelas sentimen. Nilai pada diagonal utama menunjukkan jumlah prediksi yang benar, yaitu sebanyak 302 data untuk sentimen negatif, 437 data untuk sentimen netral, dan 173 data untuk sentimen positif. Sementara itu, nilai di luar diagonal menunjukkan kesalahan klasifikasi, seperti data negatif yang diprediksi sebagai netral sebanyak 11 data dan positif sebanyak 3 data. Pada kelas netral, terdapat 10 data yang diprediksi sebagai negatif dan 11 data sebagai positif. Sedangkan pada kelas positif, terdapat masing-masing 7 data yang salah diprediksi sebagai negatif dan netral. Hasil tersebut menunjukkan bahwa model mampu mengenali pola sentimen dengan cukup baik, terutama pada kelas netral yang memiliki jumlah prediksi benar paling tinggi, meskipun masih terdapat beberapa kesalahan klasifikasi antar kelas.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================
    # WORDCLOUD
    # =====================================
    st.subheader("WordCloud")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image("asset/wordcloud.png", width=800)

    with col2:
        st.markdown("""
        <div class="insight-box">
        <p>
        Berdasarkan visualisasi wordcloud di atas, terlihat beberapa kata yang paling sering muncul dalam data tweet terkait Program Makan Bergizi Gratis (MBG), seperti “gizi”, “gratis”, “mbg”, “makan”, dan “program”. Ukuran kata yang lebih besar menunjukkan frekuensi kemunculan yang lebih tinggi dalam dataset. Visualisasi ini memberikan gambaran umum mengenai topik utama yang banyak dibahas masyarakat terkait program MBG.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================
    # TOP WORDS
    # =====================================
    st.subheader("Top Words")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image("asset/topwords.png", width=800)

    with col2:
        st.markdown("""
        <div class="insight-box">
        <p>
        Berdasarkan grafik frekuensi kata di atas, terlihat bahwa kata “mbg”, “gizi”, “makan”, “gratis”, dan “program” menjadi kata yang paling sering muncul dalam dataset. Visualisasi ini membantu memperlihatkan kata-kata dominan yang sering digunakan dalam opini masyarakat terkait program MBG.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================
    # DECISION TREE
    # =====================================
    st.subheader("Decision Tree")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image("asset/decisiontree.png", width=1000)

    with col2:
        st.markdown("""
        <div class="insight-box">
        <p>
        Berdasarkan visualisasi Decision Tree di atas, proses klasifikasi dimulai dari node paling atas sebagai titik awal pengambilan keputusan. Pada node tersebut terdapat aturan x[603] <= 0.625, yang menunjukkan bahwa model menggunakan fitur ke-603 dari hasil word embedding IndoBERT untuk membagi data. Nilai entropy sebesar 1.585 menunjukkan bahwa data pada node awal masih memiliki tingkat campuran yang tinggi. Semakin kecil nilai entropy, maka data pada node tersebut semakin seragam dan keputusan model menjadi lebih jelas. Nilai samples sebesar 3204 menunjukkan jumlah data yang diproses, sedangkan nilai value sebesar [1068, 1068, 1068] menunjukkan jumlah data pada masing-masing kelas sentimen, yaitu negatif, netral, dan positif yang telah diseimbangkan menggunakan SMOTE. Meskipun jumlah data pada ketiga kelas sama, nilai class ditampilkan sebagai negatif karena urutan kelas dimulai dari negatif, netral, lalu positif, sehingga kelas pertama dipilih sebagai kelas dominan. Percabangan pada bagian bawah merupakan lanjutan dari proses pembagian data sebelumnya dapat menentukan hasil klasifikasi sentimen akhir dengan lebih spesifik.
        </p>
        </div>
        """, unsafe_allow_html=True)
