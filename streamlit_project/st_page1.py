import numpy as np
import pandas as pd
import streamlit as st

# from st_functions import predict, predict_df, load_image, cleanse, delete_stopwords, get_wordnet_pos, lemmatize

st.set_page_config(
    page_title='NLP Project - Home Page', 
    layout='wide'
)

st.markdown(
"""
<style>
div[data-testid="stSidebarNav"] {display: none;}
.stButton > button {
    text-align: center;
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>
""", 
unsafe_allow_html=True)

with st.sidebar:
    st.title('NLP Dengan Neural Network')
    st.divider()

    if st.button('**Profil Proyek**', width="stretch"):
        st.switch_page('st_page1.py')
    
    if st.button('**EDA/Insight**', width="stretch"):
        st.switch_page('pages/st_page2.py')
    
    if st.button('**Tahap Preprocessing**', width="stretch"):
        st.switch_page('pages/st_page3.py')

    if st.button('**Prediksi Sentimen**', width="stretch"):
        st.switch_page('pages/st_page4.py')

st.title('NLP Dengan Neural Network')
st.divider()

st.write("""
        <p style='font-size:18px; text-align:justify;'>
        NLP Project yang kami lakukan ini difokuskan
        pada analisis sentimen melalui platform media sosial. 
        Tujuan dari analisis ini adalah untuk mengklasifikasikan 
        pendapat dan pandangan masyarakat umum yang tertulis dalam 
        media sosial dimana mereka membahas mengenai topik-topik 
        tertentu yang menjadi perbincangan dalam kalangan masyarakat 
        dengan analisa berdasarkan sentimen pengguna media sosial 
        terhadap topik tersebut. </p>
        """, unsafe_allow_html=True
)

st.divider()
st.write("""
        <p style='font-size:18px; text-align:justify;'>
        Dalam topik <b>"Klasifikasi Teks pada Data Tweets Media Sosial"</b>,
        persoalan yang bisa diambil adalah: </p>
        <ul style="margin-left:18px; font-size:18px">
            <li>Mengidentifikasi pengaruh perbedaan topik pembahasan terhadap sentimen pengguna media sosial
        </li>
            <li>Perbedaan kata yang muncul pada kategori sentimen negatif, netral, dan positif yang berkontribusi pada tingginya akurasi model
        </li>
            <li>Tweet yang memiliki jumlah kata yang banyak cenderung memiliki sentimen non-netral dibanding kalimat yang pendek</li>
        </ul>
        <p style='font-size:18px; text-align:justify;'>
        Persoalan yang akan kami <i>highlight</i> adalah no.2 karena 
        sesuai dengan mengaplikasian AI dengan Sains Data untuk jenis
        kategori model Machine Learning klasifikasi sehingga kita bisa
        memisahkan kategori sentimen positif, netral, dan negatif berdasarkan
        hasil klasifikasi. Namun, untuk permasalahan no.1 bisa kami simpulkan
        juga berdasarkan hasil dari pemodelan NLP.</p>
        """, unsafe_allow_html=True)

st.divider()

st.write("""
        <p style='font-size:18px; text-align:center;'>\
        Dataset yang Digunakan: </p>""", unsafe_allow_html=True)

header = ['ID', 'Topic', 
'Sentiment', 'Tweet']
df = pd.read_csv(
    'https://raw.githubusercontent.com/comradeftj/Datsets/refs/heads/main/twitter_training.csv',
    names=header,
    header=None
)
df = df.head(10000)
st.dataframe(df)

st.write("""
        <ol style="margin-left:18px; font-size:18px">
            <li>ID: Berisi data unik dari tweet yang berguna untuk mencari sumber tweet
        </li>
            <li>Topic: Berisi nama entitas / topik yang akan menjadi objek sentimen tweet 
        </li>
            <li>Sentiment: label sentimen tweet dari entitas berupa label <i>"positive", 
         "negative", "netral", dan "irrelevant"
         </li>
            <li>Tweet: berisi isi teks tweet dari media sosial
         </li>
        </ol>""", unsafe_allow_html=True)