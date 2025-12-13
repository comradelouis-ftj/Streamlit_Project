import numpy as np
import pandas as pd
import os
import sys

dir = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(dir)
sys.path.append(parent)

from st_functions import predict, load_image, predict_df, plot_pie
import streamlit as st

st.set_page_config(
    page_title='NLP Project - Model Preview', 
    layout='wide'
)

# CSS Markdown
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

# Sidebar
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

# Main Page

st.title('Model NLP - Klasifikasi Sentimen Tweets')
st.divider()

st.subheader('Try it yourself!')
st.divider()
col_write, col_up = st.columns(2)

with col_write:
    with st.container(height=400):
        st.write('**Insert tweet here:**')
        tweet = st.text_area('make sure to use english, example tweet: **I like this game!**', 'tweet....', height=150)
        st.write(f'Your text: {tweet}')

        if st.button('Predict', width='stretch'):
            word = pd.DataFrame({
                'Tweet': [f'{tweet}']
            })
            result = predict(word['Tweet'])
            st.write(f'**Prediction: {result}**')

with col_up:
    with st.container(height=400):
        st.write('**Try it with files!**')
        file_csv = st.file_uploader("Insert .csv file here, make sure your file has a 'Tweet' column", type=['csv'])
        if file_csv is not None:
            st.write("Successful upload!")
            file = pd.read_csv(file_csv)
            file['class'] = predict_df(file['Tweet'])

            st.divider()
            plot_pie(file['class'])
            st.dataframe(file.head(5))


st.divider()
st.title('Model NLP (Natural Language Processing)')
st.divider()
st.write('Metode yang digunakan untuk membangun model NLP proyek ini adalah metode neural network, atau lebih tepatnya Multi-Layer Perceptron (MLP). MLP merupakan jenis neural network yang terdiri dari tiga bagian, yakni Input, Output, dan Hidden Layer.')
st.write('Adapun struktur Neural Network yang digunakan adalah:')

col1, col2, col3 = st.columns([1.5, 4, 1.5])
with col2:
    image = load_image('struktur_nn.png')
    st.image(image, caption='Struktur Neural Network', width=400)

col11, col12 = st.columns(2)
with col11:
    image_relu = load_image('relu.png')
    st.image(image_relu, caption='Fungsi ReLU (Rectified Linear Unit)')
with col12:
    image_softmax = load_image('softmax.png')
    st.image(image_softmax, caption='Fungsi Softmax')

col21, col22 = st.columns(2)
with col21:
    st.write('''Fungsi ReLU digunakan untuk mematikan atau menghapus pengaruh dari nilai yang bernilai di bawah 0. Disini, nilai bernilai di bawah atau sama dengan 0
             merepresentasikan kata yang tidak memiliki pengaruh atau bobot signifikan berdasarkan hasil vectorization menggunaan TF-IDF. Dalam kasus ini, fungsi
             ReLU digunakan pada hidden layer pertama untuk mematikan kata yang tidak memiliki signifikansi''')
with col22:
    st.write('''Fungsi Softmax merupakan fungsi yang seringkali digunakan untuk klasifikasi multi-kelas, karena luaran (output) yang berupa probabilitas suatu nilai
             merupakan bagian dari kelas tertentu. Dalam kasus proyek ini, fungsi Softmax digunakan pada hidden layer kedua dan pada output layer''')

st.divider()
st.write('Dengan struktur deikian, dan tahap preprocessing yang telah dijelaskan pada halaman sebelumnya, model yang dibentuk memiliki hasil evaluasi berikut:')
col1, col2, col3 = st.columns([1.5, 4, 1.5])
with col2:
    image_ev = load_image('evaluasi.png')
    st.image(image_ev, caption='Hasil Evaluasi Model', width=400)
st.write('Pada seluruh kategori, nilai precision, recall, f1-score, dan akurasi memiliki nilai >=90%. Ini berarti model dapat mengidentifikasi kalimat dengan sentien positif, negatif, dan netral secara andal dengan akurasi di atas 90%')