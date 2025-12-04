import numpy as np
import pandas as pd
from PIL import Image

from model.functions import apply_cleanse, apply_tokenize, apply_delete_stopwords, apply_lemmatize
from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st
import joblib
import os

def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    path = os.path.abspath(os.path.join(script_dir, '..',  'model', 'nlp_nn.joblib')) 
    # this makes sure that we take the parent folder, and always takes the model folder from the parent folder

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    print(path)
    return joblib.load(path)

def load_image(file):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(script_dir, '..',  'assets', file))
    if os.path.exists(path):
        return Image.open(path)
    else:
        print('error')
        return None

st.set_page_config(layout='wide')

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

    if st.button('**Try the Model!**', width="stretch"):
        st.switch_page('pages/st_page4.py')

# Main Page

st.title('The NLP Model')
st.divider()

st.subheader('Try it yourself!')
tweet = st.text_area('Insert tweet here', 'tweet....', height=150)
st.write(f'Your text: {tweet}')
'''
model = load_model()
if st.button('Predict', width='stretch'):
    word = pd.DataFrame({
        'Tweet': [f'{tweet}']
    })
    st.write(f'**Prediction: {model.predict(word['Tweet'])}**')
'''

st.divider()
st.title('Model NLP (Natural Language Processing)')
st.divider()
st.write('Metode yang digunakan untuk membangun model NLP proyek ini adalah metode neural network, atau lebih tepatnya Multi-Layer Perceptron (MLP). MLP merupakan jenis neural network yang terdiri dari tiga bagian, yakni Input, Output, dan Hidden Layer.')
st.write('Adapun struktur Neural Network yang digunakan adalah:')

col1, col2, col3 = st.columns([1.5, 4, 1.5])
with col2:
    image = load_image('struktur_nn.png')
    st.image(image, caption='Struktur Neural Network', width=400)

col1, col2 = st.columns(2)
with col1:
    image_relu = load_image('relu.png')
    st.image(image_relu, caption='Fungsi ReLU (Rectified Linear Unit)')
with col2:
    image_softmax = load_image('softmax.png')
    st.image(image_softmax, caption='Fungsi Softmax')

col1, col2 = st.columns(2)
with col1:
    st.write('''Fungsi ReLU digunakan untuk mematikan atau menghapus pengaruh dari nilai yang bernilai di bawah 0. Disini, nilai bernilai di bawah atau sama dengan 0
             merepresentasikan kata yang tidak memiliki pengaruh atau bobot signifikan berdasarkan hasil vectorization menggunaan TF-IDF. Dalam kasus ini, fungsi
             ReLU digunakan pada hidden layer pertama untuk mematikan kata yang tidak memiliki signifikansi''')
with col2:
    st.write('''Fungsi Softmax merupakan fungsi yang seringkali digunakan untuk klasifikasi multi-kelas, karena luaran (output) yang berupa probabilitas suatu nilai
             merupakan bagian dari kelas tertentu. Dalam kasus proyek ini, fungsi Softmax digunakan pada hidden layer kedua dan pada output layer''')

st.divider()
st.write('Dengan struktur deikian, dan tahap preprocessing yang telah dijelaskan pada halaman sebelumnya, model yang dibentuk memiliki hasil evaluasi berikut:')
col1, col2, col3 = st.columns([1.5, 4, 1.5])
with col2:
    image_ev = load_image('evaluasi.png')
    st.image(image_ev, caption='Hasil Evaluasi Model', width=400)
st.write('Pada seluruh kategori, nilai precision, recall, f1-score, dan akurasi memiliki nilai >=90%. Ini berarti model dapat mengidentifikasi kalimat dengan sentien positif, negatif, dan netral secara andal dengan akurasi di atas 90%')