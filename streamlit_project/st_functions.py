import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import os
import joblib
import streamlit as st
from PIL import Image

import re
import emoji
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

def cleanse(words):
    words = words.strip(" ")
    words = emoji.demojize(words).replace(':', '')
    words = words.replace('_', ' ')
    words = words.lower()
    words = re.sub(r'[?|$|.|!_:"\')(-+,]', '', words)
    words = re.sub(r'\d+', '', words)
    words = re.sub(r"\b[a-zA-Z]\b", "", words)
    words = re.sub('\s+',' ', words)
    return words

def delete_stopwords(words):
    filter_words = stopwords.words('english')

    data = []
    def myFunc(x):
        if x in filter_words:
            return False
        else:
            return True
    fit = filter(myFunc, words)
    for x in fit:
        data.append(x)
    return data

def get_wordnet_pos(tag):
  if tag.startswith('J'):
    return 'a'
  elif tag.startswith('V'):
    return 'v'
  elif tag.startswith('N'):
    return 'n'
  elif tag.startswith('R'):
    return 'r'
  else:
    return 'n'

def lemmatize(words):
    lemma = WordNetLemmatizer()
    tags = pos_tag(words)
    lemmatized = [lemma.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tags]
    result = ' '.join(lemmatized)
    return result

def preprocess(data):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean = joblib.load(os.path.abspath(os.path.join(script_dir, 'model', 'punctuation_clean.pickle')))
    token = joblib.load(os.path.abspath(os.path.join(script_dir, 'model', 'tokenizer.pickle')))
    stopword = joblib.load(os.path.abspath(os.path.join(script_dir, 'model', 'stopwords.pickle')))
    lemma = joblib.load(os.path.abspath(os.path.join(script_dir, 'model', 'lemma.pickle')))
    vectorizer = joblib.load(os.path.abspath(os.path.join(script_dir, 'model', 'vectorizer.pickle')))
    
    data = data.apply(clean).apply(token).apply(stopword).apply(lemma)
    data = vectorizer.transform(data)
    return data

def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    path = os.path.abspath(os.path.join(script_dir, 'model', 'nlp_nn.pkl')) 

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    print(path)
    return joblib.load(path)

def categorize(num):
    num = np.argmax(num)
    if num == 2:
        return 'positive'
    elif num == 1:
        return 'neutral'
    else:
        return 'negative'

def predict(data):
    model = load_model()
    new_data = preprocess(data)
    prediction = model.predict(new_data)
    print(prediction)
    return categorize(prediction)

def agg_pos_neg_neu(item):
    if item == 0:
        return "Negative"
    elif item == 1:
        return "Neutral"
    elif item == 2:
        return "Positive"

def predict_df(data):
    model = load_model()
    new_data = preprocess(data)
    prediction = model.predict(new_data)
    return list(map(agg_pos_neg_neu,np.argmax(prediction, axis=1)))

def plot_pie(data):
    data_n = data.value_counts()
    colormap = cm.Wistia
    slice_colors = colormap(np.linspace(0, 1, len(data_n)))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        data_n.values,
        labels=data_n.index,
        autopct='%1.1f%%',
        textprops={'weight': 'bold'},
        shadow=True,
        startangle=90,
        colors=slice_colors
    )
    ax.add_artist(plt.Circle((0,0),0.30,fc='white'))
    plt.title('Distribusi Sentimen', fontsize=14, fontweight='bold')
    st.pyplot(fig)

@st.fragment
def load_image(file):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(script_dir, 'assets', file))
    if os.path.exists(path):
        return Image.open(path)
    else:
        print('error')
        return None