import os
import joblib
import pandas as pd

from model.functions import apply_cleanse, apply_tokenize, apply_delete_stopwords, apply_lemmatize
from sklearn.feature_extraction.text import TfidfVectorizer

path = 'D:/ai_ml/aiml_env/model/nlp_nn.joblib'
model = joblib.load(path)

word = pd.DataFrame({
    'Tweet': ['This is a good game']
})
print(model.predict(word['Tweet'])) 