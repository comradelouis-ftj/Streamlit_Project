import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
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

def pos_tagging(tags):
  if tags.startswith('J'):
    return 'a'
  elif tags.startswith('V'):
    return 'v'
  elif tags.startswith('N'):
    return 'n'
  elif tags.startswith('R'):
    return 'r'
  else:
    return 'n'

def lemmatize(words):
    lemma = WordNetLemmatizer()
    tags = pos_tag(words)
    lemmatized = [lemma.lemmatize(word, pos_tagging(tag)) for word, tag in tags]
    result = ' '.join(lemmatized)
    #print(result)
    #print(tags)
    return result

def apply_cleanse(series):
    return series.apply(cleanse)

def apply_tokenize(series):
    return series.apply(nltk.word_tokenize)

def apply_delete_stopwords(series):
    return series.apply(delete_stopwords)

def apply_lemmatize(series):
    return series.apply(lemmatize)