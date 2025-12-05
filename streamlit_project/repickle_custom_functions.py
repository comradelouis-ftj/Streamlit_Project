import os
import joblib
import sys

from st_functions import cleanse, delete_stopwords, lemmatize, get_wordnet_pos

dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(dir, 'model')

module_custom = type(sys)('custom_module_functions')
module_custom.cleanse = cleanse
module_custom.delete_stopwords = delete_stopwords
module_custom.lemmatize = lemmatize
module_custom.get_wordnet_pos = get_wordnet_pos

original_module = sys.modules.get('__main__')
sys.modules['__main__'] = module_custom

try:
    clean = joblib.load(os.path.join(model_dir, 'punctuation_clean.pickle'))
    token = joblib.load(os.path.join(model_dir, 'tokenizer.pickle'))
    stopword = joblib.load(os.path.join(model_dir, 'stopwords.pickle'))
    lemma = joblib.load(os.path.join(model_dir, 'lemma.pickle'))
    vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pickle'))

    sys.modules['__main__'] = original_module
    sys.modules['__main__'] = sys.modules['st_functions']

    joblib.dump(clean, os.path.join(model_dir, 'punctuation_clean.pickle'))
    joblib.dump(token, os.path.join(model_dir, 'tokenizer.pickle'))
    joblib.dump(stopword, os.path.join(model_dir, 'stopwords.pickle'))
    joblib.dump(lemma, os.path.join(model_dir, 'lemma.pickle'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pickle'))

finally:
    if original_module:
        sys.modules['__main__'] = original_module