# https://medium.com/xeneta/boosting-sales-with-machine-learning-fbcf2e618be3
# https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

nltk.download("stopwords")
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)

DICTIONARY = {}

def add(guid, ingestionText):
  word_arr = clean(ingestionText)
  stopped = remove_stop_words(word_arr)
  stemmed =  stem_words(stopped)
  # transformed = vectorizer.fit_transform(stemmed)
  # transformed = transformed.toarray()
  # vocab = vectorizer.get_feature_names()
  # distribution = np.sum(transformed, axis=0)
  # for tag, count in zip(vocab, distribution):
  #   print count, tag
  # print transformed.shape
  for word in stemmed:
    DICTIONARY[word] = guid



def search(queryStr):
  queryArr = clean(queryStr)
  stopped = remove_stop_words(queryArr)
  stemmed = stem_words(stopped)
  results = []
  for word in stemmed:
    results.append(DICTIONARY.get(word))
  return filter(None, results)

def stem_words(arr_of_words):
  stemmed_words = []
  for word in arr_of_words:
    stemmed_words.append(stemmer.stem(word))
  return stemmed_words

def remove_stop_words(arr_of_words):
  return [ word for word in arr_of_words if word not in stop_words ]

def clean(text):
  stripped = re.sub("[^a-zA-Z]", " ", text)
  return filter(None, stripped.lower().split(' '))

add(1, 'ingestionText in the Ryan\'s constitution independence television going country video entering &^%$# salerno chair plant ryan ryan ')
add(2, 'ryan')

print search(' @#$@ safd ryan constitution independ')
