import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
import numpy as np

'''
add to data_prep
import nlp_fraud
filename = '../data/2000rows_withnewfeatures.json'
df = nlp_fraud.nlp(filename, max_features=500, n_clusters=5)
data = pd.concat([data, pd.get_dummies(df,prefix='nlp')], axis=1)
'''

def tfidf(X, max_feat):
    '''Extracts html description string text
    and returns term-document matrix'''
    descriptions = X.apply(lambda x: BeautifulSoup(x, "html.parser").get_text().replace(u'\xa0', u''))
    vectorizer = TfidfVectorizer(strip_accents='unicode',max_features=max_feat,lowercase=True, stop_words='english').fit_transform(descriptions)
    return vectorizer

def clustering(X, n_clusters):
    '''creates n_clusters clusters of descriptions'''
    cluster_model = KMeans(n_clusters=n_clusters, max_iter=1000)
    cluster_model.fit(X)
    return cluster_model

def pickles(data):
    if not os.path.exists('nlp.pkl'):
        data.to_pickle('nlp.pkl')

def nlp(filename, max_features=1000, n_clusters=5):
    filename = '../data/2000rows_withnewfeatures.json'
    df = pd.read_json(filename)
    X = tfidf(df.description, max_features)
    cluster_model = clustering(X, n_clusters)
    prediction = cluster_model.predict(X)
    return prediction
    # return X
