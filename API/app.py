#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:05:28 2020

@author: marco
"""

#from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask
from flask import request, jsonify
import pandas as pd
import re
from joblib import load
import nltk
from nltk import word_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
    
import yake

app = Flask(__name__)

# decorator for unsupervised Tag Recommendation

def extract_tags(text):
    simple_kwextractor = yake.KeywordExtractor()
    post_keywords = simple_kwextractor.extract_keywords(text)
    post_keywords = list(set(post_keywords))
    sentence_output = ""
    for word, number in post_keywords[:2]:
        sentence_output += word + " "
    
    return sentence_output

# decorators for unsupervised Tag Recommendation
def process_text(text):
    text = text.lower() # lowercase
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"[?.!/;:']", " ", text)
    text = re.sub(r"[<>\@%*=]", " ", text)
    text = re.sub(r"[\ |\]|\[|\|\/|\#|\:]", " ", text)
    text = re.sub(r"\'\n", " ", text) #line breaks
    text = re.sub(r"\'\xa0", " ", text) # xa0 Unicode representing spaces
    text = re.sub('\s+', ' ', text) # one or more whitespace characters
    text = text.strip(' ') # spaces
    tokens = word_tokenize(text)
    return tokens

def vectorize_query(tokens):
    tfidfVectorizer = load("vectorizer.pkl")
    vectorized_query = tfidfVectorizer.transform(tokens).todense()
    return vectorized_query


def predict_tags(vectorized_query):
    model = load('model.pkl') # trained Linear Regression Classifier
    y_preds= model.predict(vectorized_query)
    popular_tags = load('tags.pkl') # I load the most popular tags from which I will choose 
    df_probs = pd.DataFrame(y_preds, columns= popular_tags).T
    df_probs["probability"] = df_probs.sum(axis=1)
    df_probs.reset_index(inplace=True)
    
    df_probs = df_probs.sort_values(by='probability', ascending=False)
    tags = df_probs['index'][:5].tolist() # I choose teh most 5 probable tags
    return tags

@app.route('/api_message', methods=["POST"])

def api_message():

    if request.headers['Content-Type'] == 'application/json':
        data = request.get_json()
        keywords = extract_tags(data.get('text',''))
        tokens = process_text(data.get('text',''))
        vectors = vectorize_query(tokens)
        tags = predict_tags(vectors)
        return jsonify(tags_supervised = tags, tags_unsupervised = keywords )
    
if __name__ == '__main__':
      app.run(debug=False, host='0.0.0.0', port=5924)





