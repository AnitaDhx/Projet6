#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:40:45 2018

@author: anita
"""

from flask import Flask ,render_template, flash, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.widgets import TextArea
from flask_wtf.csrf import CSRFProtect

import pandas as pd

import re #for regular expression
from nltk.stem import SnowballStemmer #NLTK
from nltk.corpus import stopwords
import gensim.corpora as corpora
import os
from gevent import pywsgi

#import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

csrf = CSRFProtect()

app = Flask(__name__)


class LoginForm(FlaskForm):
    Question = StringField('Enter your question')
    Body = StringField('Enter the text', widget=TextArea())
    submit = SubmitField('Search for tags')


@app.route('/', methods=['GET', 'POST'])
def base_handler():
    form = LoginForm()
    if form.validate_on_submit():
#    if request.method=='POST':
        result=request.form
        Question = result['Question']
        Body = result['Body']
        
        #get vraibles in the right format to search tags
        Text = Question + ' '+ Body
        #clean_text = text_to_words(Text)
        
        stop_words = stopwords.words('english')
        stop_words.extend(['use','get', 'like', 'work', 'name', 'want', 'need', 'would', 'know', 'x', 'l', 'b', 'e'])
        letters_only = re.sub("[^a-zA-Z]", " ", Text)
        words = letters_only.lower().split()
        snowball_stemmer = SnowballStemmer('english')
        meaningful_wordsStem = [snowball_stemmer.stem(w) for w in words]
        meaningful_words = [w for w in meaningful_wordsStem if not w in stop_words]
        
        #For unsupervise learning
        bow = id2word.doc2bow(meaningful_words)
        model = lda_model[bow]
        liste_mots = lda_model.show_topic(max(model, key=lambda item: item[1])[0])
        tagsunsup = [i[0] for i in liste_mots[0:4]]
        
        #For supervise learning
        
        def adjusted_classes(y_scores, t):
            return [1 if y >= t else 0 for y in y_scores]
        
        sentence = [" ".join(meaningful_words)]
        tagText = LogistiReg.predict_proba(sentence)
        prediction_df_adjust = pd.DataFrame(columns=tag_vocab,data=tagText)
        
        y_pred_proba_adj = pd.DataFrame()
        for category in tag_vocab:
            y_pred_proba_adj[category] = adjusted_classes(prediction_df_adjust[category], 0.2)
        
        tagsup = y_pred_proba_adj.columns[(y_pred_proba_adj == 1).iloc[0]].values
        
        flash('Tags with unsupervise learning: {} '.format(tagsunsup))
        flash('Tags with supervise learning: {} '.format(tagsup))
        
#        flash('Time enlapse {}'.format(
#            form.Time.data))
#        return redirect('/result')
        
    return render_template('home.html', form=form)



def start_server():   
    app.secret_key = os.urandom(24)
    print("SERVER STARTED")
    port = int(os.environ.get('PORT', 5000))
    pywsgi.WSGIServer(('0.0.0.0', port), app).serve_forever()

if __name__ == '__main__':
    id2word = joblib.load('id2word.pkl')
    lda_model = joblib.load('lda_model50.pkl')
    LogistiReg = joblib.load('LogisitcReg.pkl')
    tag_vocab = joblib.load('tag_vocab.pkl')
#    app.run(debug=True)
    start_server()