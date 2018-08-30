
# coding: utf-8

#############################################################
### Create tags with unsupervised and supervised learning###
#############################################################

#importation des librairies

import pandas as pd #data processing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

#NLTK
import nltk #natural language toolkit
#nltk.download()
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords # Import the stop word list
#nltk.download()

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis.gensim 

from bs4 import BeautifulSoup #to remove HTML Markup

import re #for regular expression

from nltk.stem import SnowballStemmer #Stemmming

#create matrix of words frequency
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import metrics


#save model on computer
from sklearn.externals import joblib


# ***

# In[2]:


# import data
df1 = pd.read_csv('/Users/anita/Documents/FormationDataScientist/Projet_StackOverflow/QueryResults(1).csv',encoding = 'utf-8')
df2 = pd.read_csv('/Users/anita/Documents/FormationDataScientist/Projet_StackOverflow/QueryResults(2).csv',encoding = 'utf-8')
df3 = pd.read_csv('/Users/anita/Documents/FormationDataScientist/Projet_StackOverflow/QueryResults(3).csv',encoding = 'utf-8')


# In[ ]:


#concatenat the 3 data base
df = pd.concat([df1, df2, df3], ignore_index=True)

# In[3]:

df1=0
df2=0
df3=0

# In[ ]:

#Names of the columns
df.columns.values

#############################################################
### Cleanning the dataset
#############################################################
# In[ ]:


# get rid of duplicates rows
df= df.drop_duplicates()


# In[ ]:


prin(df.isnull().sum())


# In[ ]:


df.reset_index(drop=True, inplace=True)


# * Join Title and body

# In[ ]:


#join title and body for each question in a new column
# name "text"
df['Text'] = df[['Title', 'Body']].apply(lambda x: ' '.join(x), axis=1)
print(df.head())

# In[297]:

# Check for null values in Text column
df["Text"].isnull().sum()


# In[85]:


# print the entier cell of the first row
pd.set_option('display.max_colwidth', -1)
pprint(df.loc[2])


# In[86]:


# get back to usual vizualisation
pd.set_option('display.max_colwidth', 50)
df.head()

#####################################################
#####################################################


