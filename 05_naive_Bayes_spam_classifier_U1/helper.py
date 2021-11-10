#!/usr/bin/env python3
import pandas as pd
import re, string

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from collections import Counter

def PrepareDoc(doc):
    # lower case the doc
    doc = doc.lower()
    
    # remove punctuations
    puncs_re = re.compile('[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')
    doc = puncs_re.sub('', doc)
    
    # remove the stopword
    doc = ' '.join([word for word in doc.split() if word not in stop_words])
    return doc

def FreqsWords(doc):
    return Counter(doc.split())


print (freqs)
    
