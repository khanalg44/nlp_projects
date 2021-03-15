#!/usr/bin/env python3

import numpy as np
import pylab as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import classification_report

import nltk, re, json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
st=PorterStemmer()

def process_text(text):
    text = text.lower()
    
    text = re.sub(r'[/(){}\[\]\|@,;]', '', text)
    text = re.sub(r'[^0-9a-z #+_]', '', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[0-9+]', '', text)
    #remove stopwords and stemmize
    text=" ".join(st.stem(w) for w in text.split() if w not in STOPWORDS)
    return text


def plot_history(history):
    
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.grid()
    plt.legend()


def calc_prediction(model, X_test, y_test, categorical=False, ax=None, title=None):
    data_dir="./"

    with open(data_dir+"label_dict.json", 'r') as fp:
        label_dict = json.load(fp)
        labels = label_dict.keys()

    y_pred = model.predict(X_test)
    
    if categorical:
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
            
    report = classification_report(y_test, y_pred)

    if title: print ("Method:",title)
    print ("Classification Report:\n", report)
    print ()

    #acc_test=accuracy_score(y_test, y_pred)
    #f1_macro=f1_score(y_test, y_pred, average='macro')
    #print ("Test set accuracy score: %1.2f" % (100*acc_test))
    #print ("Test set f1 score: %1.2f " % (100*f1_macro))

    confusionmatrix=confusion_matrix( y_test, y_pred)
    if ax==None:
        (fig, ax) = plt.subplots(1,1,figsize=(6,6))
    sns.heatmap(confusionmatrix, cmap='viridis', annot=True, cbar=False, ax=ax,
            xticklabels=labels, yticklabels=labels)
    ax.set_title(title, fontsize=16)
    
#calc_prediction(model_bow, X_test_bow, y_test_bow, title="LR (BOW)")



doc1="Mr Trump responded on Tuesday with his lengthiest statement since he left office a month ago. The Republican Party can never again be respected or strong with political 'leaders' like Senator Mitch McConnell at its helm, the press release reads"

doc2="England were bowled out for 134 and 164, losing all 10 second-innings wickets to spin as India levelled the four-match series at 1-1. A dry pitch was heavily criticised for offering extravagant spin throughout."

doc3=" R. Ashwin took 5 wickets in what became the last innings of the series to secure an innings and a 25 runs."

doc4="True Thompson makes an adorable cameo in Khloe Kardashian's new makeup tutorial video. Many people didn't like it."

def get_test_doc():
    return [doc1, doc2, doc3, doc4]

