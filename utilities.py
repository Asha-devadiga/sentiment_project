import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# from wordcloud import WordCloud
import seaborn as sns
import numpy as np
import re
import string
import time
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

stop_words = set(stopwords.words('english'))

def Dataset_Reader(filepath, cols):
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    df.columns = cols
    # Removing the unnecessary columns.
    df = df[['sentiment', 'tweet']]
    return df

# def del_cols(df, col_names):
#     for col in col_names:
#         del df[col]
#     return df

def text_preprocessor(text):
    #lowercase text
    text = text.lower()

    #remove urls
    text = re.sub(r"http\S+www\S+|https\S+", "", str(text), flags=re.MULTILINE)
    # text = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(text), flags=re.MULTILINE)

    #remove punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove user @ references and #
    text = re.sub(r'\@\w+|\#', "", text)

    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
              ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

    for emoji in emojis.keys():
        text = text.replace(emoji, "EMOJI" + emojis[emoji])

    #tokenize text
    text_tokens = word_tokenize(text)
    # remove stopwords
    filtered_words  = [word for word in text_tokens if word not in stop_words]

    # stemming play
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]

    #lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    if lemma_words !=[]:
        text = " ".join(lemma_words)
    print(text)
    return text

def generate_wordCloud(Words):
    return WordCloud(max_words=1000, width=1600, height=800,
              collocations=False).generate(" ".join(Words))


def model_Evaluate(model, X_test, y_test):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)

    # Print the evaluation metrics for the dataset.
    # print(classification_report(y_test, y_pred))

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()
    return classification_report(y_test, y_pred)


def SaveModel(filename, model):
    file = open(filename, 'wb')
    pickle.dump(model, file)
    file.close()

# result = text_preprocessor("this is so sad, I am going to die today and also bit happy and bad mooding, playing, jumping, loving!!!!")
# print(result)