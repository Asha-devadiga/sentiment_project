import sys
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from utilities import model_Evaluate

def load_models():
    '''
    Replace '..path/' by the path of the saved models.
    '''

    # Load the vectoriser.
    file = open('./vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('./Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()

    return vectoriser, LRmodel

df = pd.read_csv('./training_allCleanData.csv', index_col=0)
tweets, sentiment = list(df['tweet']), list(df['sentiment'])
negative_Tweets = []
positive_Tweets = []
for i, tweet in enumerate(tweets):
    if sentiment[i] == 0:
        negative_Tweets.append(tweet)
    else:
        positive_Tweets.append(tweet)

X_train, X_test, y_train, y_test = train_test_split(tweets, sentiment,
                                                    test_size = 0.05, random_state = 0)
print("Data Split done.")

vectoriser, model = load_models()
X_test  = vectoriser.transform(X_test)
X_train  = vectoriser.transform(X_train)

TrainingAccuracy = model_Evaluate(model, X_train, y_train)


print("**********Training Accuracy**********")
print(TrainingAccuracy)
print("-----------------------------------------------------------------------")

TestingAccuracy = model_Evaluate(model, X_test, y_test)


print("**********Testing Accuracy**********")
print(TestingAccuracy)
print("-----------------------------------------------------------------------")
sys.exit(0)