from utilities import *
warnings.filterwarnings("ignore")

df = pd.read_csv('./training_allCleanData.csv', index_col=0)
tweets, sentiment = list(df['tweet']), list(df['sentiment'])
negative_Tweets = []
positive_Tweets = []
for i, tweet in enumerate(tweets):
    if sentiment[i] == 0:
        negative_Tweets.append(tweet)
    else:
        positive_Tweets.append(tweet)


# Neg_cloud = generate_wordCloud(negative_Tweets)
# Pos_cloud = generate_wordCloud(positive_Tweets)
# plt.imshow(Neg_cloud)
# plt.show()
# plt.imshow(Pos_cloud)
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(tweets, sentiment,
                                                    test_size = 0.05, random_state = 0)
print("Data Split done.")

#Termfrequency InverseDocumentFrequency
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print(f'Vectoriser fitted.')
print('No. of feature_words: ', len(vectoriser.get_feature_names()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)
print(f'Data Transformed.')


#BernoulliNB Model
BNBmodel = BernoulliNB(alpha = 2)
BNBmodel.fit(X_train, y_train)
#training accuracy
NB_TrainingAccuracy = model_Evaluate(BNBmodel, X_train, y_train)
print("**********Naive Bayes Training Accuracy**********")
print(NB_TrainingAccuracy)
print("-----------------------------------------------------------------------")

#testing accuracy
NB_TestingAccuracy = model_Evaluate(BNBmodel, X_test, y_test)
print("**********Naive Bayes Testing Accuracy**********")
print(NB_TestingAccuracy)
print("-----------------------------------------------------------------------")


#Logistic Regression Model
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)

#training accuracy
LR_TrainingAccuracy = model_Evaluate(LRmodel, X_train, y_train)
print("**********Logistic Regression Training Accuracy**********")
print(LR_TrainingAccuracy)
print("-----------------------------------------------------------------------")

#testing accuracy
LR_TestingAccuracy = model_Evaluate(LRmodel, X_test, y_test)
print("**********Logistic Regression Testing Accuracy**********")
print(LR_TestingAccuracy)
print("-----------------------------------------------------------------------")


#uncomment this to save model
# Save model
#SaveModel(filename='vectoriser-ngram-(1,2)_copy.pickle', model=vectoriser)

#SaveModel(filename='Sentiment-LR.pickle', model=LRmodel)

#SaveModel(filename='Sentiment-BNB.pickle', model=BNBmodel)


