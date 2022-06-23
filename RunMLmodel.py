import pickle
from utilities import text_preprocessor

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


def predict(vectoriser, model, rawtweetList):

    # Predict the sentiment
    processedTexts = []
    for eachTweet in rawtweetList:
        processedTexts.append(text_preprocessor(eachTweet))
    textdata = vectoriser.transform(processedTexts)
    sentiment = model.predict(textdata)


    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(rawtweetList, sentiment):
        data.append((text, pred))

    
    result = []
    if data!=[]:
        print(data)
        for sentiments in data:
            result.append(sentiments[1])
    
    return result


def analyse(textList):
    vectoriser, LRmodel = load_models()
    sentiments = predict(vectoriser, LRmodel, textList)
    results = []
    for sentiment in sentiments:
        sentiment = str(sentiment)
        sentiment = sentiment.replace("0", "Negative Tweet 😢")
        sentiment = sentiment.replace("1", "Positive Tweet 😀")
        results.append(sentiment)
    return results
# if __name__ == "__main__":
#     # Loading the models.
#     vectoriser, LRmodel = load_models()
#
#     # Text to classify should be in a list.
#     text = ["I hate this job",
#             "May the Force be with you.",
#             "I am so happy for you :)",
#             "Mr.Brain, I don't feel so good today"]
#
#     df = predict(vectoriser, LRmodel, text)
#     # print(df.head())