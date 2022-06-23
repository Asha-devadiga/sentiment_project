from flask import Flask, render_template, request
from RunMLmodel import analyse
import GetTweetData
app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        Input = []
        tweetText = request.form["tweetData"]
        Input.append(tweetText)
        result  = analyse(Input)
        result = result[0]
        return render_template("index.html", tweet=result)
    else:
        return render_template("index.html")

@app.route("/livetweets", methods=["POST", "GET"])
def livetweet():
    if request.method == "POST":
        userName = request.form["twitter_add"]
        count = request.form["tweetcount"]
        Tweets = GetTweetData.get_tweet_for_user(userName, count)
        result = ["NA"]
        NumOfTweets = 0
        if Tweets == 'Error':
            Tweets = ["Error while fetching user data!!!"]
        elif Tweets == []:
            Tweets = ["No Tweets were found"]
        else:
            result = analyse(Tweets)
            NumOfTweets = count
        return render_template("livetweets.html", userName=userName, NumOfTweets= NumOfTweets, len = len(Tweets) ,Tweets= Tweets, result=result)
    else:
        return render_template("livetweets.html")


if __name__ == "__main__":
    app.run(debug=True)