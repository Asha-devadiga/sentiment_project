import tweepy
import configparser

# read creds
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

#auth
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#sosadtoday
#narendramodi
def get_tweet_for_user(uname, tweetCount):
    # userID = "sosadtoday"
    # tweets = api.search_tweets('sucide')
    try:
        tweets = api.user_timeline(screen_name=uname,
                                   # 200 is the maximum allowed count
                                   count=tweetCount,
                                   include_rts = False,
                                   # Necessary to keep full_text
                                   # otherwise only the first 140 words are extracted
                                   tweet_mode = 'extended'
                                   )
        tweet_texts = []
        for info in tweets:
            print("ID: {}".format(info.id))
            # print(info.lang)
            if info.lang == 'en':
                print(info.created_at)
                print(info.full_text)
                tweet_texts.append(info.full_text)
                print("\n")
        return tweet_texts
    except:
        return "Error"
    # # print(api_key)
    # print(tweets)
