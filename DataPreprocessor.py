from utilities import *

#download once
# import nltk
# nltk.download('stopwords')



# Dataset_Reader is function written in utilities which reads specified csv file from path and returns pandas dataframe with specified col names
csv_df = Dataset_Reader(filepath="./training.1600000.processed.noemoticon.csv", cols=["sentiment", "id", "date", "flag", "user", "tweet"])

# (not required !!) c_time captures current time, it will be used to calculate total time taken for preprocessing data
c_time = time.time()
# np.vectorize calls text_preprocessor method which written in utilities, to which each data from tweet col is passed as argument in continous loop
# text_preprocessor is method which returns Preprocessed text (ctrl+click on method to know more)
csv_df['preprocessed_text'] = np.vectorize(text_preprocessor)(csv_df['tweet'])

csv_df['sentiment'] = csv_df['sentiment'].replace(4,1)
print(f'Text Preprocessing complete.')
print(f'Time Taken: {round(time.time()-c_time)} seconds')
csv_df.to_csv('Training-Preprocessed-Data.csv', index = False, encoding='utf-8') # False: not include index
print(len(csv_df))
print(csv_df.head())