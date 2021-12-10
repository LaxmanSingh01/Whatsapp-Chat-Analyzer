import regex
import pandas as pd
import numpy as np
import emoji
from collections import Counter
import matplotlib.pyplot as plt
from regex.regex import Pattern
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import warnings
warnings.filterwarnings('ignore')
import io
from urlextract import URLExtract
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments=SentimentIntensityAnalyzer()

def date_time(s):
    pattern = '^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -'
    result = regex.match(pattern, s)
    if result:
        return True
    return False

def find_author(s):
    s = s.split(":")
    if len(s)==2:
        return True
    else:
        return False

def getDatapoint(line):
    splitline = line.split(' - ')
    dateTime = splitline[0]
    date, time = dateTime.split(", ")
    message = " ".join(splitline[1:])
    if find_author(message):
        splitmessage = message.split(": ")
        author = splitmessage[0]
        message = " ".join(splitmessage[1:])
    else:
        author= None
    return date, time, author, message


data=[]
def fetch_data(file):
    file = io.StringIO(file)
    file.readline()
    messageBuffer = []
    date, time, author = None, None, None
    while True:
        line = file.readline()
        if not line:
            break
        line = line.strip()
        if date_time(line):
            if len(messageBuffer) > 0:
                data.append([date, time, author, ' '.join(messageBuffer)])
            messageBuffer.clear()
            date, time, author, message = getDatapoint(line)
            messageBuffer.append(message)
        else:
            messageBuffer.append(line)
        df = pd.DataFrame(data, columns=["Date", 'Time', 'User', 'Message'])
        df=df.drop_duplicates()
        df['Date'] = pd.to_datetime(df['Date'])
        df.dropna(inplace=True)
        df['Year']=df['Date'].dt.year
        df['Month_name']=df['Date'].dt.month_name()
        df['Month']=df['Date'].dt.month
        df['date']=df['Date'].dt.day
        df['day_name']=df['Date'].dt.day_name()
        df["positive"]=[sentiments.polarity_scores(i)["pos"] for i in df["Message"]]
        df["negative"]=[sentiments.polarity_scores(i)["neg"] for i in df["Message"]]
        df["neutral"]=[sentiments.polarity_scores(i)["neu"] for i in df["Message"]]
    return df

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['Message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['Message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['Message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    timeline = df.groupby(['Year', 'Month', 'Month_name']).count()['Message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['Month_name'][i] + "-" + str(timeline['Year'][i]))

    timeline['Time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    daily_timeline = df.groupby('date').count()['Message'].reset_index()

    return daily_timeline

def most_active_users(df):
    x = df['User'].value_counts().head()
    df = round((df['User'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def least_active_users(df):
    x = df['User'].value_counts().tail()
    df = round((df['User'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    return df['Month_name'].value_counts()

def year_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    return df['Year'].value_counts()

def create_wordcloud(selected_user,df):
    
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    temp = df[df['Message'] != '<Media omitted>']
    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    df_wc = wc.generate(temp['Message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    
    temp = df[df['Message'] != '<Media omitted>']

    words = []

    for message in temp['Message']:
        for word in message.lower().split():
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    emojis = []
    for message in df['Message']:
        emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def analysis_time(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    return df['Time'].value_counts().head(20)

# Sentiment Analysis


def score(a,b,c):
    
    if (a>b) and (a>c):
        return "Positive "
    if (b>a) and (b>c):
        return "Negative"
    if (c>a) and (c>b):
        return "Neutral"

    
