from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from collections import Counter
import emoji
import nltk
import string
import unicodedata
import re
from sklearn.decomposition import NMF
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from sklearn.feature_extraction.text import  TfidfVectorizer 
nltk.download('all')
nltk.download('punkt')
# nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns


extract = URLExtract()
# Function to bes used in app.py file
def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]
    
    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)
    
# Finding the busiest users
def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    # Downloaded from google having english + hindi stop_words because nltk doesn't 
    # have hinglish stopwords
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA.keys()])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def sentiment_analysis(selected_user,df):
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    vds = SentimentIntensityAnalyzer()
    for index,row in df.iterrows():
        negative, neutral, positive,_ = vds.polarity_scores(row['message']).values()
        sentiment = (positive-negative)
        df.loc[index,'sentiment'] = sentiment
    pos =0
    neg =0
    neut =0
    for index,row in df.iterrows():
        if row['sentiment']>0:
            pos = pos+1
        elif row['sentiment']<0:
            neg = neg+1
        else:
            neut = neut +1
            
    sentiment_score = {"Positive" : pos, "Negative" : neg, "Nuetral" : neut}
    category = list(sentiment_score.keys())
    score = list(sentiment_score.values())
    plt.bar(category, score,width = 0.4)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.show() 
#     return pos,neg,neut


def elimina_tildes(cadena):
    s = ''.join((c for c in unicodedata.normalize('NFD',cadena) if unicodedata.category(c) != 'Mn'))
    return s

def custom_tokenizer(text):
    remove_punct = str.maketrans('', '', string.punctuation)
    text = text.translate(remove_punct)
    remove_digits = str.maketrans('', '', string.digits)
    text = text.lower().translate(remove_digits)
    shortword = re.compile(r'\W*\b\w{1,2}\b')
    text = shortword.sub('', text)
    text = elimina_tildes(text)
    text = re.sub(r'([a-z])\1+', r'\1', text)
    text = re.sub(r'(ha)[ha]*', 'ha', text)
    tokens = word_tokenize(text)
    # stop_words = stopwords.words('english')
    # stop_words = stopwords.words('english')
    # f = open('stop_hinglish.txt', 'r')
    # stop_words = f.read()
    stopwords_hinglish = pd.read_csv("stop_hinglish.txt", header = None)
    stop_words =  []
    for i in stopwords_hinglish[0]:
        stop_words.append(i)
    tokens_stop = [y for y in tokens if y not in stop_words]
    return tokens_stop

def run_NMF_model(selected_user,data,max_df,n_components):
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    tfidf = TfidfVectorizer(tokenizer=custom_tokenizer,max_df=max_df,min_df = 50, max_features=5000) 
    X = tfidf.fit_transform(data)       
    nmf = NMF(n_components=n_components,random_state=0)
    doc_topics = nmf.fit_transform(X)
    t = np.argmax(doc_topics,axis=1)
    counts = pd.Series(t).value_counts()
    d = nmf.components_
    w = tfidf.get_feature_names_out()
    words = []
    for r in range(len(d)):
        a = sorted([(v,i) for i,v in enumerate(d[r])],reverse=True)[0:20]
        words.append([w[e[1]] for e in a])
    return doc_topics, t, words

def plot_topics(selected_user,data):
    doc_topics, t, words = run_NMF_model(selected_user,data,0.9,5)
    t = np.argmax(doc_topics,axis=1)
    plt.bar(pd.Series(t).unique(),pd.Series(t).value_counts())  
    plt.xlabel("Topics")
    plt.ylabel("Count")
    plt.show()