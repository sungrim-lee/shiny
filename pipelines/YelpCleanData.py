#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup       
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer 
from nltk import bigrams
from nltk.util import ngrams
from wordcloud import WordCloud
from functools import reduce
from gensim.models import Word2Vec
from functools import reduce

import unicodedata                  
import contractions 
import re                           
import string
import json
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

import warnings
warnings.simplefilter('ignore')


def load_data(path):
    """
    Load csv file
    Arguments:
    path: type of string
    return:
    pandas dataframe
    """
    return pd.read_csv(path)


def select_index_with_thres(df, col, threshold):
    """
    Filter Values within a Column with respect to counts
    Arguments:
    df: pandas dataframe
    col: type of string
    threshold: integer
    return:
    pandas dataframe
    """
    return df[col].value_counts()[df[col].value_counts() > threshold].index


def filter_index(df, col_pairs):
    """
    Filter Dataframe given specific columns corresponding index array
    Arguments:
    df: pandas dataframe
    col_pairs: column name, and corresponding indexes
    return:
    pandas dataframe
    """
    for col, ids in col_pairs:
        df = df[df[col].isin(ids)]
    return df


def reset_index_cols(df, cols):
    """
    Reset index with respect to columns
    Arguments:
    df: pandas dataframe
    cols: column names, list of string
    return:
    pandas dataframe
    """
    return df[cols].reset_index(drop=True)


def split_most_recent(df, col='user_id'):
    """
    Filter a most recent review for each user_id
    Arguments:
    df: pandas dataframe
    col: user_id column
    return:
    not recent data, recent data, type of pandas dataframe
    """
    user_date = df.loc[:, [col,'date']]
    user_date['date'] = pd.to_datetime(user_date['date'])
    
    recent_idx = user_date[user_date.groupby([col], sort=False)['date'].transform(max) == user_date['date']].index
    recent_df = df.iloc[recent_idx].reset_index(drop=True)
    previous_df = df[~df.index.isin(recent_idx)].reset_index(drop=True)
    return previous_df, recent_df


def generate_labelEncoder(df, col):
    """
    Encode column values
    Arguments:
    df: pandas dataframe
    col: column_name, string
    return:
    Encoded data, type of LabelEncoder
    """
    le = LabelEncoder()
    le.fit(df[col])
    return le


def get_array_from_df(df, cols):
    """
    Get numpy type of values
    Arguments:
    df: pandas dataframe
    cols: column names, list of string
    return:
    numpy array
    """
    return df[cols].values


def melt_with_pivot(df, index, col, val):
    """
    Melt pandas dataframe (Unpivot)
    Arguments:
    df: pandas dataframe
    index: dataframe index name
    col: dataframe column name
    val: dataframe value name
    return:
    pandas dataframe
    """
    pivot_matrix = df.pivot(index=index, columns=col, values=val)
    pairs = pivot_matrix.melt(ignore_index=False).reset_index()
    return pairs


def open_json(filename):
    """
    Open json file
    Arguments:
    filename: file path, string
    return:
    list of json
    """
    json_list = []
    for line in open(filename, 'r'):
        json_list.append(json.loads(line))
    return json_list


def build_word2vec(data, window, sg, negative, alpha, min_alpha, epochs):
    """
    Build word2vec model
    Arguments:
    data: series of sequence based list
    [window, sg, negative, alpha, min_alpha]: hyperparams for model
    epochs: number of epochs
    return:
    Word2vec model
    """
    model = Word2Vec(window=window, sg=sg, negative=negative, alpha=alpha, min_alpha=min_alpha, seed = 1)
    model.build_vocab(data, progress_per=200)
    model.train(data, total_examples=model.corpus_count, epochs=epochs, report_delay=1)
    return model


def get_similar_cat_df(model, cat):
    """
    Get similarity score sorted with respect to a category
    Arguments:
    model: word2vec model
    cat: category name, string
    return:
    pandas dataframe
    """
    simiarityScores = model.wv.similar_by_vector(cat, topn=len(model.wv.key_to_index))
    df = pd.DataFrame(simiarityScores, columns=['Category', 'Similarity'])
    new_row = pd.DataFrame({'Category':cat, 'Similarity':1}, index =[0])
    df = pd.concat([new_row, df]).reset_index(drop = True)
#     df = df[df['Category'].isin(cat_merged)].reset_index(drop=True)
    return df


def get_n_similar_cat(model, cat, cat_set, n=5):
    """
    Get top 'n' similarity scores sorted with respect to a category that is included in the set of category
    Arguments:
    model: word2vec model
    cat: category name, string
    cat_set: a set of categoy, type of set
    return:
    top n categories closer to the given category, type of list
    """
    simiarityScores = model.wv.similar_by_vector(cat, topn=len(model.wv.key_to_index))
    df = pd.DataFrame(simiarityScores, columns=['Category', 'Similarity'])
    new_row = pd.DataFrame({'Category':cat, 'Similarity':1}, index =[0])
    df = pd.concat([new_row, df]).reset_index(drop = True)
    df = df[df['Category'].isin(cat_set)].reset_index(drop=True)
    categories = list(df.head(n)['Category'].values)
    return categories


def get_tsne_embedding(model, perplexity, n_components, n_iter):
    """
    Get T-SNE model and string labels
    Arguments:
    model: word2vec model
    [perplexity, n_components, n_iter]: hyperparams for T-SNE model
    return:
    Embedded matrix, and corresponding string labels
    """
    catLabel = []
    catVector = []

    for category in model.wv.index_to_key:
        catVector.append(model.wv[category])
        catLabel.append(category)

    tsne = TSNE(perplexity=perplexity, n_components=n_components, init='pca', n_iter=n_iter)
    embed = tsne.fit_transform(catVector)
    return embed, catLabel


def tsne_plot(embeddings, labels):
    """
    Plot a graph given T-SNE embedding
    Arguments:
    embeddings: Embedded matrix
    labels: corresponding string labels to embedded matrix
    """
    from matplotlib import pylab
    pylab.figure(figsize=(20,20))
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
    pylab.xlabel('dim-1')
    pylab.ylabel('dim-2')
    pylab.title('t-SNE')
    pylab.show()


def get_business_info_from_id(b_id):
    """
    Get a business's info
    Arguments:
    b_id: business id, string
    return:
    A row of dataframe
    """
    df = load_data('data/city_business.csv')
    b_info = df[df['business_id'] == b_id]
    return None if len(b_info) == 0 else b_info


def get_refer_table(b_id, isHighRatings=True):
    """
    Find review positive/negative ratings with respect to a business ID
    Arguments:
    b_id: business id, string
    return:
    pandas dataframe
    """
    if isHighRatings:
        df = load_data('data/high_ratings.csv')
    else:
        df = load_data('data/low_ratings.csv')
    return df[df['business_id']==b_id]


def print_business_info(info_row):
    """
    Print Business Info
    Arguments:
    info_row: a row of dataframe
    """
    info_row['postal_code'] = info_row['postal_code'].astype('int64').astype('str')
    print('Business Name:', info_row['name'].values[0])
    print('Business Address:', ', '.join(list(info_row[['address', 'city', 'state', 'postal_code']].values[0])))
    print('Average Stars:', info_row['stars'].values[0])
    print('Business Categories:', ','.join(info_row['categories']))


def screen_cat(similar_list, arr):
    """
    Check whether any element of a list is included in a given array
    Arguments:
    similar_list: a list of categories to be compared
    arr: base array
    return:
    Boolean
    """
    for ele in similar_list:
        if ele in arr:
            return True
    return False


def get_potential_reviewers_history_from_today(df1, df2, cat_list, year_diff, isHigh=True):
    """
    Get all reviews that potential positive/negative reviewers wrote for other businesses, 
    and the all review ratings can only be either 5 stars or 1 stars
    these reviews are collected from 'year_diff' years ago
    Arguments:
    df1: base dataframe, pandas dataframe
    df2: postive/negative review dataset for a specific business, pandas dataframe
    cat_list: a category list with respect to the specific business, list of string
    year_diff: reviews from 'n' years ago, integer
    isHigh: whether reviews would be collected from 5 stars reviews or 1 star reviews, Boolean
    return:
    pandas dataframe
    """
    ratings = 5 if isHigh else 1
    sim_busi_df = df1[df1['categories'].apply(lambda x:screen_cat(cat_list, x))].reset_index(drop=True)
    ref_df = sim_busi_df[(sim_busi_df['user_id'].isin(df2['user_id'])) & (sim_busi_df['stars']==ratings)].reset_index(drop=True)
    ref_df['date'] = pd.to_datetime(ref_df['date'])

    recent_time = pd.to_datetime('today').normalize() - pd.DateOffset(years=year_diff)
    ref_df = ref_df[ref_df['date'] > recent_time].reset_index(drop=True)
    return ref_df


def get_most_recent_reviews(df):
    """
    Get most recent reviews for each user_id
    Arguments:
    df: review dataset, pandas dataframe
    return:
    pandas dataframe
    """
    recent_df = df.loc[:, ['user_id', 'date']]
    if len(recent_df['user_id'].unique()) < 5:
        recent_idx = df.index
    else:
        recent_idx = recent_df[recent_df.groupby(['user_id'], sort=False)['date'].transform(max) == recent_df['date']].index
    
    return df.iloc[recent_idx].reset_index(drop=True)


def text_cleaning(text_series, n_gram):
    """
    Implement text processing and ngram counter, and get ngram list
    Arguments:
    text_series: textual reviews, pandas series
    n_gram: ngram is implemented, integer
    return:
    ngram textual reviews, pandas series
    """
    text_series = text_series.apply(lambda x:removeTags(x))
    text_series = text_series.apply(lambda x:removeAccents(x))
#     text_series = text_series.apply(lambda x:appendContractions(x))
    text_series = text_series.apply(lambda x:x.translate(str.maketrans('', '', string.punctuation)))
#     text_series = text_series.apply(lambda x:lemmatizeWords(x))
    text_series = text_series.apply(lambda x:removeStopwords(x))
    text_series = text_series.apply(lambda x:removeWhitespaces(x))
    text_series = text_series.apply(lambda x:x.lower())
    text_series = text_series.apply(lambda x: ' '.join(map('_'.join, list(ngrams(x.split(), n_gram)))))
    return text_series


def draw_word_cloud(text_series, max_word):
    """
    Draw wordcloud for textual reviews gathered
    Arguments:
    text_series: ngram textual reviews, pandas series
    max_word: maximum number of words to be appeared in a wordcloud
    """
    if len(text_series)==0:
        print('There is no potential review history for this setting')
        return
    wordcloud = WordCloud(max_words=max_word, background_color="white", random_state=1).generate(text_series.str.cat(sep=' '))
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
#    plt.savefig('test.jpg')
    plt.show()    
    
    
## Text Cleaning Functions
# remove tags
def removeTags(text):
    return BeautifulSoup(text, 'html.parser').get_text()

# remove accents
def removeAccents(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

# append contracted text
def appendContractions(text):
    appendedWords = []
    for word in text.split():
        appendedWords.append(contractions.fix(word))
    return ' '.join(appendedWords)

# remove stopwords
def removeStopwords(text):
    words = [word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
    return " ".join(words)

# remove white spaces
def removeWhitespaces(text):
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

# lemmatize words
def lemmatizeWords(text):
    ps = PorterStemmer()
    return ' '.join([ps.stem(word) for word in text.split()])