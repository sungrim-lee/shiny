from shiny import App, render, ui
from pathlib import Path

from bs4 import BeautifulSoup   
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer 
from nltk.util import ngrams
from zipfile import ZipFile

import unicodedata                  
import pickle
import re                           
import string
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors=['#5A69AF','#579E65','#F9C784','#FC944A','#F24C00','#00B825','#FC944A','#EF4026','goldenrod','green',
          '#F9C784','#FC944A','coral','#5A69AF','#579E65','#F9C784','#FC944A','#F24C00','#00B825','#FC944A']):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i], rotation=20,
                    horizontalalignment='center', verticalalignment='center')


def load_data(path):
    return pd.read_csv(path)


def get_business_info_from_id(b_id, path):
    df = load_data(path)
    b_info = df[df['business_id'] == b_id]
    return None if len(b_info) == 0 else b_info


def get_refer_table(b_id, high, low, isHighRatings=True):
    if isHighRatings:
        df = load_data(high)
    else:
        df = load_data(low)
    return df[df['business_id']==b_id]


def print_business_info(info_row):
    info_row['postal_code'] = info_row['postal_code'].astype('int64').astype('str')
    name = 'Business Name: ' + str(info_row['name'].values[0]) + '\n'
    add = 'Business Address: ' + str(', '.join(list(info_row[['address', 'city', 'state', 'postal_code']].values[0]))) + '\n'
    star = 'Business Stars: ' + str(info_row['stars'].values[0]) + '\n'
    cat = 'Business Categories: ' + str(','.join(info_row['categories'])) + '\n'
    return name, add, star, cat


def screen_cat(similar_list, arr):
    for ele in similar_list:
        if ele in arr:
            return True
    return False


def get_potential_reviewers_history_from_today(df1, df2, cat_list, year_diff, isHigh=True):
    ratings = [5, 5] if isHigh else [1, 2]
    sim_busi_df = df1[df1['categories'].apply(lambda x:screen_cat(cat_list, x))].reset_index(drop=True)
    ref_df = sim_busi_df[(sim_busi_df['user_id'].isin(df2['user_id'])) & ((sim_busi_df['stars']==ratings[0])|(sim_busi_df['stars']==ratings[1]))].reset_index(drop=True)
    ref_df['date'] = pd.to_datetime(ref_df['date'])

    recent_time = pd.to_datetime('today').normalize() - pd.DateOffset(years=year_diff)
    ref_df = ref_df[ref_df['date'] > recent_time].reset_index(drop=True)
    return ref_df


def get_most_recent_reviews(df):
    recent_df = df.loc[:, ['user_id', 'date']]
    if len(recent_df['user_id'].unique()) < 5:
        recent_idx = df.index
    else:
        recent_idx = recent_df[recent_df.groupby(['user_id'], sort=False)['date'].transform(max) == recent_df['date']].index
    
    return df.iloc[recent_idx].reset_index(drop=True)


def text_cleaning(text_series, n_gram):
    text_series = text_series.apply(lambda x:removeTags(x))
    text_series = text_series.apply(lambda x:removeAccents(x))
    text_series = text_series.apply(lambda x:x.translate(str.maketrans('', '', string.punctuation)))
    text_series = text_series.apply(lambda x:removeStopwords(x))
    text_series = text_series.apply(lambda x:removeWhitespaces(x))
    text_series = text_series.apply(lambda x:x.lower())
    text_series = text_series.apply(lambda x: ' '.join(map('_'.join, list(ngrams(x.split(), n_gram)))))
    return text_series


## Text Cleaning Functions
# remove tags
def removeTags(text):
    return BeautifulSoup(text, 'html.parser').get_text()
# remove accents
def removeAccents(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
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

app_ui = ui.page_fluid(
    ui.navset_tab(
        ui.nav("Run",
            ui.h1('Personalized WordCloud'),
            'Based on Recommendation System (Philadelphia Exclusive!)',
            ui.h3("How To Run App"),
            ui.h4("1. Download data.zip file from ", ui.tags.a('This', href='https://github.com/sungrim-lee/yelp_nn_wordcloud/raw/main/shiny_data/data/data.zip')),
            ui.h4("2. Upload data.zip "),
            ui.h4("3. Enter Your Business ID "),
            ui.tags.b('You may find business_id available on the other tab'),
            ui.h4("4. Select Various Options "),
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_checkbox(
                        "potential", 
                        "Want Potential Positive Reviewers?", 
                        value=True
                    ),
                    ui.input_checkbox(
                        "history", 
                        "Want Positive History with respect to Potential Reviewers?", 
                        value=True
                    ),
                    ui.input_select(
                        "ngram", "Select length of words", {'1': 1, '2': 2, '3': 3, '4': 4}
                    ),
                    ui.input_text(
                        "text_id", 
                        "Business ID", 
                        placeholder="Enter text"
                    ),
                    ui.input_file(
                        "file1", 
                        "Choose a file to upload:", 
                        multiple=True
                    ),
                ),
                ui.panel_main(
                    ui.output_plot("plot"),
                    ui.output_text_verbatim("txt"),
                ),
            )),
        ui.nav("Business List", ui.output_table("table")),
    ),
)

def server(input, output, session):
    @output
    @render.plot(alt="Word Count")
    def plot():
        file_infos = input.file1()
        if not file_infos:
            raise IOError('Import "data.zip" File')
        if len(file_infos) != 1:
            raise IOError('Unsupported Files Imported')
        if file_infos[0]['name'] != 'data.zip':
            raise IOError('Unsupported Files Imported')
            
        zf = ZipFile(file_infos[0]['datapath'], 'r')
        zf.extractall('./')
        zf.close()

        city_b_path = 'city_business.csv'
        isInclude = Path(city_b_path).is_file()
        high_r_path = 'high_ratings.csv'
        isInclude = isInclude & Path(high_r_path).is_file()
        low_r_path = 'low_ratings.csv'
        isInclude = isInclude & Path(low_r_path).is_file()
        ratings_path = 'rating_history.csv'
        isInclude = isInclude & Path(ratings_path).is_file()
        pickle_path = 'sim.pkl'
        isInclude = isInclude & Path(pickle_path).is_file()

        if not isInclude:
            raise IOError('Unsupported Files Imported')
        
        b_id = input.text_id()
        if not b_id:
            raise IOError('Please Enter Business ID')
        
        with open(pickle_path, "rb") as f:
            sim_dict = pickle.load(f)

        user_business_df = pd.read_csv(ratings_path)
        b_info = get_business_info_from_id(b_id, city_b_path)
        if b_info is None:
            raise ValueError('No such Business ID')
        # print_business_info(b_info)
        refer_reviews_df = get_refer_table(b_id, high_r_path, low_r_path, isHighRatings=input.potential()).reset_index(drop=True)

        user_business_df['categories'] = user_business_df['categories'].apply(lambda x: list(set(list(map(str.strip, x.split(','))))) if x is not None else 'None')
        b_cat = user_business_df[user_business_df['business_id'] == b_id]['categories'].iloc[0][0]
        similar_cat = list(np.array(sim_dict[b_cat])[:5, 0])
        similar_cat.insert(0, b_cat)

        ref_df = get_potential_reviewers_history_from_today(user_business_df, refer_reviews_df, similar_cat, year_diff=5, isHigh=input.history())
        if len(ref_df)==0:
            raise ValueError('There is no potential review history for this setting')
        nlp_df = get_most_recent_reviews(ref_df)
        text_series = text_cleaning(nlp_df['text'], n_gram=int(input.ngram()))
        if len(text_series)==0:
            raise ValueError('There is no potential review history for this setting')
        word_count = dict()
        max_n = 10
        for row in text_series.apply(lambda x:x.split(' ')):
            for elem in row:
                word_count[elem] = word_count.get(elem, 0) + 1

        word_count = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)[:max_n]}
        max_num = 20 if len(word_count) > 20 else len(word_count)
        bubble_chart = BubbleChart(area=list(word_count.values())[:max_num],
                           bubble_spacing=0.5)
        bubble_chart.collapse()
        
        fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        fig.set_size_inches(9, 13, forward=True)
        bubble_chart.plot(
            ax, list(word_count.keys())[:max_num])
        plt.title('Frequently Used Words For Potential Positive/Negative Reviewers')
        ax.axis("off")
        ax.relim()
        ax.autoscale_view()
        
    def get_b_info():
        files = input.file1()
        b_id = input.text_id()
        if not files or not b_id:
            return ['','','','']
        # files = sorted(files, key=lambda x:x['name'])
        # file_paths = list(map(lambda x:x['datapath'] , files))
        city_b_path = 'city_business.csv'
        ratings_path = 'rating_history.csv'
        
        
        user_business_df = pd.read_csv(ratings_path)
        b_info = get_business_info_from_id(b_id, city_b_path)
        return print_business_info(b_info)
        
    @output
    @render.text
    def txt():
        business = get_b_info()
        return f'''
                {business[0]}
                {business[1]}
                {business[2]}
                {business[3]}
                '''
    @output
    @render.table        
    def table():
        ref_file = Path(__file__).parent / "BusinessInfo.csv"
        df = pd.read_csv(ref_file)
        # Use the DataFrame's to_html() function to convert it to an HTML table, and
        # then wrap with ui.HTML() so Shiny knows to treat it as raw HTML.
        return df
        
app = App(app_ui, server)