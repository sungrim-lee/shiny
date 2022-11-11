from shiny import App, render, ui
              
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AB_PATH = 'https://github.com/sungrim-lee/sungrim-lee.github.io/blob/main/myshinyapp/'

def load_data(path):
    return pd.read_csv(AB_PATH+path)


def open_json(filename):
    json_list = []
    for line in open(filename, 'r'):
        json_list.append(json.loads(line))
    return json_list


def get_n_similar_cat(dic, b_cat, n=5):
    return list(dic[b_cat][:n, 0])


def get_business_info_from_id(b_id):
    df = load_data('data/city_business.csv')
    b_info = df[df['business_id'] == b_id]
    return None if len(b_info) == 0 else b_info


def get_refer_table(b_id, isHighRatings=True):
    if isHighRatings:
        df = load_data('data/high_ratings.csv')
    else:
        df = load_data('data/low_ratings.csv')
    return df[df['business_id']==b_id]


def print_business_info(info_row):
    info_row['postal_code'] = info_row['postal_code'].astype('int64').astype('str')
    print('Business Name:', info_row['name'].values[0])
    print('Business Address:', ', '.join(list(info_row[['address', 'city', 'state', 'postal_code']].values[0])))
    print('Average Stars:', info_row['stars'].values[0])
    print('Business Categories:', ','.join(info_row['categories']))


def screen_cat(similar_list, arr):
    for ele in similar_list:
        if ele in arr:
            return True
    return False


def get_potential_reviewers_history_from_today(df1, df2, cat_list, year_diff, isHigh=True):
    ratings = 5 if isHigh else 1
    sim_busi_df = df1[df1['categories'].apply(lambda x:screen_cat(cat_list, x))].reset_index(drop=True)
    ref_df = sim_busi_df[(sim_busi_df['user_id'].isin(df2['user_id'])) & (sim_busi_df['stars']==ratings)].reset_index(drop=True)
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



app_ui = ui.page_fluid(
    ui.h2("Word Count"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_checkbox(
                "potential", 
                "Potential High Review", 
                value=True
            ),
            ui.input_checkbox(
                "history", 
                "Positive Review History", 
                value=True
            ),
            ui.input_text(
                "text_id", 
                "Text input", 
                placeholder="Enter text"
            ),
        ),
        ui.panel_main(
            ui.output_plot("plot"),
        ),
    ),
)

def server(input, output, session):
    @output
    @render.plot(alt="Word Count Plot")
    def plot():
        b_id = input.text_id()

        with open(AB_PATH+'sim.pkl', 'rb') as p:
            sim_dict = pickle.load(p)

        user_business_df = load_data('data/rating_history.csv')
        b_info = get_business_info_from_id(b_id)
        if b_info is None:
            raise ValueError('No such Business ID')
        print_business_info(b_info)
        refer_reviews_df = get_refer_table(b_id, isHighRatings=input.potential()).reset_index(drop=True)

        user_business_df['categories'] = user_business_df['categories'].apply(lambda x: list(set(list(map(str.strip, x.split(','))))) if x is not None else 'None')
        b_cat = user_business_df[user_business_df['business_id'] == b_id]['categories'].iloc[0][0]
        similar_cat = list(np.array(sim_dict[b_cat])[:5, 0])
        similar_cat.insert(0, b_cat)

        ref_df = get_potential_reviewers_history_from_today(user_business_df, refer_reviews_df, similar_cat, year_diff=5, isHigh=input.history())
        if len(ref_df)==0:
            raise ValueError('There is no potential review history for this setting')
            
        nlp_df = get_most_recent_reviews(ref_df)
        text_series = nlp_df['text']
        print(text_series.iloc[0])
        if len(text_series)==0:
            raise ValueError('There is no potential review history for this setting')
        word_count = dict()
        max_n = 10
        for row in text_series.apply(lambda x:x.split(' ')):
            for elem in row:
                word_count[elem] = word_count.get(elem, 0) + 1

        word_count = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)[:max_n]}
        
        plt.bar(range(len(word_count)), list(word_count.values()), align='center')
        plt.xticks(range(len(word_count)), list(word_count.keys()))
        # plt.set_xticklabels(axes[2].get_xticklabels(), rotation=60)
        plt.xticks(rotation = 60) 



app = App(app_ui, server)

