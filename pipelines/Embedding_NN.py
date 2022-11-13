#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from functools import reduce

import YelpCleanData as cd # User Defined Fn
import warnings
warnings.simplefilter('ignore')


from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Embedding, concatenate, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K


class NN_model():
    '''
    brief notation descrioption

    m: number of training samples
    n_c: number of total classes
    n_f: number of features
    '''
    def __init__(self, num_latent, user_cnt, buss_cnt, path=None):
        self.num_latent = num_latent
        self.user_cnt = user_cnt
        self.buss_cnt = buss_cnt
        
        self.activation = 'relu'
        self.verbose = 2
        self.epochs = 15
        self.batch_size = 128
        
        self.initial_lr = 0.00005
        self.units = [512, 128, 32, 10]
        self.dropout_rate = [0.5, 0.2, 0.0, 0.0]
        self.loss = 'mse'
        self.history = None
        self.model = None

        if path is not None:
            self.load_model(path)
        elif user_cnt is not None and buss_cnt is not None:
            self.model = self.build_model(num_latent, user_cnt, buss_cnt)
        else:
            raise Exception("Arguments (input_size and num_classes) should be fed or path")
            
            
    def build_model(self, num_latent, user_cnt, buss_cnt):
        """
        Builds a neural network model architecture before complie.

        Arguments:
        num_latent: number of target embedding features, type of integer
        user_cnt: number of unique users, type of integer
        buss_cnt: number of unique businesses, type of integer

        return:
        neural network model with designed architecture, type of keras model
        """
        # User Embeddings
        u_input = Input(shape=(1,), name='user_input_layer')
        u_embedding = Embedding(input_dim=user_cnt, output_dim=num_latent, input_length=1, name='user_embedding')(u_input)
        u_vector = Flatten(name='user_vector')(u_embedding)

        # Business Embeddings
        b_input = Input(shape=(1,), name='buss_input_layer')
        b_embedding = Embedding(input_dim=buss_cnt, output_dim=num_latent, input_length=1, name='buss_embedding')(b_input)
        b_vector = Flatten(name='buss_vector')(b_embedding)

        # Concatenation
        concat = concatenate([u_vector, b_vector], name='concat_Layer')
        prev_l = concat
        for i in range(len(self.units)):
            units = self.units[i]
            dropout_rate = self.dropout_rate[i]
            
            current_l = Dense(units, activation=self.activation)(prev_l) 
            current_l = Dropout(dropout_rate)(current_l)
            current_l = BatchNormalization()(current_l)
            prev_l = current_l

        def limited_sigmoid(x, output_min=1, output_max=5) :
            x = K.sigmoid(x)
            scale = (output_max - output_min)
            return x * scale + output_min

        def relu4_1(x):
            return K.relu(x, max_value=4) + 1

        output = Dense(1, activation = limited_sigmoid)(prev_l)
        model = Model(inputs=[u_input, b_input], outputs=output)
        self.model = model

        return model


    def train_model(self, trains, vals):
        """
        Trains NN model given training dataset and save training history and trained model as instance arguments.

        Arguments:
        trains: train set, 
                first element - user_train, type of numpy array
                second element - business_train, type of numpy array
                third element - true_y, type of numpy array
        vals:   validation set
                first element - user_val, type of numpy array
                second element - business_val, type of numpy array
                third element - true_y, type of numpy array
        """
        model = self.model
#         steps_per_epoch = int(train_X_cnn.shape[0]/self.batch_size/2)
#         lr_schedule = ExponentialDecay(initial_learning_rate=self.initial_lr,
#                                         decay_steps=steps_per_epoch,
#                                         decay_rate=0.95,
#                                         staircase=True)

        user_train, buss_train, train_y = trains
        user_val, buss_val, val_y = vals
    
        callback = EarlyStopping(monitor='val_loss', patience=2)
        optimizer = Adam(lr=self.initial_lr)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=[RootMeanSquaredError()])

        # fit
        history = model.fit(x=[user_train, buss_train], y=train_y, batch_size=self.batch_size, epochs=self.epochs, 
                            verbose=2, callbacks=[callback], validation_data=([user_val, buss_val], val_y))
        self.model = model
        self.history = history


    def plot_learning_curve(self):
        """
        Plot learning curve from training history
        """
        history = self.history
        if history is None:
            return
        import matplotlib.pyplot as plt
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.ylim([0.5, 3.])
        plt.show()


    def evaluate_model(self, tests, batch_size=8192):
        """
        Evaluate model accuracy with test data

        Arguments:
        tests:  test set, 
                first element - user_test, type of numpy array
                second element - business_test, type of numpy array
                third element - true_y, type of numpy array
        batch_size: ideal number could be the power of 2, type of integer

        return:
        prediction values, numpy array of shape (m, )
        """
        model = self.model
        user_test, buss_test = tests
        preds = model.predict([user_test, buss_test], batch_size=batch_size)
        return preds


    def save_model(self, path):
        """
        Save trained model

        Arguments:
        path: directory path where you want to save the trained model, type of string
        """
        model = self.model
        model.save(path)
        print('--model saved--')


    def load_model(self, path):
        """
        Load pre-trained model

        Arguments:
        path: directory path where you want to load the pre-trained model, type of string
        """
        from tensorflow.keras import models
        self.model = models.load_model(path)

