import os
import numpy as np
import pickle

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Reshape, Dropout

from Essential import path_handler as ph
from Essential import global_params as gp



def set_low_tf_verbose():
    tf.get_logger().setLevel('ERROR')


def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

def model_classification(x_train, x_val, y_train, y_val, max_epoch= 300):
    model = Sequential([
        Dense(300, activation='relu', input_dim = 2),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')

    history = model.fit(x_train, y_train, epochs= max_epoch,
                        validation_data=(x_val, y_val), verbose=0)

    return model, history


def model_flutter(x_train, x_val, y_train, y_val,max_row, max_epoch= 300, n_features= 4):
    model = Sequential([
        Dense(500, activation='relu', input_dim= 2),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(n_features*max_row),
        Reshape([max_row,n_features])

    ])
    model.compile(loss=tf.keras.losses.huber , optimizer='adam')

    history = model.fit(x_train, y_train, epochs= max_epoch,
                        validation_data=(x_val, y_val), verbose=0)

    return model, history


def model_non_flutter(x_train, x_val, y_train, y_val,max_row, max_epoch= 300, n_features= 4):
    model = Sequential([
        Dense(500, activation='relu', input_dim= 2),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(n_features*max_row),
        Reshape([max_row,n_features])

    ])
    model.compile(loss=tf.keras.losses.huber , optimizer='adam')

    history = model.fit(x_train, y_train, epochs= max_epoch,
                        validation_data=(x_val, y_val), verbose=0)

    return model, history


def model_transonic(x_train, x_val, y_train, y_val,max_row, max_epoch= 300, n_features= 4):
    model = Sequential([
        Dense(500, activation='relu', input_dim= 2),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(n_features*max_row),
        Reshape([max_row,n_features])

    ])
    model.compile(loss=tf.keras.losses.huber , optimizer='adam')

    history = model.fit(x_train, y_train, epochs= max_epoch,
                        validation_data=(x_val, y_val), verbose=0)

    return model, history


def savemodel(model, history,type_case=4 , model_path: str=ph.get_models_master()):
    """Save both model and history"""
    
    if model_path != None:
        nomor_model = str(len(os.listdir(model_path))+1)
        folder_name = gp.CASE[type_case] + nomor_model
        model_directory = os.path.join (model_path, folder_name)
    else:
        nomor_model = str(len(os.listdir(ph.get_models_master()))+1)
        folder_name = gp.CASE[type_case] + nomor_model
        model_directory = os.path.join (ph.get_models_master(), folder_name)
    os.makedirs(model_directory)
    history_file = os.path.join(model_directory, 'history.pkl')


    model.save(model_directory)
    print ("\nModel saved to {}".format(model_directory))

    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print ("Model history saved to {}".format(history_file))


def load_model(path, type_case, num_model):
    """Load Model and optionally it's history as well"""

    folder_name = gp.CASE[type_case] + str(num_model)

    path_to_model = os.path.join(path,folder_name)
    history_file = os.path.join(path_to_model, 'history.pkl')
    model = tf.keras.models.load_model(path_to_model)
    # model = tf.saved_model.load(path_to_model)
    print ("\nmodel loaded")

    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    print ("model history loaded")

    return model, history


def predict_class(model, mach, vf):
    input = [[mach, vf]]
    pred = model.predict(input)
    pred = np.round(pred[0][0])

    return pred


def predict_non_class(model, mach, vf):
    input = [[mach, vf]]
    pred = model.predict(input)
    pred = np.squeeze(pred)

    return pred