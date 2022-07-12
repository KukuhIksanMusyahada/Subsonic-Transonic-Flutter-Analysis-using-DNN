import os
import numpy as np
import pickle

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Reshape, Dropout

from Essential import path_handler as ph
from Essential import global_params as gp


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


def model_flutter(x_train, x_val, y_train, y_val, max_epoch= 30, n_features= 4):
    model = Sequential([
        Dense(500, activation=my_leaky_relu, input_shape= (1,2)),
        Dense(500, activation=my_leaky_relu),
        Dropout(.5),
        Dense(500, activation=my_leaky_relu),
        Dropout(.3),
        Dense(n_features*gp.NROWS, kernel_initializer=tf.initializers.zeros()),
        Reshape([gp.NROWS,n_features])

    ])
    model.compile(loss="mean_squared_error" , optimizer=tf.keras.optimizers.RMSprop(learning_rate= 1e-7))

    history = model.fit(x_train, y_train, epochs= max_epoch,
                        validation_data=(x_val, y_val), verbose=1)

    return model, history


def model_non_flutter(X_train, X_val, y_train, y_val, max_epoch= 300, n_features= 4):
    model = Sequential([
        Dense(500, activation='relu', input_shape= (1,2)),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(n_features*gp.NROWS),
        Reshape((gp.NROWS,n_features))

    ])
    model.compile(loss='mse', optimizer='adam')

    history = model.fit(X_train,y_train, epochs= max_epoch, 
                        validation_data=(X_val, y_val),verbose=1)

    return model, history


def model_transonic(X_train, X_val, y_train, y_val, max_epoch= 300, n_features= 4):
    model = Sequential([
        Dense(500, activation='relu', input_shape= (1,2)),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(n_features*gp.NROWS),
        Reshape((gp.NROWS,n_features))

    ])
    model.compile(loss='mse', optimizer='adam')

    history = model.fit(X_train,y_train, epochs= max_epoch, 
                        validation_data=(X_val, y_val),verbose=1)

    return model, history


def savemodel(model, history, optional_path: str=None):
    """Save both model and history"""
    nomor_model = str(len(os.listdir(ph.get_models_data()))+1)
    folder_name = "ModelFlutterClassification" + nomor_model
    if optional_path != None:
        model_directory = os.path.join (optional_path, folder_name)
    else:
        model_directory = os.path.join (ph.get_models_data(), folder_name)
    os.makedirs(model_directory)
    history_file = os.path.join(model_directory, 'history.pkl')


    model.save(model_directory)
    print ("\nModel saved to {}".format(model_directory))

    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print ("Model history saved to {}".format(history_file))


def loadmodel(path_to_model):
    """Load Model and optionally it's history as well"""
    history_file = os.path.join(path_to_model, 'history.pkl')
    model = tf.keras.models.load_model(path_to_model)
    # model = tf.saved_model.load(path_to_model)
    print ("\nmodel loaded")

    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    print ("model history loaded")

    return model, history


def predict(model, mach= None, vf=None):
    input = [[mach, vf]]
    pred = model.predict(input)
    pred = np.round(pred[0][0])

    return pred