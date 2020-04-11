### Neural Network with CV
### Karim Carroum Sanz - karim.carroum01@estudiant.upf.edu

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import losses, backend as K, metrics
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import KFold


# Import train set

train = pd.read_csv("sselected.csv")
train = train.iloc[:,1:train.shape[1]]
train_x = train.iloc[:,1:train.shape[1]]
train_y = train.iloc[:,0]


# Model creation function

def create_model(inputs, hu1, hu2, hu_last, actfun1, actfun2, actfun_last, optimizer, metric, loss):
    model = Sequential()
    model.add(Dense(hu1, input_dim=inputs, activation=actfun1))
    model.add(Dense(hu2, activation=actfun2))
    model.add(Dense(hu_last, activation=actfun_last))
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)                                           
    return model


# CV by activation functions and hidden units

def neural_network_reg_cv(train_x, train_y, inputs, hu1, hu2, hu_last, actfun, metric, loss, optimizer, folds, epochs = 20, batch_size = 20):
    
    cv_mae = [] # np.zeros(len[actfun] ** 3 * hu1 * hu2 * hu_last
    
    for hui in range(1, hu1 + 1):
        for huj in range(1, hu2 + 1):
            for huk in range(1, hu_last + 1):
                for actfun1 in actfun:
                    for actfun2 in actfun:
                        for actfun_last in actfun:
                            folds_scores = []
                            for train_index, test_index in KFold(folds).split(train_x):
                                cv_train_x, cv_test_x = train_x.iloc[train_index], train_x.iloc[test_index]
                                cv_train_y, cv_test_y = train_y.iloc[train_index], train_y.iloc[test_index]
                                model = create_model(inputs=inputs, hu1=hui, hu2=huj, hu_last=huk, actfun1=actfun1, actfun2=actfun2,
                                                     actfun_last=actfun_last, optimizer=optimizer, metric=metric, loss=loss)
                                model.fit(cv_train_x, cv_train_y, epochs=epochs, batch_size=batch_size)
                                folds_scores.append(model.evaluate(cv_test_x,cv_test_y))
                            cv_mae.append(np.mean(folds_scores))
    print("Lowest" + folds + "- fold CV mean absolute error is " + np.min(cv_mae))
    return np.min(cv_mae), cv_mae
                

# Model's hyperparameters // Model with 2 hidden layers, regression

actfun = ["sigmoid", "tanh", "linear"]  # Activation functions
inputs = train.shape[1] - 1             # Number of training variables
hu1 = 2                                 # 1st hidden layer's maximum units
hu2 = 2                                 # 2nd hidden layer's maximum units
hu_last = 1                             # Hidden units in output layer --> 1 for regression, more for classification
optimizer = "adam"                      # "adam", "rmsprop", "sgd"
metric = [metrics.mae]                  # Performance metric
loss = losses.mean_absolute_error       # Performance metric
folds = 10
epochs = 20
batch_size = 20

# Results

neural_network_reg_cv(train_x=train_x, train_y=train_y, inputs=inputs, hu1=hu1, hu2=hu2, hu_last=hu_last, actfun=actfun, metric=metric,
                      loss=loss, optimizer=optimizer, folds=folds, epochs=epochs, batch_size=batch_size)