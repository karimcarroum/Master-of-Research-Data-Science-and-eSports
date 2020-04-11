### Neural Network with CV
### Karim Carroum Sanz - karim.carroum01@estudiant.upf.edu

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import losses, metrics
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import KFold

from datetime import datetime


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

    # Dictionary gathering info regarding the activation functions and number of hidden units at each layer, for every iteration
    summary = {}
    key = 0
    
    # Some indicators of runtime
    start_time = datetime.now()
    count = 0   # Updated after performing an iteration of the lowest level
    iterations = hu1 * hu2 * hu_last * len(actfun) ** 3 * folds
    print(iterations)
    
    # Results initial vectors 
    cv_mae = []
    
    # Small loop
    for hui in range(1, hu1 + 1):
        for huj in range(1, hu2 + 1):
            for huk in range(1, hu_last + 1):
                for actfun1 in actfun:
                    for actfun2 in actfun:
                        for actfun_last in actfun:
                            
                            folds_scores = []
                            summary_values = []
                            for train_index, test_index in KFold(folds).split(train_x):
                                cv_train_x, cv_test_x = train_x.iloc[train_index], train_x.iloc[test_index]
                                cv_train_y, cv_test_y = train_y.iloc[train_index], train_y.iloc[test_index]
                                model = create_model(inputs=inputs, hu1=hui, hu2=huj, hu_last=huk, actfun1=actfun1, actfun2=actfun2,
                                                     actfun_last=actfun_last, optimizer=optimizer, metric=metric, loss=loss)
                                model.fit(cv_train_x, cv_train_y, epochs=epochs, batch_size=batch_size)
                                folds_scores.append(model.evaluate(cv_test_x,cv_test_y))
                            count = count + 1
                            print(count, iterations / folds)
                            temporal_mean = np.mean(folds_scores)
                            cv_mae.append(temporal_mean)
                            
                            #Fill the summary dictionary
                            summary_values.append(temporal_mean)
                            summary_values.append(hui)
                            summary_values.append(huj)
                            summary_values.append(huk)
                            summary_values.append(actfun1)
                            summary_values.append(actfun2)
                            summary_values.append(actfun_last)
                            summary[key] = summary_values
                            key = key + 1
                              
    min_mae = np.min(cv_mae)
    end_time = datetime.now()
    total_time = end_time - start_time
    return min_mae, cv_mae, total_time, summary
                

# Model's hyperparameters // Model with 2 hidden layers, regression

actfun = ["sigmoid", "tanh", "linear"]  # Activation functions
inputs = train.shape[1] - 1             # Number of training variables
hu1 = 1                                 # 1st hidden layer's maximum units
hu2 = 1                                 # 2nd hidden layer's maximum units
hu_last = 1                             # Hidden units in output layer --> 1 for regression, more for classification
optimizer = "adam"                      # "adam", "rmsprop", "sgd"
metric = [metrics.mae]                  # Performance metric
loss = losses.mean_absolute_error       # Performance metric
folds = 2
epochs = 20
batch_size = 50

# Results

np.random.seed(1)
cv_neural_net = neural_network_reg_cv(train_x=train_x, train_y=train_y, inputs=inputs, hu1=hu1, hu2=hu2, hu_last=hu_last, actfun=actfun, metric=metric,
                      loss=loss, optimizer=optimizer, folds=folds, epochs=epochs, batch_size=batch_size)
print(cv_neural_net[[0,2]])

results = pd.DataFrame.from_dict(cv_neural_net[3], columns = ["mae", "hidunits1", "hidunits2", "hidunits_last", "actfun1", "actfun2", "actfun_last"])
results.to_csv(r'~', index = False, header=True)
