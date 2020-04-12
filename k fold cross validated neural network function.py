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

train = pd.read_csv("sselected.csv")        # Sample the train dataset to ensure "randomness"
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

def cross_val(train_x, train_y, folds, epochs, batch_size, hu1, hu2, hu_last, actfun1, actfun2, actfun_last, optimizer, metric, loss):
    
    #Statistics
    folds_scores_out_of_sample = []
    folds_scores_in_sample = []
    
    model = create_model(inputs=inputs, hu1=hu1, hu2=hu2, hu_last=hu_last, actfun1=actfun1, actfun2=actfun2,
                         actfun_last=actfun_last, optimizer=optimizer, metric=metric, loss=loss)
                                
    for train_index, test_index in KFold(folds).split(train_x):
       cv_train_x, cv_test_x = train_x.iloc[train_index], train_x.iloc[test_index]
       cv_train_y, cv_test_y = train_y.iloc[train_index], train_y.iloc[test_index]
       model.fit(cv_train_x, cv_train_y, epochs=epochs, batch_size=batch_size)
       folds_scores_out_of_sample.append(model.evaluate(cv_test_x,cv_test_y))
       folds_scores_in_sample.append(model.evaluate(cv_train_x,cv_train_y))

    return np.mean(folds_scores_out_of_sample), np.mean(folds_scores_in_sample)

# CV by activation functions and hidden units

def neural_network_reg_cv(train_x, train_y, inputs, hu1, hu2, hu_last, actfun, metric, loss, optimizer, folds, epochs = 20, batch_size = 20,
                          multistarts = False, multistarts_info = False):
    
    # Dictionary gathering info regarding the mae (out and in sample) corresponding to combinations of different activation functions and
    # numbers of hidden units at each layer
    summary = {}
    key = 0         # Used also as a counter of iterations
    
    # Some indicators of runtime
    start_time = datetime.now()
    iterations = hu1 * hu2 * hu_last * len(actfun) ** 3 * folds
    print(iterations)
    
    # With multistarts
    if multistarts:
        
        # If we want information for each multistart, not just the optimal models
        if multistarts_info:
            
            # Small loop
            for hui in range(1, hu1 + 1):
                for huj in range(1, hu2 + 1):
                    for huk in range(1, hu_last + 1):
                        for actfun1 in actfun:
                            for actfun2 in actfun:
                                for actfun_last in actfun:
                                    summary_values = []
                                    
                                    # Multistarts loop
                                    for start in range(1, multistarts + 1):
                                        
                                        current_best_mae_out_of_sample = 100000000000000000000000000 # Set arbitrarily high so first multistart's mae is always lower
                                        
                                        # Cross-validation
                                        stats = cross_val(train_x, train_y, folds, epochs, batch_size, hui, huj, huk, actfun1, actfun2,
                                                      actfun_last, optimizer, metric, loss)
                                        
                                        # Selection of best multistart
                                        if stats[0] >= current_best_mae_out_of_sample:
                                            continue
                                            
                                        else:
                                            current_best_mae_out_of_sample = stats[0]
                                            
                                            # Update of stats dictionary if model is better than current best
                                            summary_values.append(stats[0])
                                            summary_values.append(stats[1])
                                            summary_values.append(hui)
                                            summary_values.append(huj)
                                            summary_values.append(huk)
                                            summary_values.append(actfun1)
                                            summary_values.append(actfun2)
                                            summary_values.append(actfun_last)
    
                                    summary[key] = summary_values
                                    key = key + 1
                                    print(key, iterations / folds)
                            
            end_time = datetime.now()
            total_time = end_time - start_time
            return summary, total_time    
        
        # If we do not want information for each multistart, just the optimal models
        else:
        
            # Small loop
            for hui in range(1, hu1 + 1):
                for huj in range(1, hu2 + 1):
                    for huk in range(1, hu_last + 1):
                        for actfun1 in actfun:
                            for actfun2 in actfun:
                                for actfun_last in actfun:
                                    summary_values = ["mae_out_of_sample", "mae_in_sample", "hu1", "hu2", "hu_last", "actfun1", "actfun2", "actfun_last"]
                                    
                                    # Multistarts loop
                                    for start in range(1, multistarts + 1):
                                        
                                        current_best_mae_out_of_sample = 100000000000000000000000000 # Set arbitrarily high so first multistart's mae is always lower
                                        
                                        # Cross-validation
                                        stats = cross_val(train_x, train_y, folds, epochs, batch_size, hui, huj, huk, actfun1, actfun2,
                                                      actfun_last, optimizer, metric, loss)
                                        
                                        # Selection of best multistart
                                        if stats[0] >= current_best_mae_out_of_sample:
                                            continue
                                            
                                        else:
                                            current_best_mae_out_of_sample = stats[0]
                                            
                                            # Update of stats dictionary if model is better than current best
                                            summary_values[0] = stats[0]
                                            summary_values[1] = stats[1]
                                            summary_values[2] = hui
                                            summary_values[3] = huj
                                            summary_values[4] = huk
                                            summary_values[5] = actfun1
                                            summary_values[6] = actfun2
                                            summary_values[7] = actfun_last
    
                                    summary[key] = summary_values
                                    key = key + 1
                                    print(key, iterations / folds)
                            
            end_time = datetime.now()
            total_time = end_time - start_time
            return summary, total_time 
    
    # Without multistarts
    else:
        # Small loop
        for hui in range(1, hu1 + 1):
            for huj in range(1, hu2 + 1):
                for huk in range(1, hu_last + 1):
                    for actfun1 in actfun:
                        for actfun2 in actfun:
                            for actfun_last in actfun:
                                
                                summary_values = []
                                
                                # Cross-validation
                                stats = cross_val(train_x, train_y, folds, epochs, batch_size, hui, huj, huk, actfun1, actfun2,
                                                  actfun_last, optimizer, metric, loss)
                                
                                # Fill the summary dictionary
                                summary_values.append(stats[0])
                                summary_values.append(stats[1])
                                summary_values.append(hui)
                                summary_values.append(huj)
                                summary_values.append(huk)
                                summary_values.append(actfun1)
                                summary_values.append(actfun2)
                                summary_values.append(actfun_last)
                                summary[key] = summary_values
                                key = key + 1
                                print(key, iterations / folds)

        end_time = datetime.now()
        total_time = end_time - start_time
        return summary, total_time
                

# Model's hyperparameters // Model with 2 hidden layers, regression

actfun = ["sigmoid", "tanh", "linear"]  # Activation functions
inputs = train.shape[1] - 1             # Number of training variables
hu1 = 10                                 # 1st hidden layer's maximum units
hu2 = 5                                 # 2nd hidden layer's maximum units
hu_last = 1                             # Hidden units in output layer --> 1 for regression, more for classification
optimizer = "rmsprop"                   # "adam", "rmsprop", "nadam", "sgd", "adagrad", "adadelta" --> rmsprop faster, adam yields higher accuracy
metric = [metrics.mae]                  # Epoch's performance metric
loss = losses.mean_absolute_error       # Loss function epoch's score
folds = 2                               # Number of folds for the cross-validation
epochs = 20                             # Starting weights close to linerity, so lower amount of epochs implies activation fcts closer to linearity
batch_size = train.shape[0]             # Reduce it to reasonable levels to improve the generalisation of the models, but runtime increases multiplicatively
multistarts = 2                        # TODO (check what happens if set to 1 or 0); Set to False if no multistarts are desired. Best multistart is choosen, not the average
multistarts_info = True                 # Set to true to output mae's out and in sample for each multistart, even if they are not the optimal for their corresponding iteration
                                        # Can be used to calculate mae as an average of multistarts instead than as a min
# Results

np.random.seed(1)
cv_nn = neural_network_reg_cv(train_x=train_x, train_y=train_y, inputs=inputs, hu1=hu1, hu2=hu2, hu_last=hu_last, actfun=actfun, metric=metric,
                      loss=loss, optimizer=optimizer, folds=folds, epochs=epochs, batch_size=batch_size, multistarts=multistarts, multistarts_info=multistarts_info)
print("Your computer has been suffering for the last", cv_nn[1])

results = pd.DataFrame.from_dict(cv_nn[0])
compression_opts = dict(method='zip', archive_name='results.csv')
results.to_csv('results.zip', index=False, compression=compression_opts)
