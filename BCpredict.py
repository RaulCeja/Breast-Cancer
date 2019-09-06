# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:34:34 2019

@author: Raul Ceja
"""
import pandas as pd
import tensorflow as tf
from tensorflow import keras


data = pd.read_csv("breast-cancer.csv")
#split data into train_X and train_Y
#train_Y is used to train train_X
print(data.head())
train_X = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
#if you're not me and you're reading this, I didn't use the column drop function
#because there is an invisible column of NaN data that is messing with the keras model 
train_Y = data[['diagnosis']]

print(train_X.isnull().any())
#convert diagnoses into boolean
train_Y.loc[train_Y.diagnosis == 'M','diagnosis'] = 1
train_Y.loc[train_Y.diagnosis == 'B','diagnosis'] = 0
#this works, ignore the warning

#normalize data function
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
#normalize float data
train_X = normalize(train_X)


def train_model(train_frame, num_epochs):
    n_cols = train_frame.shape[1]
    #create new dataframe from test_data with the same columns from training_data
    #test_frame = test_Y[train_frame.columns]
    model = keras.Sequential()
    #using tanh activation because the data has been rescaled to [-1,1]
    model.add(keras.layers.Dense(len(train_frame.columns), activation='tanh',input_shape=(n_cols,)))#, 
                             #batch_size=len(dataframe.columns)))
    #model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.15))
    model.add(keras.layers.Dense(50, activation='tanh'))
    model.add(keras.layers.Dropout(0.15))
    model.add(keras.layers.Dense(200, activation='tanh'))
    model.add(keras.layers.Dropout(0.15))
    model.add(keras.layers.Dense(len(train_Y.columns)))
    
    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics = ['accuracy'])
    model.fit(train_frame, train_Y, validation_split = 0.2, epochs = num_epochs, shuffle = True)
    
    test_Y_predictions = model.predict(train_frame)
    #outputs predictions to csv for better viewing
    #nColumns = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
    tYp = pd.DataFrame(test_Y_predictions, columns = ['diagnosis'])#, columns=nColumns)
    #tYp.to_csv('predictions.csv')
    
    return tYp.head()

def random_test_hyperparameters(dataframe,iterations, columns, num_epochs):
    parameters = (dataframe.sample(n=columns,axis=1))
    for i in range(0,iterations):
        train_model(parameters, num_epochs)
        print(parameters.columns)