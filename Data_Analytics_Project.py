# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import sklearn

dataset = pd.read_csv('cars.csv')
X = dataset.iloc[:,1:7].values
y = dataset.iloc[:,7].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[1,3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test[:,9:] = sc.fit_transform(X_test[:,9:])
X_train[:,9:] = sc.fit_transform(X_train[:,9:])

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import math
regressor = RandomForestRegressor(n_estimators=500, random_state=1)
regressor.fit(X_train,y_train)
results = (regressor.predict(X_test))
accuracy = regressor.score(X_test,y_test)
rfmodel_r2score = r2_score(y_test, results)
MSE = mean_squared_error(y_test, results)
RMSE = math.sqrt(MSE)
