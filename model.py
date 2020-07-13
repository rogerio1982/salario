# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#import dataset
dataset = pd.read_csv('hiring.csv')

#target and features
y = dataset.salary #target coluna alvo
X = dataset.drop('salary', axis=1) #todas ascolunas menos target

#Splitting Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#model
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X_train, y_train)

#acuracia
print(regressor.score(X_test,y_test))


#predictor
print(regressor.predict([[2, 9, 6]]))


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))


'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

'''
