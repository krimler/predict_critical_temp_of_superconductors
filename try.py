import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
import os
import sys
import sklearn.metrics as mets
from review import set_metrics as set_metrics
from algo import Regression
import draw
#https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#https://datascienceplus.com/keras-regression-based-neural-networks/

#xgboost
#random forest
#lstm
#rnn
#dec tree
#logistic regression
#ann
#naive bayes
#monte carlo

def read_atomic_data(path):
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        print("To begin with, your path to data should be proper!")
        sys.exit(1)
    df = pd.read_csv(path)
    columns = df.columns.tolist() # get the columns
    columns = columns[:-1]
    df = pd.read_csv(path, usecols=columns)
    return df, columns

def get_dataset(df, columns):
    X = df[col[:-1]]
    y = df.critical_temp
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
    return (X_train, X_test, y_train, y_test)

df, col = read_atomic_data("unique_m.csv")
(X_train, X_test, y_train, y_test) = get_dataset(df, col)
from sklearn import preprocessing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
results  = {}
R = Regression(X_train, X_test, y_train, y_test)
dict = R.run()
print (dict)
draw.draw(dict, 'r2_score')
draw.draw(dict, 'max_error')
draw.draw(dict, 'explained_variance_score')
draw.draw(dict, 'mean_absolute_error')
draw.draw(dict, 'mean_squared_error')
draw.draw(dict, 'mean_squared_log_error')
draw.draw(dict, 'median_absolute_error')

sys.exit()
