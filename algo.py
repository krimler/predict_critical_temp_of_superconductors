###############################################################################
import os
import pandas as pd
import sys
from datetime import datetime
###############################################################################
from review import set_metrics as set_metrics
###############################################################################
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
###############################################################################
# polynomial
def time():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string

class Regression:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test = y_test
    def dec_tree_2(self):
        from sklearn.tree import DecisionTreeRegressor
        r = DecisionTreeRegressor(random_state=0)
        r.fit(self.X_train, self.y_train)
        y_pred = r.predict(self.X_test)
        dict = {}
        set_metrics(y_pred, self.y_test, dict)
        return dict

    def pca(self):
        train_img, test_img, train_lbl, test_lbl = self.X_train, self.X_test, self.y_train, self.y_test
        from sklearn.preprocessing import StandardScaler
        #scaler = StandardScaler()
        # Fit on training set only.
        #scaler.fit(train_img)
        # Apply transform to both the training set and the test set.
        #train_img = scaler.transform(train_img)
        #test_img = scaler.transform(test_img)

        from sklearn.decomposition import PCA
        pca = PCA(.95)
        pca.fit(train_img)
        train_img = pca.transform(train_img)
        test_img = pca.transform(test_img)
       # from sklearn.linear_model import LogisticRegression
        #logisticRegr = LogisticRegression(solver = 'lbfgs')
        #logisticRegr.fit(train_img, train_lbl)
        #y_pred = logisticRegr.predict(test_img)
        from sklearn.tree import DecisionTreeRegressor
        r = DecisionTreeRegressor(random_state=0)
        r.fit(train_img, train_lbl)
        y_pred = r.predict(test_img)
        dict = {}
        set_metrics(y_pred, test_lbl, dict)
        return dict

    def bayes(self):
        clf = BayesianRidge(compute_score=True)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        dict = {}
        set_metrics(y_pred, self.y_test, dict)
        return dict
    def linear(self):
        clf = LinearRegression()
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        dict = {}
        set_metrics(y_pred, self.y_test, dict)
        return dict
    def svr(self):
        clf = svm.SVR()
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        dict = {}
        set_metrics(y_pred, self.y_test, dict)
        return dict
    def svr_rbf(self):
        clf = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        dict = {}
        set_metrics(y_pred, self.y_test, dict)
        return dict
    def svr_linear(self):
        clf = svm.SVR(kernel='linear', C=100, gamma='auto')
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        dict = {}
        set_metrics(y_pred, self.y_test, dict)
        return dict
    def svr_poly(self):
        clf = svm.SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        dict = {}
        set_metrics(y_pred, self.y_test, dict)
        return dict
    def ard(self):
        clf = linear_model.ARDRegression()
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        dict = {}
        set_metrics(y_pred, self.y_test, dict)
        return dict
    def random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(self.X_train, self.y_train)
        y_pred = regr.predict(self.X_test)
        dict = {}
        set_metrics(y_pred, self.y_test, dict)
        return dict

    def boost(self):
        import xgboost as xgb
        xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
        xg_reg.fit(self.X_train, self.y_train)
        y_pred  = xg_reg.predict(self.X_test)
        dict = {}
        set_metrics(y_pred, self.y_test, dict)
        return dict

    def lstm(self):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        look_back = 1
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(self.X_train, self.y_train,epochs=100, batch_size=1, verbose=2)    
        y_pred = model.predict(self.X_test)
        dict = {}
        set_metrics(y_pred, self.y_test, dict)
        return dict

    def run(self):
        dict = {}
        print('start -> ' + time())
        #dict["lstm"] = self.lstm()
        print("before random forest ->" + time())
        dict["RF"] = self.random_forest()

        dict["DecTree"] = self.dec_tree_2()
        dict["Bayes"] = self.bayes()
        dict["Linear"] = self.linear()
        dict["DecPCA"] = self.pca()
        print('before SVM -> ' + time())
        dict["SVM"] = self.svr()

        print("before xgboost ->" + time())
        dict["XGboost"] = self.boost()
        print("before rbf ->" + time())
        dict["SVM-RBF"] = self.svr_rbf()
        return dict
        print("before svm with lienar kernel regression ->" + time())
        dict["SVMLin"] = self.svr_linear()
        print("before svm with polynomial linear regression ->" + time())
        dict["SVMpoly"] = self.svr_poly()
        dict["ARD"] = self.ard()
        return dict
###############################################################################
    
 
