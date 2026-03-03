import pandas as pd 
import numpy as np 
from scipy import stats
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import xgboost
import lightgbm
import catboost
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import accuracy_score

class SCORE():
    def __init__(self, method, score):
        self.method = method
        self.score = score
    def to_return(self):
        return {self.name}, {self.score}



class LOF():
    def __init__(self, X_train = None, X_val = None, y_train =None, y_val = None ,features = None:list,base_model = LinearRegression(), metric = accuracy_score):
        self.X_train  = X_train
        self.y_train = y_train
        self.X_val  = X_val
        self.y_val = y_val
        self.features = features
        self.base_model = base_model
        self.metric = metric
    def IQR(self):
        Q1 = X_train[self.features].quantile(0.25)
        Q3 = X_train[self.features].quantile(0.75)

        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 - 1.5*IQR
        self.X_train =self.X_train[(self.X_train[self.features] >=lower) &(self.X_train[self.features] <=upper)]
        self.X_val = self.X_val[(self.X_val[self.features] >=lower) & (self.X_val[self.features] <=upper)]
        return self.X_train, self.X_val
    def z_score(self):
        z_scores = np.abs(stats.zscore(X))
        t = 3
        self.X_train[self.features]  = self.X_train[self.features][z_scores <=t]
        self.X_val[self.features] = self.X_val[self.features][z_scores <=t]
        return self.X_train, self.X_val
    def forward(self):
        [X_train, X_val] = z_score()

        model = base_model()
        scores =[]
        # Iqr
        [X_train, X_val] = IQR()
        model.fit(X_train, self.y_train)
        iqr_score = self.metric(model.predict(X_val), self.y_val)
        iqr_score = SCORE("IQR", iqr_score)
        scores.append(iqr_score)
        ##z_score
        [X_train, X_val] = z_score()
        model.fit(X_train, self.y_train)
        z_score_score = self.metric(model.predict(X_val), self.y_val)
        z_score_score = SCORE("Z_score", z_score_score)
        scores.append(z_score_score)


        scores.sort(key = lambda: method :method.score)
    
