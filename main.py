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

class LOF():
    def __init__(self, X_train = None, X_val = None, y_train =None, y_val = None ,features = None:list,base_model = LinearRegression()):
        self.X_train  = X_train
        self.y_train = y_train
        self.X_val  = X_val
        self.y_val = y_val
        self.features = features
        self.base_model = base_model
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
