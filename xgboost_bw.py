# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import xgboost as xgb
import numpy as np

df = pd.read_csv("us_bwt_2018q1.csv")

#%%

print(df.info())
print(df.isnull().values.any())
#no missing data
 
#%%
### Data Split
#Split the data randomly in a train and test set and could stratify after the babys gender since it is a
#significant contributor to the birthweight. However sample is so big that it might not matter
from sklearn.model_selection import train_test_split

y = df["birthweight"].copy()
x = df.drop("birthweight", axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=6)

print(x_test["sex"].value_counts() / len(x_test))
print(x_train["sex"].value_counts() / len(x_train))

#%%
### Data preparation and cleaning
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

## Pipeline for transforming the data

#define the data transformation steps for the different. Note if multiple transformations are needed for a column
#use a pipeline to define the various steps

cat_pipe = Pipeline([("ohe", OneHotEncoder())])
num_pipe = Pipeline([("st_scaler", StandardScaler())])

col_trans = ColumnTransformer([('num', num_pipe, x.select_dtypes(include=['int64', 'float64']).columns), 
                           ('cat', cat_pipe, x.select_dtypes(include=['object', 'bool']).columns)])

#%%
### Setting up the model to be used
from sklearn.linear_model import LinearRegression

xgb_reg = xgb.XGBRegressor()
lin_reg = LinearRegression()

#%%
pipe_xgb = Pipeline([("preprocessing", col_trans),
                 ("xgb", xgb_reg)])

pipe_lin = Pipeline([("preprocessing", col_trans),
                 ("lin", lin_reg)])
#%%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe_xgb, x_train, y_train, scoring= "neg_mean_squared_error",
                         cv = 10, verbose= 3)
                         
final_score = np.sqrt(np.abs(scores))

#%%
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("SD:", scores.std())
    
display_scores(final_score)

#%% compare with linear regression
scores_lin = cross_val_score(pipe_lin, x_train, y_train, scoring= "neg_mean_squared_error", cv = 10, verbose= 3)
final_score_lin = np.sqrt(np.abs(scores_lin))
display_scores(final_score_lin)
 
#%%
### Tune hyperparameters with randomized search
from sklearn.model_selection import RandomizedSearchCV

#set up the parameter dictionary 
param_grid = {
        'xgb__max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], #how many many trees after each other, high more likely to overfit
        'xgb__gamma': np.arange(0.0,0.40, 0.05),
        'xgb__learning_rate': np.arange(0.05, 0.3, 0.01),
        'xgb__subsample': np.arange(0.01, 1.0, 0.01),
        'xgb__colsample_bylevel': np.arange(0.1,1.0,0.1),
        'xgb__colsample_bytree': np.arange(0.1,1.0,0.01)
        }

random_search = RandomizedSearchCV(pipe_xgb, param_grid , scoring="neg_mean_squared_error", 
                                   n_iter = 1, cv=100, verbose = 3, n_jobs = -1)
random_search.fit(x_train, y_train) 

#%%
### test the final model on the held out test set
#investigate and save the best model
from sklearn.metrics import mean_squared_error 

print(random_search.best_params_)

final_model = random_search.best_estimator_

#%%
from scipy import stats

#Final prediction on the test set and compuation of point-wise performance estimate
final_pred = final_model.predict(x_test)
final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
print(final_rmse)

conf = 0.95
squ_err = (final_pred - y_test) ** 2

#95% confidence band for the prediction error
print(np.sqrt(stats.t.interval(conf, len(squ_err)-1, loc=squ_err.mean(), scale = stats.sem(squ_err))))


#%%
### Save the model
import joblib

joblib.dump(final_model, "xgb_bw_model.pkl")