import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
%matplotlib inline

# set up the build data set
df_build = pd.read_csv("./data/builddata.csv")
build = df_build[df_build[target].notnull()]

# set up the test data set
df_test = pd.read_csv("./data/testdata.csv")
test=df_test[df_test[target].notnull()]

# get rid of all the columns that were not used in building model
cols_to_drop = ['ID','Segment']
features = list(set(build.columns.tolist()) - set(cols_to_drop))

test_X =test[features]
test_y = test[target]

build_X = build[features]
build_y = build[target]

# build model
premium_indicator = np.array(1*(build_y>100000)) # classification
model = xgb.XGBClassifier(seed=10)
fitted_prob_model = model.fit(build_X,premium_indicator)

fitted_model = model.fit(build_X,build_y) # regression

importance1 = xgb.plot_importance(fitted_prob_model,max_num_features=20)
importance2 = xgb.plot_importance(fitted_prob,max_num_features=20)

# top important varables dataframe
feature_importance=pd.DataFrame({'Colnames':test_X.columns,'Importance':fitted_model.feature_importances_})
feature_importance = feature_importance.sort_values(by="Importance",ascending=False)
top_thres = 20
topvar = feature_importance[0:top_thres]
topvar

# fit to test
prob_test_predict_prob = np.array([e[1] for e in fitted_prob_model.predict_proba(test_X)]) # get probability
prob_test_predict_class = np.array(fitted_prob_model.predict(test_X)) # get class
regression_test_predict = np.array(fitted_model.predict(test_X))


