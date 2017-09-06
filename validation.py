import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
%matplotlib inline

lift = test[target]
lift['Prediction'] = prediction

def liftchart_bin(lift,nbins=20,predict='Prediction',target=target):
    buckets = pd.qcut(lift[predict], nbins, labels=False)
    lift['Bucket'] = buckets
    liftchart = pd.DataFrame()
    liftchart['Avg_target'] = lift.groupby(['Bucket'])[target].mean()
    liftchart['Avg_predict'] = lift.groupby(['Bucket'])[predict].mean()
    liftchart['Bucket'] = pd.Series(range(nbins))
    return liftchart

# performance by model score bucket
liftchart = liftchart_bin(lift,50)
fig, ax1 = plt.subplots()
fig.tight_layout()
ax1.plot(liftchart['Bucket'], liftchart['Avg_predict'], 'r-')
ax1.set_xlabel('Model Bucket')
ax1.set_ylabel('Average Prediction',color='r')
ax1.tick_params('y', colors='r')
ax1.xaxis.set_ticks(np.arange(min(liftchart['Bucket']), max(liftchart['Bucket'])+1, 5.0))

# R2 by model score bucket
def R2_bin(lift,nbins=20,predict='Prediction',target=target):
    liftchart = liftchart_bin(lift,nbins)
    return r2_score(liftchart['Avg_target'],liftchart['Avg_predict'])

nbins = 100
r2_bin = R2_bin(fs_liftchart,nbins)
print 'R^2 of model on %i bins is %f' %(nbins, r2_bin)   
