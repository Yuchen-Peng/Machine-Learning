import pandas as pd
import numpy as np
import pyodbc
import ast
from __future__ import division
import getpass
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
import sklearn.cluster as cluster
from sklearn import tree
import xgboost as xgb

import scipy as sp
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve


bins=60
bins_for_top=20
threshold_value=100000
tail_cut=80

# sloping actual (averaged by prediction bucket) vs. true actual
def sloping_validation_chart(predict,true,nbins):
    x_indx = range(1,nbins+1)
    y_indx = []
    y_indx_2 =[]
    n = predict.shape[0]
    last_ind=0
    df = pd.DataFrame({'Predict':predict,'Actual':true})
    df=df.sort_values(by='Predict')
    for x in range(nbins):
        count = int(np.round(n / nbins * (x + 1), 0))
        subset = df.iloc[last_ind:count]
        y_indx.append(np.mean(subset['Actual']))
        last_ind = count
    return x_indx,y_indx

pred_x1,pred_y1 = sloping_validation_chart(predict_test,test_y,bins)
base_x1,base_y1 = sloping_validation_chart(benchmark_test,test_y,bins)
act_x1,act_y1 = sloping_validation_chart(test_y,test_y,bins)

plt.figure(figsize=(10,8))
plt.plot(base_x1,base_y1,color="purple",label="Benchmark")
plt.plot(pred_x1,pred_y1,color="blue",label="Model prediction")
plt.plot(act_x1,act_y1,color="darkgoldenrod",label="Actual")
plt.axhline(y=np.mean(test_y), color='grey', linestyle='--',label="Pop. Average")
plt.xlabel("Score Bucket")
plt.ylabel("Average Target")
plt.title("Sloping Chart")
plt.legend(loc='upper left')

# cumulative value
def cum_validation_chart(predict,true,nbins):
    x_indx = range(1,nbins+1)
    y_indx = []
    n = predict.shape[0]
    last_ind=0
    df = pd.DataFrame({'Predict':predict,'Actual':true})
    df=df.sort_values(by='Predict')
    for x in range(nbins):
        count = int(np.round(n / nbins * (x + 1), 0))
        subset = df.iloc[last_ind:count]
        y_indx.append(np.sum(subset['Actual']))
        last_ind = count
    y_indx_2 = np.cumsum(y_indx)/sum(y_indx)*100
    return x_indx,y_indx_2

pred_x2,pred_y2 = cum_validation_chart(predict_test,test_y,bins)
base_x2,base_y2 = cum_validation_chart(benchmark_test,test_y,bins)

plt.figure(figsize=(10,8))
plt.plot(base_x2,base_y2,color="purple",label="Benchmark")
plt.plot(pred_x2,pred_y2,color="blue",label="Model Predict")
plt.xlabel("Score Bucket")
plt.ylabel("Average Target")
plt.title("Cumulative Value Chart")
plt.legend(loc='upper left')

# cumulative gain chart
def cum_gain(predict,true,nbins,threshold):
    x_indx = range(1,nbins+1)
    y_indx = []
    n = predict.shape[0]
    last_ind=0
    ps_indicator= 1*(true>threshold)
    df = pd.DataFrame({'Predict':predict,'Actual':ps_indicator})
    df=df.sort_values(by='Predict',ascending=False)
    for x in range(nbins):
        count = int(np.round(n / nbins * (x + 1), 0))
        subset = df.iloc[last_ind:count]
        y_indx.append(np.sum(subset['Actual']))
        last_ind = count
    y_indx_2 = np.cumsum(y_indx)/float(sum(ps_indicator))*100
    return x_indx,y_indx_2

pred_x3,pred_y3 = cum_gain(predict_test,test_y,bins,threshold_value)
base_x3,base_y3 = cum_gain(benchmark_test,test_y,bins,threshold_value)

# cumulative gain chart in the highest buckets
def cum_gain_top(predict,true,nbins,threshold,top_cut):
    x_indx = range(1,nbins+1)
    y_indx = []    
    last_ind=0
    ps_indicator= 1*(true>threshold)
    df = pd.DataFrame({'Predict':predict,'Actual':ps_indicator})
    df = df[df['Predict'] >= np.percentile(df['Predict'],top_cut)]
    n = df.shape[0]
    df = df.sort_values(by='Predict',ascending=False)
    for x in range(nbins):
        count = int(np.round(n / nbins * (x + 1), 0))
        subset = df.iloc[last_ind:count]
        y_indx.append(np.sum(subset['Actual']))
        last_ind = count
    y_indx_2 = np.cumsum(y_indx)/float(sum(ps_indicator))*100
    return x_indx,y_indx_2

pred_x4,pred_y4 = cum_gain_top(predict_test,test_y,bins_for_top,threshold_value,tail_cut)


# Somers'D 
def dollar_somersd(score,target, bins=500):
    # this works for continuous target
    liftchart_somersd = liftchart_bin(score,target,bins)
    x = liftchart_somersd['Avg_target']
    y = liftchart_somersd['Avg_predict']
    tau1, p_value = sp.stats.kendalltau(x, y,False)
    #tau1, p_value = sp.stats.weightedtau(x,y)
    tau2, p_value = sp.stats.kendalltau(x, x,False)
    #tau2, p_value = sp.stats.weightedtau(x,x)
    return tau1/tau2

# ROC
plt.figure(figsize=(10,8))
ps_indicator = 1*(test_y>threshold_value)
pred_fpr,pred_tpr,pred_tt=roc_curve(ps_indicator,pred_test)
base_fpr,base_tpr,base_tt=roc_curve(ps_indicator,base_y)

plt.plot(base_fpr,base_tpr,label="benchmark",color="purple")
plt.plot(pred_fpr,pred_tpr,label="Predict",color="blue")
plt.plot([0,1],[0,1],label="Random",color="grey",linestyle ="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.title("ROC")

# value captured against actual
def fs_dollar_func(predict_actual,actual):
    value = 0
    for i in predict_actual:
        value = value +  abs(i - actual)
    return value

# cumulative value
def liftchart_bin(score, target, nbins):
    lift1 = pd.DataFrame()
    lift1['predict'] = score
    lift1['target'] = target
    buckets = pd.qcut(lift1['predict'], nbins, labels=False,duplicates='drop')
    lift1['Bucket'] = buckets
    liftchart = pd.DataFrame()
    liftchart['Avg_target'] = lift1.groupby(['Bucket'])['target'].mean()
    liftchart['Avg_predict'] = lift1.groupby(['Bucket'])['predict'].mean()
    liftchart['Bucket'] = pd.Series(range(nbins))
    return liftchart

def cum_value(score,target,bins = 500,threshold = 100000):
    x1,y1 = sloping_validation_chart(score,target,bins)
    cum_value_list = []
    cumvalue = 0
    for i in list(range(bins)):
        R = y1[bins-1-i] - threshold
        C = threshold - y1[bins-1-i]
        if R >= 0:
            cumvalue += R
        else: cumvalue -= C 
        cum_value_list.append(cumvalue)
    return np.array(cum_value_list)

# R2 by model score bucket
def R2_bin(lift,nbins=20,predict='Prediction',target=target):
    liftchart = liftchart_bin(lift,nbins)
    return r2_score(liftchart['Avg_target'],liftchart['Avg_predict'])

nbins = 100
r2_bin = R2_bin(fs_liftchart,nbins)
print 'R^2 of model on %i bins is %f' %(nbins, r2_bin)   
