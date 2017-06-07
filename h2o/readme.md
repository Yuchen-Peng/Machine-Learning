# H2O

code for ML using H2O, including GBM, classification, clustering


## H2O basics

```python
df = h2o.import_file(path="~/data.csv")

train = df.drop("ID")

h2o.exportFile(df, path="~/result.csv")
```

```python
'''
Build a GBM
'''

model = H2OGradientBoostingEstimator(distribution="bernoulli", 
                                   ntrees=100, 
                                   max_depth=3, 
                                   learn_rate=0.01)
model.train(x = list_of_predictors, 
          y = target, 
          training_frame  = df_train,
          validation_frame= df_valid)
          
# predicting & performance on test file
model_pred = model.predict(df_test)
print("GBM predictions: ")
model_pred.head()

model_perf = model.model_performance(air_test)
print("GBM performance: ")
model_perf.show()
```
