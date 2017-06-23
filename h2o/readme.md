# H2O

code for ML using H2O, including GBM, classification, clustering


## H2O basics

```python
import h2o
import pandas as pd
import seaborn as sns
import imp
import math
import numpy as np

h2o.init()
```

```python
df = h2o.import_file(path="~/data.csv")

build = df.drop("ID")

train,test,valid = build.split_frame(ratios=[.6, .2]) # split

h2o.exportFile(build, path="~/result.csv")
```

## Build a GBM
```python
model = H2OGradientBoostingEstimator(distribution="bernoulli", 
                                   ntrees=100, 
                                   max_depth=3, 
                                   learn_rate=0.01)
model.train(x = list_of_predictors, 
          y = target, 
          training_frame  = train,
          validation_frame= valid)
          
# predicting & performance on test file
model_pred = model.predict(test)
print("GBM predictions: ")
model_pred.head()

model_perf = model.model_performance(test)
print("GBM performance: ")
model_perf.show()
```
