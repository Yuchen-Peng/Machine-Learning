# H2O

code for ML using H2O, including GBM, classification, clustering


## H2O basics

```python
df = h2o.import_file(path="~/data.csv")

train = df.drop("ID")

h2o.exportFile(df, path="~/result.csv")
```
