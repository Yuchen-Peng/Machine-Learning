df_build = pd.read_csv('./drv.csv',index_col = 0, sep = ',')

# Categorical to dummy
col_category = df_build.select_dtypes(include=['O']).columns.values.tolist()
if target_col in col_category:
        col_category.remove(target_col)
column_dummies = [pd.get_dummies(df_build[col], prefix=col, prefix_sep='_', dummy_na=True).iloc[:,1:] for col in col_category]
all_dummies = pd.concat(column_dummies, axis=1)

df_build = pd.concat([df_build[list(set(df_build.columns.tolist())-set(col_category))],all_dummies], axis = 1)

# missing imputation
df_build = df_build.fillna(df_build.median())

# cap and floor
#cap/floor each column between 0.01 and 0.99 quantile
df_build_new=df_build.apply(lambda col: col.clip(col.quantile(0.01), col.quantile(0.99)))
