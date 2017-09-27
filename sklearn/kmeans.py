from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

#data clean: missing imputation and scale
features = build_X.columns.tolist()
build_X_clean = build_X.fillna(build_X.median())
build_X_normalized = MinMaxScaler().fit(build_X_clean).transform(build_X_clean)  
build_X_normalized = pd.DataFrame(data = build_X_normalized, columns = features)

# build Kmean
kmeans = KMeans(n_clusters = 10, random_state = 0)
kmeans.fit(build_X_normalized)

cluster = kmeans.predict(build_X_normalized)
build_X['Cluster'] = cluster

# the centroid for each cluster
cluster_center = pd.DataFrame(data = kmeans.cluster_centers_, columns = features)

# Test the optimal cluster
