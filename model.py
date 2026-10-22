import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

def preprocessing(df,features):
  X = df[features].copy()

  numerical = X.select_dtypes(include=['int64', 'float64']).columns
  categorical = X.select_dtypes(include=['object']).columns

  scaler = StandardScaler()
  X[numerical] = scaler.fit_transform(X[numerical])
  encoder = LabelEncoder()
  X[categorical] = X[categorical].apply(encoder.fit_transform)

  return X

def clustering(X, n_clusters,linkage):
  model = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
  model.fit(X)
  return model.labels_

def dbscan_clustering(X, eps, min_samples):
  model = DBSCAN(eps=eps,min_samples=min_samples)
  model.fit(X)
  return model.labels_

def evaluate(X,labels):
  return silhouette_score(X, labels), davies_bouldin_score(X, labels)
