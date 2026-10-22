import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import preprocessing, clustering, dbscan_clustering, evaluate

st.title('Clustering App')

file = st.file_uploader('upload your dataset',type=['csv'])

if file is not None:
    df = pd.read_csv(file)
else:
    st.stop()

st.subheader('Dataset Preview')
st.dataframe(df.head())

features = st.multiselect('Select the features for clustering',df.columns.to_list())

if len(features) < 2:
    st.warning('Select at least 2 features')
    st.stop()

algorithm = st.selectbox('Select clustering algorithm',['Hierarchical Clustering','DBSCAN'])

X = preprocessing(df,features)
if algorithm == 'Hierarchical Clustering':
    n_cluster = st.slider('Number of clusters',2,10,3)
    linkage = st.selectbox('Select linkage',['ward','single','complete'])
    labels = clustering(X,n_clusters=n_cluster,linkage=linkage)
else:
    eps = st.slider('eps',0.1,5.0,0.5,0.1)
    min_samples = st.slider('min_samples',2,20,5)
    labels = dbscan_clustering(X,eps=eps,min_samples=min_samples)
df['cluster'] = labels
st.subheader('Clustered Dataset')
st.dataframe(df)
if len(set(labels)) < 2:
    st.warning('Only one cluster found, try other parameters')
    st.stop()
silhoutte, davis = evaluate(X,labels)
st.write('Silhouette',silhoutte)
st.write('Davis Index',davis)


st.subheader('Charts')

st.markdown('**Clusters in two selected features**')
col_x = st.selectbox('X axis', features, index=0)
col_y = st.selectbox('Y axis', features, index=1)
fig, ax = plt.subplots()
scatter = ax.scatter(X[col_x], X[col_y], c=labels, cmap='viridis')
ax.set_xlabel(col_x)
ax.set_ylabel(col_y)
ax.legend(*scatter.legend_elements(), title='Cluster')
st.pyplot(fig)
