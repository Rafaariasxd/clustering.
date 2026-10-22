import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import preprocessing, clustering, evaluate

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

n_cluster = st.slider('Number of clusters',2,10,3)
linkage = st.selectbox('Select linkage',['ward','single','complete'])

X = preprocessing(df,features)
labels = clustering(X,n_clusters=n_cluster,linkage=linkage)
df['cluster'] = labels
st.subheader('Clustered Dataset')
st.dataframe(df)
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
