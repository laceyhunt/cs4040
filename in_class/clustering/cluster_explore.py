import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../abalone.csv')
df = df.iloc[0:100]
print('Features') # Scale down b/c dataset is larger
# print(df.columns)

outcome = 'Rings'
features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
feature_df = df[features]

# # Look at the weights
# plt.hist(df['Shell weight'])
# plt.savefig('abalone_weights_hist.png')
# plt.clf()

# plt.scatter(df['Diameter'], df['Rings'])
# plt.savefig('diam_rings_scatter.png')


# Normalize the data
scaler = MinMaxScaler()
feature_df = scaler.fit_transform(feature_df)
print(feature_df[0])

k_clusters = 2
clustering = KMeans(n_clusters=k_clusters).fit(feature_df)
plt.scatter(df.index, clustering.labels_)
plt.savefig('cluster_test.png')
plt.clf()